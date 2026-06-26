#!/usr/bin/env python3
"""
Phase 7 entry-point: train an XGBoost stock direction classifier.

Loads a Phase 6 ML dataset, engineers features, trains a three-class
XGBoost classifier (BUY / HOLD / SELL), evaluates on a held-out test
split, and writes the model artifact + evaluation metrics + feature
importance to ``artifacts/``.

Run from the project root
-------------------------
    # Auto-detect the latest ML dataset in data/ml/
    python scripts/train_model.py

    # Explicit dataset path
    python scripts/train_model.py \
        --dataset data/ml/ml_dataset_2025-01-01_2026-06-17.csv

    # Custom artifact paths
    python scripts/train_model.py \
        --model-out   artifacts/models/my_model.joblib \
        --metrics-out artifacts/metrics/my_metrics.json \
        --importance-out artifacts/plots/my_importance.png

    # Change random seed
    python scripts/train_model.py --random-seed 123

    # Dry-run: print configuration and exit without training
    python scripts/train_model.py --dry-run

Output
------
    artifacts/models/xgboost_direction_model.joblib
    artifacts/metrics/xgboost_metrics.json
    artifacts/plots/feature_importance.png
    artifacts/plots/feature_importance.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.trainer import (
    DataPreparationError,
    ModelNotTrainedError,
    ModelTrainer,
    ModelTrainingError,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

# ── Default artifact paths ─────────────────────────────────────────────────────

_DEFAULT_MODEL_OUT      = _PROJECT_ROOT / "artifacts" / "models"  / "xgboost_direction_model.joblib"
_DEFAULT_METRICS_OUT    = _PROJECT_ROOT / "artifacts" / "metrics" / "xgboost_metrics.json"
_DEFAULT_IMPORTANCE_OUT = _PROJECT_ROOT / "artifacts" / "plots"   / "feature_importance.png"
_DEFAULT_ML_DIR         = _PROJECT_ROOT / "data" / "ml"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_latest_dataset(ml_dir: Path) -> Path | None:
    """Return the most recently modified ``ml_dataset_*.csv``, or ``None``."""
    csvs = sorted(
        ml_dir.glob("ml_dataset_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return csvs[0] if csvs else None


def _print_summary(trainer: ModelTrainer, model_out: Path) -> None:
    """Print a structured Phase 7 training summary to stdout."""
    m = trainer.metrics
    if m is None:
        return

    print("\n" + "=" * 60)
    print("  Phase 7: XGBoost Model Training")
    print("=" * 60)
    print(f"  Dataset rows    : {trainer.n_total}")
    print(f"  Feature columns : {len(trainer.feature_columns)}")
    print(f"  Train rows      : {trainer.n_train}")
    print(f"  Test rows       : {trainer.n_test}")
    print(f"  Accuracy        : {m['accuracy']:.4f}")
    print(f"  Macro F1        : {m['f1']['macro']:.4f}")
    print(f"  Macro Precision : {m['precision']['macro']:.4f}")
    print(f"  Macro Recall    : {m['recall']['macro']:.4f}")
    print(f"  Model saved     : {model_out}")
    print("=" * 60)
    print()
    print(m["classification_report"])

    cm     = m["confusion_matrix"]
    labels = m["labels"]
    print("  Confusion matrix (rows=true, cols=predicted):")
    header = "  " + "".join(f"  {lbl:<6}" for lbl in labels)
    print(header)
    for i, row in enumerate(cm):
        cells = "".join(f"  {v:<6}" for v in row)
        print(f"  {labels[i]:<6}{cells}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost stock direction classifier (Phase 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help=(
            "Path to an ML dataset CSV (default: auto-detect latest from data/ml/). "
            "Supports both single-date and historical backfill datasets."
        ),
    )
    parser.add_argument(
        "--model-out",
        default=str(_DEFAULT_MODEL_OUT),
        metavar="PATH",
        dest="model_out",
        help="Output path for the trained model artifact (.joblib)",
    )
    parser.add_argument(
        "--metrics-out",
        default=str(_DEFAULT_METRICS_OUT),
        metavar="PATH",
        dest="metrics_out",
        help="Output path for the evaluation metrics JSON",
    )
    parser.add_argument(
        "--importance-out",
        default=str(_DEFAULT_IMPORTANCE_OUT),
        metavar="PATH",
        dest="importance_out",
        help="Output path for the feature importance plot (.png)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        metavar="N",
        dest="random_seed",
        help="Random seed for the train/test split and XGBoost",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit without training",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    model_out      = Path(args.model_out)
    metrics_out    = Path(args.metrics_out)
    importance_out = Path(args.importance_out)

    # Resolve dataset path.
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = _find_latest_dataset(_DEFAULT_ML_DIR)
        if dataset_path is None:
            logger.error(
                "No ML dataset found in data/ml/. "
                "Run scripts/build_ml_dataset.py first to generate one."
            )
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 7: XGBoost Training")
    logger.info("=" * 60)
    logger.info(f"  Dataset        : {dataset_path}")
    logger.info(f"  Model out      : {model_out}")
    logger.info(f"  Metrics out    : {metrics_out}")
    logger.info(f"  Importance out : {importance_out}")
    logger.info(f"  Random seed    : {args.random_seed}")
    logger.info(f"  Dry-run        : {args.dry_run}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without training.")
        sys.exit(0)

    trainer = ModelTrainer(
        dataset_path=dataset_path,
        model_out=model_out,
        metrics_out=metrics_out,
        importance_out=importance_out,
        random_seed=args.random_seed,
    )

    try:
        trainer.load_dataset()
        trainer.prepare_features()
        trainer.train()
        trainer.evaluate()
        trainer.save_model()
        _print_summary(trainer, model_out)
    except DataPreparationError as exc:
        logger.error(f"Data preparation failed: {exc}")
        sys.exit(1)
    except ModelNotTrainedError as exc:
        logger.error(f"Model operation failed: {exc}")
        sys.exit(1)
    except ModelTrainingError as exc:
        logger.error(f"Training error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
