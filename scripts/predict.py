#!/usr/bin/env python3
"""
Phase 7 entry-point: predict stock direction using a trained XGBoost model.

Loads a saved model artifact and runs inference on an input CSV, writing
the predictions (direction label + per-class probabilities) to an output CSV.

Run from the project root
-------------------------
    # Predict on an ML dataset CSV
    python scripts/predict.py \
        --input data/ml/ml_dataset_2026-06-16.csv

    # Specify model and output paths explicitly
    python scripts/predict.py \
        --model  artifacts/models/xgboost_direction_model.joblib \
        --input  data/ml/ml_dataset_2026-06-16.csv \
        --output predictions/predictions_2026-06-16.csv

Output
------
    CSV file identical to --input with four extra columns appended:
        predicted_direction   — "BUY", "HOLD", or "SELL"
        prob_BUY              — probability score for BUY
        prob_HOLD             — probability score for HOLD
        prob_SELL             — probability score for SELL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.model_io import ModelNotFoundError, ModelIOError
from src.model.predictor import ModelPredictor
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = _PROJECT_ROOT / "artifacts" / "models" / "xgboost_direction_model.joblib"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict stock direction (BUY / HOLD / SELL) using a trained "
            "XGBoost model artifact (Phase 7)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        metavar="PATH",
        help="Path to the trained model artifact (.joblib)",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to input CSV containing feature columns",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Path to write predictions CSV "
            "(default: <input_stem>_predictions.csv in the same directory)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    model_path = Path(args.model)
    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_predictions.csv"

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 7: Predict")
    logger.info("=" * 60)
    logger.info(f"  Model  : {model_path}")
    logger.info(f"  Input  : {input_path}")
    logger.info(f"  Output : {output_path}")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    predictor = ModelPredictor(model_path=model_path)

    try:
        predictor.load_model()
        result_df = predictor.predict_from_csv(input_path)
    except (ModelNotFoundError, ModelIOError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except ValueError as exc:
        logger.error(f"Feature mismatch: {exc}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    if "predicted_direction" in result_df.columns:
        counts = result_df["predicted_direction"].value_counts()
        print("\n" + "─" * 60)
        print("  PREDICTION SUMMARY")
        print("─" * 60)
        print(f"  Input rows   : {len(result_df)}")
        for direction in ["BUY", "HOLD", "SELL"]:
            n = int(counts.get(direction, 0))
            pct = n / len(result_df) * 100 if len(result_df) else 0
            print(f"  {direction:<6}        : {n:>4}  ({pct:.1f}%)")
        print(f"  Output saved : {output_path}")
        print("─" * 60 + "\n")

    logger.info(f"Predictions written → {output_path}")


if __name__ == "__main__":
    main()
