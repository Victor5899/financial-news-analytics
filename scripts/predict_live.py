#!/usr/bin/env python3
"""
Phase 8 entry-point: live stock direction prediction from Finnhub news.

Fetches the latest company news, runs FinBERT sentiment, engineers features,
loads the trained XGBoost model, predicts BUY/HOLD/SELL, and stores the
result in PostgreSQL.

Run from the project root
-------------------------
    python scripts/predict_live.py --ticker AAPL

    python scripts/predict_live.py \\
        --ticker TSLA \\
        --model artifacts/models/xgboost_direction_model.joblib \\
        --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.model_io import ModelIOError, ModelNotFoundError
from src.realtime.finnhub_client import FinnhubConfigError, FinnhubError
from src.realtime.realtime_pipeline import (
    NoNewsError,
    PredictionResult,
    RealtimePipeline,
    RealtimePipelineError,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = _PROJECT_ROOT / "artifacts" / "models" / "xgboost_direction_model.joblib"

_SENTIMENT_LABEL_DISPLAY = {
    "positive": "Positive",
    "neutral":  "Neutral",
    "negative": "Negative",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live stock direction prediction from Finnhub news "
            "(Phase 8: Real-Time Prediction Engine)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        required=True,
        metavar="SYMBOL",
        help="Stock ticker symbol (e.g. AAPL)",
    )
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        metavar="PATH",
        help="Path to the trained XGBoost model artifact (.joblib)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the prediction to PostgreSQL",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def _print_result(result: PredictionResult, *, saved: bool) -> None:
    """Pretty-print the prediction result to the terminal."""
    label_display = _SENTIMENT_LABEL_DISPLAY.get(
        result.sentiment_label.lower(), result.sentiment_label.title()
    )

    print()
    print("=" * 50)
    print("Live Prediction")
    print("=" * 50)
    print()
    print(f"Ticker: {result.ticker}")
    print()
    print("Headline:")
    print(result.headline)
    print()
    print("Sentiment:")
    print(f"{label_display} ({result.sentiment_confidence:.2f})")
    print()
    print("Prediction:")
    print(result.prediction)
    print()
    print("Confidence:")
    print(f"{result.confidence * 100:.1f}%")
    print()
    print("Probabilities")
    print()
    for label in ("BUY", "HOLD", "SELL"):
        prob = result.probabilities.get(label, 0.0)
        print(f"{label:<5}: {prob * 100:.1f}%")
    print()
    if saved:
        print("Saved to PostgreSQL")
    print()
    print("=" * 50)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    ticker = args.ticker.upper()
    model_path = Path(args.model)

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 8: Live Prediction")
    logger.info("=" * 60)
    logger.info(f"  Ticker : {ticker}")
    logger.info(f"  Model  : {model_path}")
    logger.info(f"  Save   : {not args.no_save}")

    try:
        pipeline = RealtimePipeline(model_path=model_path)
        result = pipeline.predict(ticker, save=not args.no_save)
    except FinnhubConfigError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except FinnhubError as exc:
        logger.error(f"Finnhub error: {exc}")
        sys.exit(1)
    except NoNewsError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except (ModelNotFoundError, ModelIOError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    except RealtimePipelineError as exc:
        logger.error(str(exc))
        sys.exit(1)

    _print_result(result, saved=not args.no_save)


if __name__ == "__main__":
    main()
