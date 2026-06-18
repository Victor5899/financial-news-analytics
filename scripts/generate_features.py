#!/usr/bin/env python3
"""
Phase 4 entry-point: generate ML features from PostgreSQL sentiment data.

Reads joined ``news_articles`` + ``sentiment_results`` from PostgreSQL
(populated by Phase 3), computes per-ticker per-day feature vectors, and
writes a ready-to-use CSV to ``data/features/``.

Run from the project root
-------------------------
    # Features for today (all tickers in the database)
    python scripts/generate_features.py

    # Specific tickers
    python scripts/generate_features.py --tickers AAPL TSLA NVDA

    # Specific date
    python scripts/generate_features.py --date 2026-06-16

    # Custom output directory
    python scripts/generate_features.py --output-dir /tmp/features

    # Extend history window for rolling features (default: 7 days)
    python scripts/generate_features.py --lookback-days 14

    # Dry-run: print config and exit without touching the database
    python scripts/generate_features.py --dry-run

Output
------
  data/features/feature_dataset_<YYYY-MM-DD>.csv   one row per ticker
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.feature_engineer import (
    DataLoadError,
    FeatureEngineer,
    FeatureGenerationError,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

OUTPUT_DIR = _PROJECT_ROOT / "data" / "features"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ML feature dataset from PostgreSQL sentiment data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to process (default: all tickers found in the database)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Target date for feature generation (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        dest="output_dir",
        help=f"Directory to write the CSV (default: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        metavar="N",
        dest="lookback_days",
        help="Days of history to load for rolling features",
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
        help="Print resolved configuration and exit without accessing the database",
    )
    return parser.parse_args()


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(features_df, date_tag: str) -> None:  # noqa: ANN001
    print("\n" + "─" * 80)
    print("  FEATURE GENERATION SUMMARY")
    print("─" * 80)

    if features_df.empty:
        print("  (no features generated)")
        print("─" * 80 + "\n")
        return

    for _, row in features_df.iterrows():
        ticker = row["ticker"]
        n      = int(row["article_count"])
        pos_r  = float(row["positive_ratio"])
        mean_s = float(row["mean_sentiment_score"])
        vol_7d = int(row.get("rolling_7d_article_volume", 0))
        print(
            f"  ✓  {ticker:<6}  "
            f"{n:>4} articles  "
            f"pos_ratio={pos_r:.3f}  "
            f"mean_score={mean_s:+.4f}  "
            f"7d_vol={vol_7d}"
        )

    total_articles = int(features_df["article_count"].sum())
    print("─" * 80)
    print(
        f"     {len(features_df)} tickers  |  "
        f"{total_articles} total articles on {date_tag}  |  "
        f"{len(features_df.columns) - 2} features per ticker"
    )
    print("─" * 80 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    if not settings.database_url:
        logger.error(
            "DATABASE_URL is not set. "
            "Add it to your .env file:\n"
            "  DATABASE_URL=postgresql://user:password@localhost:5432/financial_news"
        )
        sys.exit(1)

    tickers  = [t.upper() for t in args.tickers] if args.tickers else None
    date_tag = args.date or datetime.now(UTC).strftime("%Y-%m-%d")
    out_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 4: Feature Engineering")
    logger.info("=" * 60)
    logger.info(f"  Database     : {_safe_url(settings.database_url)}")
    logger.info(f"  Target date  : {date_tag}")
    logger.info(f"  Tickers      : {tickers or 'all (auto-discover)'}")
    logger.info(f"  Lookback     : {args.lookback_days} days")
    logger.info(f"  Output dir   : {out_dir.relative_to(_PROJECT_ROOT)}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without accessing the database.")
        sys.exit(0)

    from datetime import date  # noqa: PLC0415

    try:
        target_date = date.fromisoformat(date_tag)
    except ValueError:
        logger.error(f"Invalid date format: {date_tag!r}  (expected YYYY-MM-DD)")
        sys.exit(1)

    eng = FeatureEngineer(database_url=settings.database_url)

    try:
        features_df = eng.run(
            tickers=tickers,
            target_date=target_date,
            output_dir=out_dir,
            lookback_days=args.lookback_days,
        )
    except DataLoadError as exc:
        logger.error(f"Fatal: database load failed — {exc}")
        sys.exit(1)
    except FeatureGenerationError as exc:
        logger.error(f"Fatal: feature generation failed — {exc}")
        sys.exit(1)
    finally:
        eng.dispose()

    _print_summary(features_df, date_tag)


def _safe_url(url: str) -> str:
    """Return the URL with any password redacted for safe logging."""
    try:
        from sqlalchemy.engine.url import make_url  # noqa: PLC0415
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return "<db_url>"


if __name__ == "__main__":
    main()
