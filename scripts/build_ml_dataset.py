#!/usr/bin/env python3
"""
Phase 6 entry-point: build a supervised ML dataset for model training.

Combines Phase 4 engineered features with future stock-price labels loaded
from PostgreSQL to produce a complete, training-ready dataset with binary
and multi-class direction labels.

Run from the project root
-------------------------
    # Build dataset for today using the matching feature CSV
    python scripts/build_ml_dataset.py

    # Specific date
    python scripts/build_ml_dataset.py --date 2026-06-16

    # Historical backfill — supply a multi-date feature CSV directly
    python scripts/build_ml_dataset.py \
        --feature-file data/features/feature_dataset_2025-01-01_2026-06-01.csv

    # Custom output directory
    python scripts/build_ml_dataset.py --output-dir /tmp/ml

    # Extend the lookahead price window (default: 14 calendar days)
    python scripts/build_ml_dataset.py --lookahead-days 21

    # Dry-run: print resolved configuration and exit
    python scripts/build_ml_dataset.py --dry-run

Output
------
  data/ml/ml_dataset_<YYYY-MM-DD>.csv              single-date mode
  data/ml/ml_dataset_<start>_<end>.csv             date-range mode (--feature-file)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ml.dataset_builder import (
    DataLoadError,
    LabelGenerationError,
    MLDatasetBuilder,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
OUTPUT_DIR   = _PROJECT_ROOT / "data" / "ml"

_LABEL_COLS = {
    "future_close_1d", "future_close_3d", "future_close_5d", "future_close_7d",
    "return_1d", "return_3d", "return_5d", "return_7d",
    "label_up_1d", "label_up_3d", "label_up_5d", "label_up_7d",
    "label_direction",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a supervised ML dataset by combining Phase 4 features "
            "with future stock-price labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Target feature date (default: today). "
            "Cannot be combined with --feature-file."
        ),
    )
    # ── Date-range backfill (additive, mutually exclusive with --date) ────────
    parser.add_argument(
        "--feature-file",
        default=None,
        metavar="PATH",
        dest="feature_file",
        help=(
            "Explicit path to a feature CSV (enables range mode). "
            "The date range is inferred from the CSV's 'date' column. "
            "Cannot be combined with --date."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        dest="output_dir",
        help=f"Directory to write the ML CSV (default: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=14,
        metavar="N",
        dest="lookahead_days",
        help=(
            "Calendar days beyond the feature date(s) to load from stock_prices "
            "for lookahead label computation"
        ),
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
        help="Print resolved configuration and exit without accessing any data",
    )
    return parser.parse_args()


# ── Console summaries ─────────────────────────────────────────────────────────

def _print_range_summary(
    dataset_df,           # noqa: ANN001
    features_path: Path,
    out_path: Path | None,
) -> None:
    import pandas as pd  # noqa: PLC0415

    print("\n" + "─" * 80)
    print("  ML DATASET BUILDER SUMMARY  (date-range backfill mode)")
    print("─" * 80)

    if dataset_df.empty:
        print("  (no ML dataset rows generated)")
        print("─" * 80 + "\n")
        return

    label_col_count = sum(1 for c in dataset_df.columns if c in _LABEL_COLS)
    feature_count   = len(dataset_df.columns) - label_col_count - 2

    dates_with_data = dataset_df["date"].nunique() if "date" in dataset_df.columns else "N/A"
    buy_count  = int((dataset_df.get("label_direction") == "BUY").sum())
    sell_count = int((dataset_df.get("label_direction") == "SELL").sum())
    hold_count = int((dataset_df.get("label_direction") == "HOLD").sum())
    null_count = int(dataset_df.get("label_direction", pd.Series([])).isna().sum())

    min_date = dataset_df["date"].min() if "date" in dataset_df.columns else "N/A"
    max_date = dataset_df["date"].max() if "date" in dataset_df.columns else "N/A"

    print(f"  Feature file : {features_path.name}")
    print(f"  Date range   : {min_date} → {max_date}  ({dates_with_data} date(s) with data)")
    ticker_count = dataset_df["ticker"].nunique() if "ticker" in dataset_df.columns else "N/A"
    print(f"  Tickers      : {ticker_count}")
    print(f"  Total rows   : {len(dataset_df)}")
    print(f"  Features/row : {feature_count}")
    print(f"  Label cols   : {label_col_count}")
    print(
        f"  Directions   : BUY={buy_count}  SELL={sell_count}  "
        f"HOLD={hold_count}  NULL={null_count}"
    )
    if out_path:
        print(f"  Saved        → {out_path.relative_to(_PROJECT_ROOT)}")
    print("─" * 80 + "\n")


def _print_summary(
    dataset_df,  # noqa: ANN001
    date_tag: str,
    out_path: Path | None,
) -> None:
    import pandas as pd  # noqa: PLC0415

    print("\n" + "─" * 80)
    print("  ML DATASET BUILDER SUMMARY")
    print("─" * 80)

    if dataset_df.empty:
        print("  (no ML dataset rows generated)")
        print("─" * 80 + "\n")
        return

    label_col_count = sum(1 for c in dataset_df.columns if c in _LABEL_COLS)
    feature_count   = len(dataset_df.columns) - label_col_count - 2  # minus ticker + date

    buy_count  = int((dataset_df.get("label_direction") == "BUY").sum())
    sell_count = int((dataset_df.get("label_direction") == "SELL").sum())
    hold_count = int((dataset_df.get("label_direction") == "HOLD").sum())

    for _, row in dataset_df.iterrows():
        ticker    = row["ticker"]
        ret_1d    = row.get("return_1d")
        ret_5d    = row.get("return_5d")
        direction = row.get("label_direction", "N/A")

        ret_1d_str = f"{ret_1d:+.4f}" if pd.notna(ret_1d) else "N/A"
        ret_5d_str = f"{ret_5d:+.4f}" if pd.notna(ret_5d) else "N/A"
        print(
            f"  ✓  {ticker:<6}  "
            f"ret_1d={ret_1d_str}  "
            f"ret_5d={ret_5d_str}  "
            f"direction={direction}"
        )

    print("─" * 80)
    print(
        f"     {len(dataset_df)} tickers  |  "
        f"{feature_count} features  |  "
        f"{label_col_count} label columns  |  "
        f"BUY={buy_count}  SELL={sell_count}  HOLD={hold_count}  |  "
        f"date={date_tag}"
    )
    if out_path:
        print(f"     Saved → {out_path.relative_to(_PROJECT_ROOT)}")
    print("─" * 80 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # ── Determine mode ────────────────────────────────────────────────────────
    is_range_mode = args.feature_file is not None

    if is_range_mode and args.date:
        logger.error("--date cannot be combined with --feature-file")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 6: ML Dataset Builder")
    logger.info("=" * 60)
    logger.info(f"  Database     : {_safe_url(settings.database_url or '')}")
    logger.info(f"  Mode         : {'date-range backfill' if is_range_mode else 'single-date'}")
    logger.info(f"  Output dir   : {out_dir.relative_to(_PROJECT_ROOT)}")
    logger.info(f"  Lookahead    : {args.lookahead_days} calendar days")
    logger.info(f"  Dry-run      : {args.dry_run}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without accessing any data.")
        sys.exit(0)

    if not settings.database_url:
        logger.error(
            "DATABASE_URL is not set. "
            "Add it to your .env file:\n"
            "  DATABASE_URL=postgresql://user:password@localhost:5432/financial_news"
        )
        sys.exit(1)

    builder = MLDatasetBuilder(database_url=settings.database_url)

    try:
        if is_range_mode:
            # ── Date-range backfill mode ──────────────────────────────────────
            features_path = Path(args.feature_file)
            logger.info(f"  Feature file : {features_path}")

            dataset_df = builder.run_range(
                features_path=features_path,
                output_dir=out_dir,
                lookahead_days=args.lookahead_days,
            )

            out_path: Path | None = None
            if not dataset_df.empty and "date" in dataset_df.columns:
                from datetime import date  # noqa: PLC0415

                valid = dataset_df["date"].dropna()
                if not valid.empty:
                    min_d = min(valid)
                    max_d = max(valid)
                    fmt = "%Y-%m-%d"
                    start_tag = min_d.strftime(fmt) if hasattr(min_d, "strftime") else str(min_d)
                    end_tag   = max_d.strftime(fmt) if hasattr(max_d, "strftime") else str(max_d)
                    date_tag  = start_tag if start_tag == end_tag else f"{start_tag}_{end_tag}"
                    out_path  = out_dir / f"ml_dataset_{date_tag}.csv"

            _print_range_summary(dataset_df, features_path, out_path)

        else:
            # ── Single-date mode (existing behaviour — unchanged) ─────────────
            from datetime import date  # noqa: PLC0415

            date_tag      = args.date or datetime.now(UTC).strftime("%Y-%m-%d")
            features_path = FEATURES_DIR / f"feature_dataset_{date_tag}.csv"

            logger.info(f"  Feature date : {date_tag}")
            logger.info(f"  Features file: {features_path.relative_to(_PROJECT_ROOT)}")

            try:
                target_date = date.fromisoformat(date_tag)
            except ValueError:
                logger.error(f"Invalid date format: {date_tag!r}  (expected YYYY-MM-DD)")
                sys.exit(1)

            dataset_df = builder.run(
                features_path=features_path,
                target_date=target_date,
                output_dir=out_dir,
                lookahead_days=args.lookahead_days,
            )

            out_path = (out_dir / f"ml_dataset_{date_tag}.csv") if not dataset_df.empty else None
            _print_summary(dataset_df, date_tag, out_path)

    except DataLoadError as exc:
        logger.error(f"Fatal: data load failed — {exc}")
        sys.exit(1)
    except LabelGenerationError as exc:
        logger.error(f"Fatal: label generation failed — {exc}")
        sys.exit(1)
    finally:
        builder.dispose()


def _safe_url(url: str) -> str:
    """Return the URL with any password redacted for safe logging."""
    try:
        from sqlalchemy.engine.url import make_url  # noqa: PLC0415
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return "<db_url>"


if __name__ == "__main__":
    main()
