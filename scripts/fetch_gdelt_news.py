#!/usr/bin/env python3
"""
Phase 1B entry-point script: fetch historical financial news via GDELT and save to CSV.

GDELT (Global Database of Events, Language, and Tone) provides free,
unlimited historical news coverage indexed from thousands of sources
worldwide — making it ideal for generating large ML training datasets.

Run from the project root
--------------------------
    # Backfill a full calendar year for all configured tickers
    python scripts/fetch_gdelt_news.py \\
        --start-date 2025-01-01 \\
        --end-date   2025-12-31

    # Specific tickers only
    python scripts/fetch_gdelt_news.py \\
        --start-date 2025-01-01 \\
        --end-date   2025-06-30 \\
        --tickers AAPL NVDA

    # Limit articles per chunk request (default: 250)
    python scripts/fetch_gdelt_news.py \\
        --start-date 2025-01-01 \\
        --end-date   2025-03-31 \\
        --max-records 100

    # Dry-run: validate config and exit without making API calls
    python scripts/fetch_gdelt_news.py \\
        --start-date 2025-01-01 \\
        --end-date   2025-12-31 \\
        --dry-run

Output
------
  data/raw/gdelt/<TICKER>_gdelt_<start>_<end>.csv   one file per ticker
  data/raw/gdelt/gdelt_summary_<start>_<end>.csv    one-row-per-ticker summary

Output schema (identical to Finnhub Phase 1 output)
----------------------------------------------------
  ticker, source_id, source_name, author, title, description,
  url, published_at, content, fetched_at

FinBERT / load_to_db.py compatibility
--------------------------------------
GDELT CSVs share the exact same column schema as Finnhub CSVs.  To run
sentiment analysis on GDELT articles use the FinBERTSentimentAnalyzer
directly::

    from src.processing.sentiment_analyzer import FinBERTSentimentAnalyzer
    import pandas as pd

    df = pd.read_csv("data/raw/gdelt/AAPL_gdelt_2025-01-01_2025-12-31.csv")
    analyzer = FinBERTSentimentAnalyzer()
    analyzer.load()
    enriched = analyzer.analyse_dataframe(df)

Then load the enriched CSV into PostgreSQL with ``scripts/load_to_db.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.gdelt_client import (
    GDELT_MAX_RECORDS,
    GDELTAPIError,
    GDELTError,
    GDELTValidationError,
    fetch_all_tickers,
    summarise_results,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "raw" / "gdelt"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical financial news from GDELT and save to CSV. "
            "Use this for large-scale ML training data backfills."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date of the backfill window (inclusive)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="End date of the backfill window (inclusive)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help=(
            "Ticker symbols to fetch (default: uses TICKERS from .env or "
            "AAPL, TSLA, NVDA, MSFT, AMZN)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help="Directory to write output CSVs",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=GDELT_MAX_RECORDS,
        metavar="N",
        help=f"Max articles per GDELT sub-request (hard limit: {GDELT_MAX_RECORDS})",
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
        help="Print resolved configuration and exit without making API calls",
    )
    return parser.parse_args()


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date_arg(date_str: str, param_name: str) -> datetime:
    """Parse a ``YYYY-MM-DD`` argument into a UTC-aware datetime.

    Raises ``SystemExit`` with a human-readable message on bad input.
    """
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.replace(tzinfo=UTC)
    except ValueError:
        logger.error(
            f"Invalid {param_name} '{date_str}'. "
            "Expected format: YYYY-MM-DD (e.g. 2025-01-01)"
        )
        sys.exit(1)


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_results(
    results: dict,
    output_dir: Path,
    start_tag: str,
    end_tag: str,
) -> None:
    """Write one CSV per ticker plus a summary CSV to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in results.items():
        if df.empty:
            logger.warning(f"[{ticker}] Skipping save — no data returned")
            continue
        out = output_dir / f"{ticker}_gdelt_{start_tag}_{end_tag}.csv"
        df.to_csv(out, index=False)
        logger.info(
            f"[{ticker}] Saved {len(df)} rows → "
            f"{out.relative_to(_PROJECT_ROOT)}"
        )

    summary_df = summarise_results(results)
    summary_path = output_dir / f"gdelt_summary_{start_tag}_{end_tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(
        f"Summary saved → {summary_path.relative_to(_PROJECT_ROOT)}"
    )


def _print_summary(results: dict) -> None:
    summary = summarise_results(results)
    print("\n" + "─" * 64)
    print("  GDELT FETCH RESULTS SUMMARY")
    print("─" * 64)
    for _, row in summary.iterrows():
        status_marker = "✓" if row["status"] == "ok" else "✗"
        print(
            f"  {status_marker}  {row['ticker']:<6}  "
            f"{row['article_count']:>5} articles  "
            f"{row['unique_sources']:>2} sources"
        )
    total = int(summary["article_count"].sum())
    print("─" * 64)
    print(
        f"     TOTAL: {total} articles across "
        f"{len(results)} tickers"
    )
    print("─" * 64 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    from_dt = _parse_date_arg(args.start_date, "--start-date")
    to_dt   = _parse_date_arg(args.end_date,   "--end-date")

    if from_dt > to_dt:
        logger.error(
            f"--start-date ({args.start_date}) must be before "
            f"--end-date ({args.end_date})"
        )
        sys.exit(1)

    tickers    = [t.upper() for t in args.tickers] if args.tickers else settings.tickers
    output_dir = Path(args.output_dir)
    max_rec    = min(args.max_records, GDELT_MAX_RECORDS)
    start_tag  = from_dt.strftime("%Y-%m-%d")
    end_tag    = to_dt.strftime("%Y-%m-%d")

    logger.info("=" * 64)
    logger.info("financial-news-analytics | Phase 1B: GDELT Historical Backfill")
    logger.info("=" * 64)
    logger.info(f"  Tickers    : {tickers}")
    logger.info(f"  Start date : {start_tag}")
    logger.info(f"  End date   : {end_tag}")
    logger.info(f"  Max records: {max_rec} per chunk")
    logger.info(f"  Output dir : {output_dir.relative_to(_PROJECT_ROOT)}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without making API calls.")
        sys.exit(0)

    try:
        results = fetch_all_tickers(
            tickers=tickers,
            from_date=from_dt,
            to_date=to_dt,
            max_records=max_rec,
        )
    except GDELTValidationError as exc:
        logger.error(f"Configuration error: {exc}")
        sys.exit(1)
    except GDELTAPIError as exc:
        logger.error(f"GDELT API error: {exc}")
        sys.exit(1)
    except GDELTError as exc:
        logger.error(f"Unexpected GDELT error: {exc}")
        sys.exit(1)

    _print_summary(results)
    _save_results(results, output_dir, start_tag, end_tag)


if __name__ == "__main__":
    main()
