#!/usr/bin/env python3
"""
Phase 1 entry-point script: fetch financial news via Finnhub and save to CSV.

Run from the project root:

    python scripts/fetch_news.py                     # all configured tickers
    python scripts/fetch_news.py --tickers AAPL TSLA # specific tickers
    python scripts/fetch_news.py --days 3            # narrow date window

Output
------
  data/raw/<TICKER>_news_<YYYY-MM-DD>.csv   one file per ticker
  data/raw/summary_<YYYY-MM-DD>.csv         one-row-per-ticker summary
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

# Allow running as `python scripts/fetch_news.py` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.news_client import (
    FinnhubAuthError,
    FinnhubError,
    fetch_all_tickers,
    summarise_results,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

OUTPUT_DIR = _PROJECT_ROOT / "data" / "raw"


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch financial news via Finnhub and save to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to fetch (overrides TICKERS in .env)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Days back to fetch (overrides NEWS_LOOKBACK_DAYS in .env; "
            "Finnhub free tier supports up to 365 days)"
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
        help="Print config and exit without making any API calls",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    summary = summarise_results(results)
    print("\n" + "─" * 60)
    print("  RESULTS SUMMARY")
    print("─" * 60)
    for _, row in summary.iterrows():
        status_marker = "✓" if row["status"] == "ok" else "✗"
        print(
            f"  {status_marker}  {row['ticker']:<6}  "
            f"{row['article_count']:>4} articles  "
            f"{row['unique_sources']:>2} sources"
        )
    total = summary["article_count"].sum()
    print("─" * 60)
    print(f"     TOTAL: {int(total)} articles across {len(results)} tickers")
    print("─" * 60 + "\n")


def _save_results(results: dict, date_tag: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, df in results.items():
        if df.empty:
            logger.warning(f"[{ticker}] Skipping save — no data")
            continue
        out = OUTPUT_DIR / f"{ticker}_news_{date_tag}.csv"
        df.to_csv(out, index=False)
        logger.info(f"[{ticker}] Saved {len(df)} rows → {out.relative_to(_PROJECT_ROOT)}")

    summary_df = summarise_results(results)
    summary_path = OUTPUT_DIR / f"summary_{date_tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved → {summary_path.relative_to(_PROJECT_ROOT)}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    tickers = [t.upper() for t in args.tickers] if args.tickers else settings.tickers
    days    = args.days or settings.news_lookback_days

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 1: News Collection")
    logger.info("=" * 60)
    logger.info(f"  Tickers    : {tickers}")
    logger.info(f"  Lookback   : {days} days")
    logger.info(f"  Output dir : {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without making API calls.")
        sys.exit(0)

    date_tag = datetime.now(UTC).strftime("%Y-%m-%d")

    try:
        results = fetch_all_tickers(tickers=tickers, days_back=days)
    except FinnhubAuthError as exc:
        logger.error(f"Fatal: {exc}")
        sys.exit(1)
    except FinnhubError as exc:
        logger.error(f"Unexpected error: {exc}")
        sys.exit(1)

    _print_summary(results)
    _save_results(results, date_tag)


if __name__ == "__main__":
    main()
