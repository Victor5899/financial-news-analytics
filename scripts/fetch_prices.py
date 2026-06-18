#!/usr/bin/env python3
"""
Phase 5 entry-point: ingest daily stock prices from Yahoo Finance.

Downloads historical daily OHLCV data via ``yfinance`` and upserts it into
the ``stock_prices`` table in PostgreSQL.  Idempotent — re-running with the
same arguments refreshes existing rows in-place.

Run from the project root
-------------------------
    # Create the stock_prices table (safe on existing DBs — uses IF NOT EXISTS)
    python scripts/fetch_prices.py --create-tables

    # Fetch one year of history for all tickers defined in .env
    python scripts/fetch_prices.py --lookback-days 365

    # Specific tickers with a fixed date range
    python scripts/fetch_prices.py --tickers AAPL TSLA NVDA \\
        --start-date 2025-01-01 --end-date 2026-01-01

    # Dry-run: fetch data from Yahoo Finance but skip all database writes
    python scripts/fetch_prices.py --tickers AAPL --lookback-days 30 --dry-run

    # One-shot: create tables then populate
    python scripts/fetch_prices.py --create-tables --lookback-days 365

Output
------
  PostgreSQL database table:
    stock_prices   — deduplicated by (ticker, trading_date)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

UTC = timezone.utc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.prices.price_client import (
    PriceFetchError,
    PriceValidationError,
    YFinancePriceClient,
)
from src.prices.price_repository import PriceRepository
from src.storage.database import DatabaseConnectionError, DatabaseManager, SchemaError
from src.storage.repository import UpsertResult
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TickerPriceResult:
    """Holds per-ticker fetch-and-store outcome for the summary table."""

    ticker:       str
    rows_fetched: int
    upsert:       UpsertResult
    date_min:     Optional[date] = field(default=None)
    date_max:     Optional[date] = field(default=None)
    status:       str = "ok"
    error:        Optional[str] = field(default=None)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch daily OHLCV stock prices from Yahoo Finance "
            "and store them in PostgreSQL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to fetch (default: tickers from TICKERS in .env)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        dest="start_date",
        help="Inclusive start date for price history (default: today minus --lookback-days)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        dest="end_date",
        help="End date for price history (default: today)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        metavar="N",
        dest="lookback_days",
        help="Days of history to fetch when --start-date is not provided",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Run CREATE TABLE IF NOT EXISTS before fetching (safe on existing DBs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch data from Yahoo Finance but skip all database writes. "
            "Useful for validating the pipeline without modifying the DB."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# ── Per-ticker fetch + store ──────────────────────────────────────────────────

def _fetch_and_store_ticker(
    ticker: str,
    client: YFinancePriceClient,
    db: DatabaseManager,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
    lookback_days: int,
    dry_run: bool,
) -> TickerPriceResult:
    """
    Fetch OHLCV data for one ticker and (optionally) upsert into the DB.

    Never raises — all errors are caught and stored in the result object
    so the caller can process remaining tickers.
    """
    logger.info(f"[{ticker}] Starting price ingestion")

    try:
        rows = client.fetch_prices(
            ticker,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
        )
    except (PriceFetchError, PriceValidationError) as exc:
        logger.error(f"[{ticker}] Fetch failed: {exc}")
        return TickerPriceResult(
            ticker=ticker,
            rows_fetched=0,
            upsert=UpsertResult(),
            status="error",
            error=str(exc),
        )

    if not rows:
        logger.warning(f"[{ticker}] No rows returned — skipping")
        return TickerPriceResult(
            ticker=ticker,
            rows_fetched=0,
            upsert=UpsertResult(),
            status="empty",
        )

    trading_dates = [r["trading_date"] for r in rows]
    date_min = min(trading_dates)
    date_max = max(trading_dates)
    logger.info(
        f"[{ticker}] Fetched {len(rows)} rows  ({date_min} → {date_max})"
    )

    if dry_run:
        logger.info(f"[{ticker}] Dry-run — skipping database write")
        return TickerPriceResult(
            ticker=ticker,
            rows_fetched=len(rows),
            upsert=UpsertResult(),
            date_min=date_min,
            date_max=date_max,
            status="dry_run",
        )

    try:
        with db.get_session() as session:
            repo = PriceRepository(session, dialect_name=db.engine.dialect.name)
            upsert_result = repo.upsert_prices(rows)
        logger.info(f"[{ticker}] DB upsert — {upsert_result}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[{ticker}] Database error: {exc}")
        return TickerPriceResult(
            ticker=ticker,
            rows_fetched=len(rows),
            upsert=UpsertResult(),
            date_min=date_min,
            date_max=date_max,
            status="error",
            error=str(exc),
        )

    return TickerPriceResult(
        ticker=ticker,
        rows_fetched=len(rows),
        upsert=upsert_result,
        date_min=date_min,
        date_max=date_max,
    )


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(results: list[TickerPriceResult]) -> None:
    print("\n" + "─" * 80)
    print("  PRICE INGESTION SUMMARY")
    print("─" * 80)
    print(f"  {'Ticker':<6}  {'Rows':<6}  {'Date Range':<26}  Status")
    print("  " + "─" * 56)

    for r in results:
        if r.status in ("ok", "dry_run"):
            date_range = f"{r.date_min} → {r.date_max}"
            if r.status == "dry_run":
                status_str = f"[dry-run]  {r.rows_fetched} rows not written"
            else:
                status_str = (
                    f"+{r.upsert.inserted} inserted  "
                    f"upd={r.upsert.updated}"
                )
            marker = "✓"
            print(
                f"  {marker}  {r.ticker:<6}  {r.rows_fetched:<6}  "
                f"{date_range:<26}  {status_str}"
            )
        elif r.status == "empty":
            print(f"  ~  {r.ticker:<6}  (no data returned from Yahoo Finance)")
        else:
            print(f"  ✗  {r.ticker:<6}  ERROR: {r.error}")

    ok_count  = sum(1 for r in results if r.status in ("ok", "dry_run"))
    total_rows = sum(r.rows_fetched for r in results)
    print("─" * 80)
    print(
        f"     {ok_count}/{len(results)} tickers  |  "
        f"{total_rows} total rows fetched"
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

    tickers: list[str] = (
        [t.upper() for t in args.tickers]
        if args.tickers
        else list(settings.tickers)
    )

    start_date: Optional[date] = None
    end_date:   Optional[date] = None

    if args.start_date:
        try:
            start_date = date.fromisoformat(args.start_date)
        except ValueError:
            logger.error(
                f"Invalid --start-date: {args.start_date!r}  (expected YYYY-MM-DD)"
            )
            sys.exit(1)

    if args.end_date:
        try:
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            logger.error(
                f"Invalid --end-date: {args.end_date!r}  (expected YYYY-MM-DD)"
            )
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 5: Stock Price Ingestion")
    logger.info("=" * 60)
    logger.info(f"  Database     : {_safe_url(settings.database_url)}")
    logger.info(f"  Tickers      : {tickers}")
    logger.info(
        f"  Start date   : {start_date or f'today - {args.lookback_days} days'}"
    )
    logger.info(f"  End date     : {end_date or 'today'}")
    logger.info(f"  Lookback     : {args.lookback_days} days")
    logger.info(f"  Dry-run      : {args.dry_run}")

    if args.dry_run:
        logger.info(
            "Dry-run mode — prices will be fetched from Yahoo Finance "
            "but no rows will be written to the database."
        )

    db = DatabaseManager(settings.database_url)

    try:
        db.verify_connection()
    except DatabaseConnectionError as exc:
        logger.error(f"Fatal: {exc}")
        sys.exit(1)

    if args.create_tables:
        try:
            db.create_tables()
        except SchemaError as exc:
            logger.error(f"Fatal: {exc}")
            sys.exit(1)

    client = YFinancePriceClient()
    results: list[TickerPriceResult] = []

    for ticker in tickers:
        result = _fetch_and_store_ticker(
            ticker,
            client,
            db,
            start_date=start_date,
            end_date=end_date,
            lookback_days=args.lookback_days,
            dry_run=args.dry_run,
        )
        results.append(result)

    _print_summary(results)
    db.dispose()


def _safe_url(url: str) -> str:
    """Return the URL with any password redacted for safe logging."""
    try:
        from sqlalchemy.engine.url import make_url  # noqa: PLC0415
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return "<db_url>"


if __name__ == "__main__":
    main()
