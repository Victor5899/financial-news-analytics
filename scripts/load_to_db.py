#!/usr/bin/env python3
"""
Phase 3 entry-point: load Phase 2 sentiment CSVs into PostgreSQL.

Reads ``data/processed/<TICKER>_sentiment_<date>.csv`` files produced by
Phase 2, upserts the news-article rows into ``news_articles``, then links
the sentiment scores into ``sentiment_results``.

Run from the project root
-------------------------
    # Load all processed CSVs for today
    python scripts/load_to_db.py

    # Specific tickers
    python scripts/load_to_db.py --tickers AAPL TSLA

    # Specific date
    python scripts/load_to_db.py --date 2026-06-16

    # Create tables then load (safe on existing DB — uses IF NOT EXISTS)
    python scripts/load_to_db.py --create-tables

    # Override model name stored in sentiment_results
    python scripts/load_to_db.py --model-name ProsusAI/finbert

    # Dry-run: parse config + find files, exit before touching the DB
    python scripts/load_to_db.py --dry-run

Output
------
  PostgreSQL database tables:
    news_articles      — deduplicated by url
    sentiment_results  — deduplicated by (article_id, model_name)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UTC = timezone.utc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.storage.database import DatabaseConnectionError, DatabaseManager, SchemaError
from src.storage.repository import ArticleRepository, UpsertResult
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

INPUT_DIR = _PROJECT_ROOT / "data" / "processed"


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TickerLoadResult:
    ticker:    str
    articles:  UpsertResult
    sentiment: UpsertResult
    status:    str = "ok"
    error:     str | None = None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Phase 2 sentiment CSVs into PostgreSQL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to load (default: auto-discover all processed CSVs)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Date tag of the processed CSVs to load (default: today)",
    )
    parser.add_argument(
        "--model-name",
        default=settings.finbert_model,
        dest="model_name",
        help="Model name to record in sentiment_results.model_name",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Run CREATE TABLE IF NOT EXISTS before loading (safe on existing DBs)",
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
        help="Print resolved config and exit without touching the database",
    )
    return parser.parse_args()


# ── Input discovery ───────────────────────────────────────────────────────────

def _find_input_files(
    tickers: list[str] | None,
    date_tag: str,
) -> list[tuple[str, Path]]:
    """Return ``(ticker, path)`` pairs for processed sentiment CSVs."""
    if not INPUT_DIR.exists():
        logger.error(
            f"Input directory not found: {INPUT_DIR.relative_to(_PROJECT_ROOT)}. "
            "Run scripts/run_sentiment.py first."
        )
        return []

    if tickers:
        pairs: list[tuple[str, Path]] = []
        for ticker in tickers:
            path = INPUT_DIR / f"{ticker}_sentiment_{date_tag}.csv"
            if path.exists():
                pairs.append((ticker, path))
            else:
                logger.warning(f"[{ticker}] Input file not found: {path.name}")
        return pairs

    discovered: list[tuple[str, Path]] = []
    for path in sorted(INPUT_DIR.glob(f"*_sentiment_{date_tag}.csv")):
        ticker = path.stem.split("_sentiment_")[0]
        discovered.append((ticker, path))

    if not discovered:
        logger.warning(
            f"No sentiment CSVs found for date '{date_tag}' in "
            f"{INPUT_DIR.relative_to(_PROJECT_ROOT)}. "
            "Run scripts/run_sentiment.py first."
        )
    return discovered


# ── Per-ticker loader ─────────────────────────────────────────────────────────

def _load_ticker(
    ticker: str,
    csv_path: Path,
    db: DatabaseManager,
    model_name: str,
) -> TickerLoadResult:
    """
    Load a single ticker's processed CSV into the database.

    Reads the CSV, upserts articles, then upserts linked sentiment rows —
    all within one session (one transaction per ticker for atomicity).
    """
    logger.info(f"[{ticker}] Reading {csv_path.name}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[{ticker}] Failed to read CSV: {exc}")
        return TickerLoadResult(
            ticker=ticker,
            articles=UpsertResult(),
            sentiment=UpsertResult(),
            status="error",
            error=str(exc),
        )

    if df.empty:
        logger.warning(f"[{ticker}] CSV is empty — skipping")
        return TickerLoadResult(
            ticker=ticker,
            articles=UpsertResult(),
            sentiment=UpsertResult(),
            status="empty",
        )

    rows = df.to_dict(orient="records")
    logger.info(f"[{ticker}] Loaded {len(rows)} rows from CSV")

    try:
        with db.get_session() as session:
            repo = ArticleRepository(
                session,
                dialect_name=db.engine.dialect.name,
            )

            article_result, url_to_id = repo.upsert_articles(rows)
            logger.info(
                f"[{ticker}] Articles — {article_result} "
                f"| mapped {len(url_to_id)} IDs"
            )

            sentiment_result = repo.upsert_sentiment_results(
                url_to_id=url_to_id,
                records=rows,
                model_name=model_name,
            )
            logger.info(f"[{ticker}] Sentiment — {sentiment_result}")

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[{ticker}] Database error: {exc}")
        return TickerLoadResult(
            ticker=ticker,
            articles=UpsertResult(),
            sentiment=UpsertResult(),
            status="error",
            error=str(exc),
        )

    return TickerLoadResult(
        ticker=ticker,
        articles=article_result,
        sentiment=sentiment_result,
    )


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(results: list[TickerLoadResult]) -> None:
    print("\n" + "─" * 80)
    print("  DATABASE LOAD SUMMARY")
    print("─" * 80)
    for r in results:
        marker = "✓" if r.status == "ok" else "✗"
        if r.status in ("ok", "empty"):
            print(
                f"  {marker}  {r.ticker:<6}  "
                f"articles: +{r.articles.inserted} upd={r.articles.updated} "
                f"skip={r.articles.skipped}  |  "
                f"sentiment: +{r.sentiment.inserted} upd={r.sentiment.updated} "
                f"skip={r.sentiment.skipped}"
            )
        else:
            print(f"  {marker}  {r.ticker:<6}  ERROR: {r.error}")

    ok_count = sum(1 for r in results if r.status == "ok")
    total_articles  = sum(r.articles.total  for r in results)
    total_sentiment = sum(r.sentiment.total for r in results)
    print("─" * 80)
    print(
        f"     {ok_count}/{len(results)} tickers loaded "
        f"| {total_articles} articles  |  {total_sentiment} sentiment rows"
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

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 3: PostgreSQL Storage")
    logger.info("=" * 60)
    logger.info(f"  Database   : {_safe_url(settings.database_url)}")
    logger.info(f"  Model name : {args.model_name}")
    logger.info(f"  Date tag   : {date_tag}")
    logger.info(f"  Input dir  : {INPUT_DIR.relative_to(_PROJECT_ROOT)}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without touching the database.")
        sys.exit(0)

    input_files = _find_input_files(tickers, date_tag)
    if not input_files:
        logger.error("No input files to load — exiting.")
        sys.exit(1)

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

    results: list[TickerLoadResult] = []
    for ticker, csv_path in input_files:
        result = _load_ticker(ticker, csv_path, db, args.model_name)
        results.append(result)

    _print_summary(results)
    db.dispose()


def _safe_url(url: str) -> str:
    """Return the URL with any password redacted."""
    try:
        from sqlalchemy.engine.url import make_url  # noqa: PLC0415
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return "<db_url>"


if __name__ == "__main__":
    main()
