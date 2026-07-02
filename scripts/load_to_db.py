#!/usr/bin/env python3
"""
Phase 3 entry-point: load Phase 2 sentiment CSVs into PostgreSQL.

Reads ``data/processed/<TICKER>_sentiment_<date>.csv`` files produced by
Phase 2, upserts the news-article rows into ``news_articles``, then links
the sentiment scores into ``sentiment_results``.

Three input modes are supported and are mutually exclusive:

  Default (date-based)
    Auto-discover all ``*_sentiment_<date>.csv`` files for today (or a given
    ``--date``) in ``data/processed/``.  Optionally filter by ``--tickers``.

  Single file (``--input-file``)
    Load exactly one processed sentiment CSV.  The ticker is inferred from
    the filename (``<TICKER>_sentiment_<tag>.csv``).

  Directory scan (``--input-dir``)
    Scan a directory for every valid sentiment CSV and load them all in
    alphabetical filename order.  Files that do not match the expected
    naming pattern are skipped; a per-file error does not abort the run.

Run from the project root
-------------------------
    # Default: load all processed CSVs for today
    python scripts/load_to_db.py

    # Specific tickers (default mode)
    python scripts/load_to_db.py --tickers AAPL TSLA

    # Specific date (default mode)
    python scripts/load_to_db.py --date 2026-06-16

    # Create tables then load (safe on existing DB — uses IF NOT EXISTS)
    python scripts/load_to_db.py --create-tables

    # Override model name stored in sentiment_results
    python scripts/load_to_db.py --model-name ProsusAI/finbert

    # Dry-run: parse config + exit before touching the DB
    python scripts/load_to_db.py --dry-run

    # Single file
    python scripts/load_to_db.py \\
        --input-file data/processed/AAPL_sentiment_2023.csv

    # Whole directory (historical backfill)
    python scripts/load_to_db.py \\
        --input-dir data/processed

Output
------
  PostgreSQL database tables:
    news_articles      — deduplicated by url
    sentiment_results  — deduplicated by (article_id, model_name)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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

#: Pattern component used to identify and parse sentiment CSV filenames.
_SENTIMENT_TAG = "_sentiment_"


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TickerLoadResult:
    ticker:    str
    articles:  UpsertResult
    sentiment: UpsertResult
    status:    str = "ok"
    error:     str | None = None
    rows_csv:  int = 0   # number of rows read from the source CSV


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Phase 2 sentiment CSVs into PostgreSQL.  "
            "Supports three mutually exclusive input modes: default date-based "
            "discovery, a single file (--input-file), or a full directory scan "
            "(--input-dir)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help=(
            "Ticker symbols to load (default mode only; "
            "default: auto-discover all processed CSVs for the given date)"
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Date tag of the processed CSVs to load (default mode only; default: today)",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        metavar="PATH",
        dest="input_file",
        help=(
            "Load exactly one processed sentiment CSV.  "
            "The ticker is inferred from the filename "
            "(<TICKER>_sentiment_<tag>.csv).  "
            "Mutually exclusive with --input-dir, --tickers, and --date."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        metavar="PATH",
        dest="input_dir",
        help=(
            "Scan a directory for all valid sentiment CSVs and load them in "
            "alphabetical order.  Non-matching files are skipped; a per-file "
            "error does not abort the run.  "
            "Mutually exclusive with --input-file, --tickers, and --date."
        ),
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

def _ticker_from_path(path: Path) -> str | None:
    """Extract the ticker symbol from a sentiment CSV filename.

    Expects the naming pattern ``<TICKER>_sentiment_<tag>.csv``.
    Returns ``None`` if the filename does not match.

    Parameters
    ----------
    path : Path
        CSV file path whose stem is inspected.

    Returns
    -------
    str | None
        Upper-cased ticker string, or ``None`` on a pattern mismatch.
    """
    parts = path.stem.split(_SENTIMENT_TAG, maxsplit=1)
    if len(parts) != 2 or not parts[0]:
        return None
    return parts[0].upper()


def _resolve_single_file(path: Path) -> tuple[str, Path] | None:
    """Validate *path* and return ``(ticker, path)``, or ``None`` on error.

    Checks that the file exists, has a ``.csv`` extension, and that a ticker
    can be inferred from its name.  Logs a descriptive error for each failure.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the candidate CSV file.

    Returns
    -------
    tuple[str, Path] | None
        ``(ticker, resolved_path)`` on success, ``None`` on any validation failure.
    """
    if not path.exists():
        logger.error(f"Input file not found: {path}")
        return None
    if path.suffix.lower() != ".csv":
        logger.error(f"Input file is not a CSV: {path.name}")
        return None
    ticker = _ticker_from_path(path)
    if ticker is None:
        logger.error(
            f"Cannot infer ticker from filename '{path.name}'. "
            f"Expected pattern: <TICKER>{_SENTIMENT_TAG}<tag>.csv"
        )
        return None
    return ticker, path


def _discover_dir_files(dir_path: Path) -> list[tuple[str, Path]]:
    """Return ``(ticker, path)`` pairs for all valid sentiment CSVs in *dir_path*.

    Scans *dir_path* non-recursively and filters to files whose names match
    the ``<TICKER>_sentiment_<tag>.csv`` pattern.  Results are sorted by
    filename for deterministic ordering.  Files that do not match — including
    non-CSV files — are silently skipped at DEBUG level.

    Parameters
    ----------
    dir_path : Path
        Directory to scan.

    Returns
    -------
    list[tuple[str, Path]]
        Pairs of ``(ticker, path)``, sorted alphabetically by filename.
        Empty list if the directory does not exist or contains no matches.
    """
    if not dir_path.exists():
        logger.error(f"Input directory not found: {dir_path}")
        return []

    pairs: list[tuple[str, Path]] = []
    skipped = 0
    for path in sorted(dir_path.iterdir()):
        if path.suffix.lower() != ".csv":
            continue
        ticker = _ticker_from_path(path)
        if ticker is None:
            logger.debug(f"Skipping non-sentiment CSV: {path.name}")
            skipped += 1
            continue
        pairs.append((ticker, path))

    if skipped:
        logger.debug(f"Skipped {skipped} non-sentiment file(s) in {dir_path}")

    if not pairs:
        logger.warning(f"No valid sentiment CSVs found in {dir_path}")
    else:
        logger.info(f"Discovered {len(pairs)} sentiment CSV(s) in {dir_path}")

    return pairs


def _find_input_files(
    tickers: list[str] | None,
    date_tag: str,
) -> list[tuple[str, Path]]:
    """Return ``(ticker, path)`` pairs for processed sentiment CSVs (default mode).

    When *tickers* is provided, resolves explicit paths and warns about any
    missing files.  Otherwise auto-discovers all ``*_sentiment_<date_tag>.csv``
    files in :data:`INPUT_DIR`.
    """
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
        ticker = path.stem.split(_SENTIMENT_TAG)[0]
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
    Returns a :class:`TickerLoadResult` regardless of success or failure;
    callers should inspect ``result.status`` rather than catching exceptions.
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
        rows_csv=len(rows),
    )


# ── Console summaries ─────────────────────────────────────────────────────────

def _print_summary(results: list[TickerLoadResult]) -> None:
    """Print a per-ticker load summary (default and single-file modes)."""
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


def _print_historical_summary(results: list[TickerLoadResult]) -> None:
    """Print the aggregate summary for a historical directory load."""
    succeeded   = sum(1 for r in results if r.status == "ok")
    failed      = sum(1 for r in results if r.status == "error")
    total_rows  = sum(r.rows_csv for r in results)

    print("\n" + "=" * 36)
    print("Historical Load Summary")
    print("=" * 36)
    print(f"Files processed  : {len(results)}")
    print(f"Files succeeded  : {succeeded}")
    print(f"Files failed     : {failed}")
    print(f"Total rows loaded: {total_rows}")
    print("=" * 36 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    # ── Mutual exclusivity guard ───────────────────────────────────────────────
    if args.input_file and args.input_dir:
        logger.error("--input-file and --input-dir are mutually exclusive.")
        sys.exit(1)

    if not settings.database_url:
        logger.error(
            "DATABASE_URL is not set. "
            "Add it to your .env file:\n"
            "  DATABASE_URL=postgresql://user:password@localhost:5432/financial_news"
        )
        sys.exit(1)

    tickers  = [t.upper() for t in args.tickers] if args.tickers else None
    date_tag = args.date or datetime.now(UTC).strftime("%Y-%m-%d")

    # ── Determine input mode ───────────────────────────────────────────────────
    if args.input_file:
        input_mode = "single-file"
    elif args.input_dir:
        input_mode = "historical-dir"
    else:
        input_mode = "daily"

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 3: PostgreSQL Storage")
    logger.info("=" * 60)
    logger.info(f"  Database   : {_safe_url(settings.database_url)}")
    logger.info(f"  Model name : {args.model_name}")
    logger.info(f"  Input mode : {input_mode}")
    if input_mode == "daily":
        logger.info(f"  Date tag   : {date_tag}")
        logger.info(f"  Input dir  : {INPUT_DIR.relative_to(_PROJECT_ROOT)}")
    elif input_mode == "single-file":
        logger.info(f"  Input file : {args.input_file}")
    else:
        logger.info(f"  Input dir  : {args.input_dir}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting without touching the database.")
        sys.exit(0)

    # ── Resolve input files ────────────────────────────────────────────────────
    if input_mode == "single-file":
        pair = _resolve_single_file(Path(args.input_file))
        if pair is None:
            sys.exit(1)
        input_files = [pair]

    elif input_mode == "historical-dir":
        input_files = _discover_dir_files(Path(args.input_dir))
        if not input_files:
            logger.error("No input files to load — exiting.")
            sys.exit(1)

    else:  # daily (default)
        input_files = _find_input_files(tickers, date_tag)
        if not input_files:
            logger.error("No input files to load — exiting.")
            sys.exit(1)

    # ── Connect to database ────────────────────────────────────────────────────
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

    # ── Load files ─────────────────────────────────────────────────────────────
    results: list[TickerLoadResult] = []

    if input_mode == "historical-dir":
        for ticker, csv_path in input_files:
            print(f"\nLoading {csv_path.name} ...")
            result = _load_ticker(ticker, csv_path, db, args.model_name)
            results.append(result)
            if result.status == "ok":
                print(f"Loaded {result.rows_csv} rows.")
            elif result.status == "empty":
                print("CSV is empty — skipped.")
            else:
                print(f"Failed: {result.error}")
        _print_historical_summary(results)

    else:
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
