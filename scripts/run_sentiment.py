#!/usr/bin/env python3
"""
Phase 2 entry-point: run FinBERT sentiment analysis on Phase 1 news CSVs.

Reads ``data/raw/<TICKER>_news_<date>.csv`` files produced by Phase 1,
enriches them with FinBERT sentiment labels and scores, then writes results
to ``data/processed/``.

Run from the project root
-------------------------
    # All CSVs in data/raw/ for today
    python scripts/run_sentiment.py

    # Specific tickers
    python scripts/run_sentiment.py --tickers AAPL TSLA

    # Specific date
    python scripts/run_sentiment.py --date 2026-06-15

    # Override model / batch size / device
    python scripts/run_sentiment.py --model ProsusAI/finbert --batch-size 16 --device cpu

    # Dry-run (prints config, exits without inference)
    python scripts/run_sentiment.py --dry-run

Output
------
  data/processed/<TICKER>_sentiment_<YYYY-MM-DD>.csv   enriched per-ticker CSV
  data/processed/sentiment_summary_<YYYY-MM-DD>.csv    one-row-per-ticker summary
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UTC = timezone.utc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.processing.sentiment_analyzer import (
    FinBERTSentimentAnalyzer,
    ModelLoadError,
    SentimentAnalysisError,
)
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

INPUT_DIR  = _PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinBERT sentiment analysis on ingested financial news CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Ticker symbols to process (default: auto-discover all CSVs for the given date)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Date tag of the input CSVs to process (default: today)",
    )
    parser.add_argument(
        "--model",
        default=settings.finbert_model,
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.finbert_batch_size,
        metavar="N",
        help="Number of articles per inference batch",
    )
    parser.add_argument(
        "--device",
        default=settings.finbert_device,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for model inference",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--input-file",
        metavar="PATH",
        default=None,
        help=(
            "Process a single CSV file directly (Finnhub or GDELT). "
            "Ticker and output tag are inferred from the filename. "
            "Skips date-based discovery when provided."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit without running inference",
    )
    return parser.parse_args()


# ── Input discovery ───────────────────────────────────────────────────────────

def _find_input_files(
    tickers: list[str] | None,
    date_tag: str,
) -> list[tuple[str, Path]]:
    """Return ``(ticker, path)`` pairs for CSVs matching the given criteria."""
    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR.relative_to(_PROJECT_ROOT)}")
        return []

    if tickers:
        pairs: list[tuple[str, Path]] = []
        for ticker in tickers:
            path = INPUT_DIR / f"{ticker}_news_{date_tag}.csv"
            if path.exists():
                pairs.append((ticker, path))
            else:
                logger.warning(f"[{ticker}] Input file not found: {path.name}")
        return pairs

    # Auto-discover: all <TICKER>_news_<date_tag>.csv files, skip summary files
    discovered: list[tuple[str, Path]] = []
    for path in sorted(INPUT_DIR.glob(f"*_news_{date_tag}.csv")):
        ticker = path.stem.split("_news_")[0]
        discovered.append((ticker, path))

    if not discovered:
        logger.warning(
            f"No news CSVs found for date '{date_tag}' in "
            f"{INPUT_DIR.relative_to(_PROJECT_ROOT)}. "
            "Run scripts/fetch_news.py first."
        )
    return discovered


# ── Filename parsing ──────────────────────────────────────────────────────────

_GDELT_RE   = re.compile(r"^([A-Z]+)_gdelt_(\d{4})-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}$")
_FINNHUB_RE = re.compile(r"^([A-Z]+)_news_(\d{4}-\d{2}-\d{2})$")


def _parse_input_filename(path: Path) -> tuple[str, str]:
    """Return ``(ticker, output_tag)`` inferred from *path*'s filename stem.

    Supported patterns
    ------------------
    Finnhub : ``<TICKER>_news_<YYYY-MM-DD>.csv``    → output_tag = ``YYYY-MM-DD``
    GDELT   : ``<TICKER>_gdelt_<START>_<END>.csv``  → output_tag = start year (``YYYY``)

    Raises
    ------
    ValueError
        When the filename does not match either expected pattern.
    """
    stem = path.stem

    m = _GDELT_RE.match(stem)
    if m:
        return m.group(1), m.group(2)  # ticker, start-year

    m = _FINNHUB_RE.match(stem)
    if m:
        return m.group(1), m.group(2)  # ticker, date-tag

    raise ValueError(
        f"Cannot infer ticker from '{path.name}'. "
        "Expected '<TICKER>_news_<YYYY-MM-DD>.csv' or "
        "'<TICKER>_gdelt_<YYYY-MM-DD>_<YYYY-MM-DD>.csv'."
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_enriched(ticker: str, df: pd.DataFrame, date_tag: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{ticker}_sentiment_{date_tag}.csv"
    df.to_csv(out, index=False)
    logger.info(f"[{ticker}] Saved {len(df)} rows → {out.relative_to(_PROJECT_ROOT)}")


def _save_summary(results: dict[str, pd.DataFrame], date_tag: str) -> None:
    rows: list[dict] = []
    for ticker, df in results.items():
        if df.empty or "sentiment_label" not in df.columns:
            rows.append({
                "ticker":        ticker,
                "article_count": 0,
                "positive":      0,
                "neutral":       0,
                "negative":      0,
                "mean_score":    None,
                "status":        "empty",
            })
            continue
        rows.append({
            "ticker":        ticker,
            "article_count": len(df),
            "positive":      int((df["sentiment_label"] == "positive").sum()),
            "neutral":       int((df["sentiment_label"] == "neutral").sum()),
            "negative":      int((df["sentiment_label"] == "negative").sum()),
            "mean_score":    round(float(df["sentiment_score"].mean()), 4),
            "status":        "ok",
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"sentiment_summary_{date_tag}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logger.info(f"Summary saved → {summary_path.relative_to(_PROJECT_ROOT)}")


def _print_summary(results: dict[str, pd.DataFrame]) -> None:
    print("\n" + "─" * 74)
    print("  SENTIMENT RESULTS SUMMARY")
    print("─" * 74)
    for ticker, df in results.items():
        if df.empty or "sentiment_label" not in df.columns:
            print(f"  ✗  {ticker:<6}  no data")
            continue
        pos  = int((df["sentiment_label"] == "positive").sum())
        neu  = int((df["sentiment_label"] == "neutral").sum())
        neg  = int((df["sentiment_label"] == "negative").sum())
        mean = float(df["sentiment_score"].mean())
        print(
            f"  ✓  {ticker:<6}  "
            f"{len(df):>4} articles  "
            f"+{pos} pos  {neu} neu  -{neg} neg  "
            f"mean={mean:+.3f}"
        )
    total = sum(len(df) for df in results.values())
    print("─" * 74)
    print(f"     TOTAL: {total} articles across {len(results)} tickers")
    print("─" * 74 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("financial-news-analytics | Phase 2: Sentiment Analysis")
    logger.info("=" * 60)
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  Batch size : {args.batch_size}")
    logger.info(f"  Device     : {args.device}")
    logger.info(f"  Input dir  : {INPUT_DIR.relative_to(_PROJECT_ROOT)}")
    logger.info(f"  Output dir : {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    if args.input_file:
        # ── Single-file mode (Finnhub or GDELT backfill) ──────────────────────
        input_path = Path(args.input_file).resolve()
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        try:
            ticker, output_tag = _parse_input_filename(input_path)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)
        input_files: list[tuple[str, Path]] = [(ticker, input_path)]
        logger.info(f"  Input file : {input_path.name}")
        logger.info(f"  Ticker     : {ticker}")
        logger.info(f"  Output tag : {output_tag}")

        if args.dry_run:
            logger.info("Dry-run mode — exiting without running inference.")
            sys.exit(0)
    else:
        # ── Date-based discovery mode (original Finnhub behaviour) ────────────
        tickers  = [t.upper() for t in args.tickers] if args.tickers else None
        date_tag = args.date or datetime.now(UTC).strftime("%Y-%m-%d")
        output_tag = date_tag
        logger.info(f"  Date tag   : {date_tag}")

        if args.dry_run:
            logger.info("Dry-run mode — exiting without running inference.")
            sys.exit(0)

        input_files = _find_input_files(tickers, date_tag)
        if not input_files:
            logger.error("No input files to process — exiting.")
            sys.exit(1)

    logger.info(f"Found {len(input_files)} input file(s) to process")

    analyzer = FinBERTSentimentAnalyzer(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    try:
        analyzer.load()
    except ModelLoadError as exc:
        logger.error(f"Fatal: could not load model — {exc}")
        sys.exit(1)

    results: dict[str, pd.DataFrame] = {}

    for ticker, csv_path in input_files:
        logger.info(f"[{ticker}] Reading {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{ticker}] Failed to read CSV: {exc}")
            results[ticker] = pd.DataFrame()
            continue

        if df.empty:
            logger.warning(f"[{ticker}] CSV is empty — skipping")
            results[ticker] = pd.DataFrame()
            continue

        try:
            enriched = analyzer.analyse_dataframe(df)
        except SentimentAnalysisError as exc:
            logger.error(f"[{ticker}] Sentiment analysis failed: {exc}")
            results[ticker] = pd.DataFrame()
            continue

        results[ticker] = enriched
        _save_enriched(ticker, enriched, output_tag)

    _print_summary(results)
    _save_summary(results, output_tag)


if __name__ == "__main__":
    main()
