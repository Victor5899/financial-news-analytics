"""
Feature engineering for the financial-news-analytics pipeline.

``FeatureEngineer`` reads sentiment-enriched article data from PostgreSQL,
computes per-ticker per-day ML feature vectors, and writes a ready-to-use
CSV to ``data/features/``.

Feature groups
--------------
Sentiment : article counts, pos/neu/neg ratios, score statistics
Source    : unique source count, per-source article counts
Time      : article volumes in sliding windows (24 h, 3 d, 7 d)
Rolling   : rolling mean sentiment and article volume over 3- and 7-day windows

The rolling mean is calculated from *daily aggregates* (each calendar day
contributes equally regardless of how many articles it contains), which
produces a smoother, less volume-biased signal for ML.

Usage
-----
    from src.features.feature_engineer import FeatureEngineer
    from datetime import date

    eng = FeatureEngineer()
    df  = eng.run(tickers=["AAPL", "TSLA"], target_date=date(2026, 6, 16))

Output column order
-------------------
    ticker, date,
    article_count, positive_count, neutral_count, negative_count,
    positive_ratio, neutral_ratio, negative_ratio,
    mean_sentiment_score, sentiment_score_std, sentiment_score_min, sentiment_score_max,
    unique_source_count, yahoo_article_count, benzinga_article_count, cnbc_article_count,
    articles_last_24h, articles_last_3d, articles_last_7d,
    rolling_3d_mean_sentiment, rolling_7d_mean_sentiment,
    rolling_3d_article_volume, rolling_7d_article_volume
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from src.storage.database import DatabaseManager
from src.storage.models import NewsArticle, SentimentResult
from src.utils.config import settings
from src.utils.logger import get_logger

UTC = timezone.utc
logger = get_logger(__name__)


# ── Feature column ordering ───────────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    "ticker",
    "date",
    # Sentiment
    "article_count",
    "positive_count",
    "neutral_count",
    "negative_count",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "mean_sentiment_score",
    "sentiment_score_std",
    "sentiment_score_min",
    "sentiment_score_max",
    # Source
    "unique_source_count",
    "yahoo_article_count",
    "benzinga_article_count",
    "cnbc_article_count",
    # Time
    "articles_last_24h",
    "articles_last_3d",
    "articles_last_7d",
    # Rolling
    "rolling_3d_mean_sentiment",
    "rolling_7d_mean_sentiment",
    "rolling_3d_article_volume",
    "rolling_7d_article_volume",
]


# ── Exceptions ────────────────────────────────────────────────────────────────

class FeatureEngineeringError(Exception):
    """Base exception for all feature-engineering errors."""


class DataLoadError(FeatureEngineeringError):
    """Raised when data cannot be loaded from the database."""


class FeatureGenerationError(FeatureEngineeringError):
    """Raised when feature computation fails."""


# ── Private feature-computation helpers ──────────────────────────────────────

def _compute_sentiment_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute sentiment aggregate features from a set of articles.

    Parameters
    ----------
    df : pd.DataFrame
        Articles for a single ticker on the target date.
        Must contain ``sentiment_label`` (str) and ``sentiment_score`` (int)
        columns.

    Returns
    -------
    dict
        Eleven sentiment feature key-value pairs.
    """
    n = len(df)
    if n == 0:
        return {
            "article_count":        0,
            "positive_count":       0,
            "neutral_count":        0,
            "negative_count":       0,
            "positive_ratio":       0.0,
            "neutral_ratio":        0.0,
            "negative_ratio":       0.0,
            "mean_sentiment_score": 0.0,
            "sentiment_score_std":  0.0,
            "sentiment_score_min":  0,
            "sentiment_score_max":  0,
        }

    pos = int((df["sentiment_label"] == "positive").sum())
    neu = int((df["sentiment_label"] == "neutral").sum())
    neg = int((df["sentiment_label"] == "negative").sum())
    scores = df["sentiment_score"].astype(float)

    # ddof=1 (sample std); guard against NaN when n == 1
    std: float = float(scores.std(ddof=1)) if n > 1 else 0.0
    if pd.isna(std):
        std = 0.0

    return {
        "article_count":        n,
        "positive_count":       pos,
        "neutral_count":        neu,
        "negative_count":       neg,
        "positive_ratio":       round(pos / n, 6),
        "neutral_ratio":        round(neu / n, 6),
        "negative_ratio":       round(neg / n, 6),
        "mean_sentiment_score": round(float(scores.mean()), 6),
        "sentiment_score_std":  round(std, 6),
        "sentiment_score_min":  int(scores.min()),
        "sentiment_score_max":  int(scores.max()),
    }


def _compute_source_features(df: pd.DataFrame) -> dict[str, int]:
    """
    Compute article-source distribution features.

    Source matching is case-insensitive substring search, so "Yahoo Finance",
    "yahoo_finance", and "Yahoo" all register as a Yahoo article.

    Parameters
    ----------
    df : pd.DataFrame
        Articles for a single ticker on the target date.
        Must contain a ``source_name`` column (str or ``None``).

    Returns
    -------
    dict
        Four source feature key-value pairs.
    """
    if df.empty:
        return {
            "unique_source_count":    0,
            "yahoo_article_count":    0,
            "benzinga_article_count": 0,
            "cnbc_article_count":     0,
        }

    lowered = df["source_name"].fillna("").str.lower()
    return {
        "unique_source_count":    int(df["source_name"].nunique()),
        "yahoo_article_count":    int(lowered.str.contains("yahoo",    na=False).sum()),
        "benzinga_article_count": int(lowered.str.contains("benzinga", na=False).sum()),
        "cnbc_article_count":     int(lowered.str.contains("cnbc",     na=False).sum()),
    }


def _compute_time_features(
    ticker_df: pd.DataFrame,
    target_date: date,
) -> dict[str, int]:
    """
    Compute time-window article volume features.

    All windows include and end on ``target_date``:
    - ``articles_last_24h`` — articles published on ``target_date``
    - ``articles_last_3d``  — articles in ``[target_date − 2 d, target_date]``
    - ``articles_last_7d``  — articles in ``[target_date − 6 d, target_date]``

    Parameters
    ----------
    ticker_df : pd.DataFrame
        All articles for a single ticker (potentially multiple days).
        Must contain a ``date`` column of ``datetime.date`` objects.
    target_date : date
        The reference date for the window endpoints.

    Returns
    -------
    dict
        Three time-window feature key-value pairs.
    """
    if ticker_df.empty:
        return {
            "articles_last_24h": 0,
            "articles_last_3d":  0,
            "articles_last_7d":  0,
        }

    dates = ticker_df["date"]
    return {
        "articles_last_24h": int((dates == target_date).sum()),
        "articles_last_3d": int(
            ((dates >= target_date - timedelta(days=2)) & (dates <= target_date)).sum()
        ),
        "articles_last_7d": int(
            ((dates >= target_date - timedelta(days=6)) & (dates <= target_date)).sum()
        ),
    }


def _compute_rolling_features(
    ticker_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | int]:
    """
    Compute rolling statistics over 3-day and 7-day windows.

    Rolling means are built from *daily aggregates*: each calendar day in the
    window contributes one data point (its mean sentiment) regardless of
    article volume, producing a smoother signal than a raw article-weighted
    mean.  Missing days (no articles) are excluded from the average.

    Parameters
    ----------
    ticker_df : pd.DataFrame
        All articles for a single ticker (potentially multiple days).
        Must contain ``date`` (``datetime.date``) and ``sentiment_score``
        (numeric) columns.
    target_date : date
        The reference date; all windows look back from (and including) this
        date.

    Returns
    -------
    dict
        Four rolling feature key-value pairs.
    """

    if ticker_df.empty or "date" not in ticker_df.columns:
        return {
            "rolling_3d_mean_sentiment": 0.0,
            "rolling_7d_mean_sentiment": 0.0,
            "rolling_3d_article_volume": 0,
            "rolling_7d_article_volume": 0,
        }

    def _daily_stats(days: int) -> tuple[float, int]:
        start = target_date - timedelta(days=days - 1)
        window = ticker_df[
            (ticker_df["date"] >= start) & (ticker_df["date"] <= target_date)
        ]
        if window.empty:
            return 0.0, 0
        daily = (
            window.groupby("date")["sentiment_score"]
            .agg(daily_mean="mean", daily_count="count")
            .reset_index()
        )
        rolling_mean = round(float(daily["daily_mean"].mean()), 6)
        rolling_volume = int(daily["daily_count"].sum())
        return rolling_mean, rolling_volume

    mean_3d, vol_3d = _daily_stats(3)
    mean_7d, vol_7d = _daily_stats(7)

    return {
        "rolling_3d_mean_sentiment": mean_3d,
        "rolling_7d_mean_sentiment": mean_7d,
        "rolling_3d_article_volume": vol_3d,
        "rolling_7d_article_volume": vol_7d,
    }


# ── FeatureEngineer ───────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms PostgreSQL sentiment data into a ticker-level ML feature set.

    The class is designed as a thin orchestration layer: it delegates
    feature computation to the module-level helper functions so that each
    feature group can be tested and reasoned about independently.

    Parameters
    ----------
    database_url : str | None
        SQLAlchemy-compatible connection URL.
        Defaults to ``settings.database_url`` when ``None``.

    Raises
    ------
    DataLoadError
        If no database URL is configured and one is required.
    """

    def __init__(self, database_url: str | None = None) -> None:
        resolved = database_url or settings.database_url
        if not resolved:
            raise DataLoadError(
                "No database URL configured. "
                "Set DATABASE_URL in your .env file or pass database_url explicitly.\n"
                "  Example: DATABASE_URL=postgresql://user:pass@localhost:5432/financial_news"
            )
        self._database_url: str = resolved
        self._db: DatabaseManager | None = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_db(self) -> DatabaseManager:
        """Lazily initialise and cache the DatabaseManager."""
        if self._db is None:
            self._db = DatabaseManager(self._database_url)
        return self._db

    # ── Public API ────────────────────────────────────────────────────────────

    def load_data(
        self,
        tickers: list[str] | None = None,
        target_date: date | None = None,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """
        Load joined article + sentiment data from the database.

        Executes a single SQL JOIN across ``news_articles`` and
        ``sentiment_results`` for the window
        ``[target_date − lookback_days, target_date]``.  Only articles that
        already have an associated sentiment result are included.

        Parameters
        ----------
        tickers : list[str] | None
            Ticker symbols to load.  ``None`` loads all available tickers.
        target_date : date | None
            End date of the load window.  Defaults to today (UTC).
        lookback_days : int
            Number of calendar days to look back from ``target_date``.
            Increase this value if you need rolling features beyond 7 days.
            Default: ``7``.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``source_name``, ``published_at``, ``date``,
            ``sentiment_label``, ``sentiment_score``, ``sentiment_confidence``.
            Returns an empty DataFrame (not an error) when no rows are found.

        Raises
        ------
        DataLoadError
            On any database connectivity or query failure.
        """
        if target_date is None:
            target_date = datetime.now(UTC).date()

        start_dt = datetime.combine(
            target_date - timedelta(days=lookback_days), time.min
        ).replace(tzinfo=UTC)
        end_dt = datetime.combine(target_date, time.max).replace(tzinfo=UTC)

        logger.debug(
            f"load_data: window={start_dt.date()} → {end_dt.date()} "
            f"tickers={tickers or 'all'}"
        )

        try:
            db = self._get_db()

            stmt = (
                select(
                    NewsArticle.ticker.label("ticker"),
                    NewsArticle.source_name.label("source_name"),
                    NewsArticle.published_at.label("published_at"),
                    SentimentResult.sentiment_label.label("sentiment_label"),
                    SentimentResult.sentiment_score.label("sentiment_score"),
                    SentimentResult.sentiment_confidence.label("sentiment_confidence"),
                )
                .join(SentimentResult, NewsArticle.id == SentimentResult.article_id)
                .where(NewsArticle.published_at >= start_dt)
                .where(NewsArticle.published_at <= end_dt)
            )

            if tickers:
                upper = [t.upper() for t in tickers]
                stmt = stmt.where(NewsArticle.ticker.in_(upper))

            with db.engine.connect() as conn:
                df = pd.read_sql(stmt, conn)

        except FeatureEngineeringError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"Failed to load data from database: {exc}"
            ) from exc

        if df.empty:
            logger.warning(
                f"No articles found for tickers={tickers or 'all'}, "
                f"window={start_dt.date()} → {end_dt.date()}"
            )
            return df

        df["published_at"] = pd.to_datetime(
            df["published_at"], utc=True, errors="coerce"
        )
        df["date"] = df["published_at"].dt.date

        logger.info(
            f"Loaded {len(df)} articles across "
            f"{df['ticker'].nunique()} ticker(s) "
            f"({start_dt.date()} → {end_dt.date()})"
        )
        return df

    def generate_features(
        self,
        raw_df: pd.DataFrame,
        target_date: date,
    ) -> pd.DataFrame:
        """
        Compute per-ticker feature vectors for the given target date.

        For each ticker present in ``raw_df``, the full feature set is
        computed.  Tickers that have no articles *on* ``target_date`` are
        silently skipped (rolling and time features can still use historical
        data from the window).

        Parameters
        ----------
        raw_df : pd.DataFrame
            Output of :meth:`load_data`.  Must contain the columns documented
            there.
        target_date : date
            The date for which same-day features are computed (sentiment,
            source).  Rolling and time features look back from this date.

        Returns
        -------
        pd.DataFrame
            One row per ticker; columns follow ``FEATURE_COLUMNS`` order.

        Raises
        ------
        FeatureGenerationError
            If ``raw_df`` is empty or no ticker has articles on
            ``target_date``.
        """
        if raw_df.empty:
            raise FeatureGenerationError(
                "Cannot generate features from an empty DataFrame. "
                "Ensure load_data() returned data before calling generate_features()."
            )

        rows: list[dict[str, Any]] = []

        for ticker in sorted(raw_df["ticker"].unique()):
            ticker_df = raw_df[raw_df["ticker"] == ticker].copy()
            target_df = ticker_df[ticker_df["date"] == target_date]

            if target_df.empty:
                logger.debug(
                    f"[{ticker}] No articles on {target_date} — skipping"
                )
                continue

            logger.debug(
                f"[{ticker}] Generating features for {target_date} "
                f"({len(target_df)} same-day articles, "
                f"{len(ticker_df)} total in window)"
            )

            row: dict[str, Any] = {"ticker": ticker, "date": target_date}
            row.update(_compute_sentiment_features(target_df))
            row.update(_compute_source_features(target_df))
            row.update(_compute_time_features(ticker_df, target_date))
            row.update(_compute_rolling_features(ticker_df, target_date))
            rows.append(row)

        if not rows:
            raise FeatureGenerationError(
                f"No features generated for target_date={target_date}. "
                "No tickers had articles on that date in the loaded window."
            )

        features_df = pd.DataFrame(rows)[FEATURE_COLUMNS]
        logger.info(
            f"Generated {len(features_df[FEATURE_COLUMNS]) - 2} features "  # minus ticker+date
            f"for {len(features_df)} ticker(s) on {target_date}"
        )
        return features_df

    def save_features(
        self,
        features_df: pd.DataFrame,
        output_dir: Path,
        date_tag: str,
    ) -> Path:
        """
        Save the feature DataFrame to a dated CSV file.

        The output filename follows the convention
        ``feature_dataset_<date_tag>.csv``.  The directory is created
        (including any missing parents) if it does not already exist.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of :meth:`generate_features`.
        output_dir : Path
            Target directory.  Created automatically if absent.
        date_tag : str
            Date string appended to the filename (e.g. ``"2026-06-16"``).

        Returns
        -------
        Path
            Absolute path of the written CSV.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"feature_dataset_{date_tag}.csv"
        features_df.to_csv(out_path, index=False)
        logger.info(
            f"Saved {len(features_df)} feature row(s) → {out_path}"
        )
        return out_path

    def run_range(
        self,
        start_date: date,
        end_date: date,
        tickers: list[str] | None = None,
        output_dir: Path | None = None,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """
        Generate features for every calendar date in ``[start_date, end_date]``.

        Loads article + sentiment data in a **single** database query covering
        ``[start_date - lookback_days, end_date]``, then iterates over dates
        one at a time and applies the standard feature-generation logic for
        each.  Dates that have no articles on the target day are silently
        skipped (rolling / time features for adjacent dates are unaffected
        because the full window is already in memory).

        Parameters
        ----------
        start_date : date
            First date to generate features for (inclusive).
        end_date : date
            Last date to generate features for (inclusive).
        tickers : list[str] | None
            Tickers to process.  ``None`` processes all available tickers.
        output_dir : Path | None
            When provided the combined DataFrame is saved to
            ``<output_dir>/feature_dataset_<start>_<end>.csv``.
        lookback_days : int
            History window for rolling features.  Default: ``7``.

        Returns
        -------
        pd.DataFrame
            All feature rows for all processed dates, sorted by ``date`` then
            ``ticker``.  Returns an empty DataFrame (with correct columns)
            when no data is found.

        Raises
        ------
        ValueError
            If ``start_date`` is after ``end_date``.
        DataLoadError
            On database connectivity failure.
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must not be after end_date ({end_date})"
            )

        total_lookback = (end_date - start_date).days + lookback_days

        logger.info(
            f"run_range: {start_date} → {end_date}  "
            f"({(end_date - start_date).days + 1} calendar days)  "
            f"lookback={lookback_days}d"
        )

        raw_df = self.load_data(
            tickers=tickers,
            target_date=end_date,
            lookback_days=total_lookback,
        )

        if raw_df.empty:
            logger.warning(
                "run_range: no articles found for the requested range — "
                "returning empty feature DataFrame."
            )
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        all_frames: list[pd.DataFrame] = []
        dates_processed = 0
        dates_skipped = 0

        current = start_date
        while current <= end_date:
            try:
                features_df = self.generate_features(raw_df, current)
                all_frames.append(features_df)
                dates_processed += 1
            except FeatureGenerationError:
                logger.debug(
                    f"run_range: no same-day articles on {current} — skipping"
                )
                dates_skipped += 1
            current += timedelta(days=1)

        logger.info(
            f"run_range complete: {dates_processed} dates with features, "
            f"{dates_skipped} dates skipped (no articles)"
        )

        if not all_frames:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        combined = pd.concat(all_frames, ignore_index=True)

        if output_dir is not None:
            start_tag = start_date.strftime("%Y-%m-%d")
            end_tag   = end_date.strftime("%Y-%m-%d")
            date_tag  = f"{start_tag}_{end_tag}"
            self.save_features(combined, output_dir, date_tag)

        return combined

    def run(
        self,
        tickers: list[str] | None = None,
        target_date: date | None = None,
        output_dir: Path | None = None,
        lookback_days: int = 7,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """
        Full pipeline: load → generate → save.

        Parameters
        ----------
        tickers : list[str] | None
            Tickers to process.  ``None`` processes all available tickers.
        target_date : date | None
            Feature date.  Defaults to today (UTC).
        output_dir : Path | None
            Where to save the CSV.  If ``None`` the caller is responsible for
            saving (useful when ``generate_features.py`` wants a custom path).
        lookback_days : int
            History window for rolling features.  Default: ``7``.
        dry_run : bool
            When ``True``, skip all I/O and return an empty DataFrame.

        Returns
        -------
        pd.DataFrame
            The generated feature DataFrame, or an empty DataFrame when
            ``dry_run=True`` or no data was found.

        Raises
        ------
        DataLoadError
            On database connectivity failure.
        FeatureGenerationError
            If the loaded data contains no articles on ``target_date``.
        """
        if target_date is None:
            target_date = datetime.now(UTC).date()

        if dry_run:
            logger.info("Dry-run: skipping data load and feature generation.")
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        raw_df = self.load_data(
            tickers=tickers,
            target_date=target_date,
            lookback_days=lookback_days,
        )

        if raw_df.empty:
            logger.warning("No data loaded — returning empty feature DataFrame.")
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        features_df = self.generate_features(raw_df, target_date)

        if output_dir is not None:
            date_tag = target_date.strftime("%Y-%m-%d")
            self.save_features(features_df, output_dir, date_tag)

        return features_df

    def dispose(self) -> None:
        """Release all pooled database connections at application shutdown."""
        if self._db is not None:
            self._db.dispose()
            self._db = None
            logger.debug("FeatureEngineer: database connections disposed")
