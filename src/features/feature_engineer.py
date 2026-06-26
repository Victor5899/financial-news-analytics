"""
Feature engineering for the financial-news-analytics pipeline.

``FeatureEngineer`` reads sentiment-enriched article data from PostgreSQL,
computes per-ticker per-day ML feature vectors, and writes a ready-to-use
CSV to ``data/features/``.

Feature groups
--------------
Sentiment   : article counts, pos/neu/neg ratios, score statistics
Source      : unique source count, per-source article counts
Time        : article volumes in sliding windows (24 h, 3 d, 7 d)
Rolling     : rolling mean sentiment and article volume over 3- and 7-day windows
Trend       : SMA 10/20, EMA 10/20
Momentum    : RSI 14, MACD line, MACD signal, MACD histogram
Volatility  : Bollinger upper/lower/width, ATR 14, 20-day rolling volatility
Returns     : 1-day, 5-day, 10-day price percentage changes
Volume      : volume change (%), 5-day average volume, volume-to-average ratio

The rolling mean is calculated from *daily aggregates* (each calendar day
contributes equally regardless of how many articles it contains), which
produces a smoother, less volume-biased signal for ML.

All technical indicators are derived from historical OHLCV data stored in
the ``stock_prices`` table (Phase 5).  They are computed with pandas only —
no external TA libraries are used.  Rows where insufficient price history
exists for an indicator are left as NaN rather than filled with arbitrary
values.

Usage
-----
    from src.features.feature_engineer import FeatureEngineer
    from datetime import date

    eng = FeatureEngineer()
    df  = eng.run(tickers=["AAPL", "TSLA"], target_date=date(2026, 6, 16))

Output column order
-------------------
    ticker, date,
    # Sentiment (11)
    article_count, positive_count, neutral_count, negative_count,
    positive_ratio, neutral_ratio, negative_ratio,
    mean_sentiment_score, sentiment_score_std, sentiment_score_min,
    sentiment_score_max,
    # Source (4)
    unique_source_count, yahoo_article_count, benzinga_article_count,
    cnbc_article_count,
    # Time (3)
    articles_last_24h, articles_last_3d, articles_last_7d,
    # Rolling (4)
    rolling_3d_mean_sentiment, rolling_7d_mean_sentiment,
    rolling_3d_article_volume, rolling_7d_article_volume,
    # Trend (4)
    sma_10, sma_20, ema_10, ema_20,
    # Momentum (4)
    rsi_14, macd, macd_signal, macd_histogram,
    # Volatility (5)
    bb_upper, bb_lower, bb_width, atr_14, volatility_20d,
    # Returns (3)
    price_chg_1d, price_chg_5d, price_chg_10d,
    # Volume (3)
    volume_change_pct, volume_avg_5d, volume_ratio
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from src.storage.database import DatabaseManager
from src.storage.models import NewsArticle, SentimentResult, StockPrice
from src.utils.config import settings
from src.utils.logger import get_logger

UTC = timezone.utc
logger = get_logger(__name__)


# ── Feature column ordering ───────────────────────────────────────────────────

#: Sentiment, source, time, and rolling feature columns (Phase 4 originals).
_SENTIMENT_FEATURE_COLUMNS: list[str] = [
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

#: Technical indicator feature columns added in Phase 4 extension.
TECHNICAL_FEATURE_COLUMNS: list[str] = [
    # Trend
    "sma_10",
    "sma_20",
    "ema_10",
    "ema_20",
    # Momentum
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    # Volatility
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "volatility_20d",
    # Returns (price-based; prefixed to avoid collision with ML label columns)
    "price_chg_1d",
    "price_chg_5d",
    "price_chg_10d",
    # Volume
    "volume_change_pct",
    "volume_avg_5d",
    "volume_ratio",
]

#: Full canonical output column order (sentinel + sentiment + technical).
FEATURE_COLUMNS: list[str] = _SENTIMENT_FEATURE_COLUMNS + TECHNICAL_FEATURE_COLUMNS


# ── Exceptions ────────────────────────────────────────────────────────────────

class FeatureEngineeringError(Exception):
    """Base exception for all feature-engineering errors."""


class DataLoadError(FeatureEngineeringError):
    """Raised when data cannot be loaded from the database."""


class FeatureGenerationError(FeatureEngineeringError):
    """Raised when feature computation fails."""


# ── Private sentiment / source / time / rolling helpers ──────────────────────

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


# ── Private technical-indicator primitives ────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over ``window`` periods."""
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average with ``span`` periods (pandas EWM, adjust=False)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EWM with alpha = 1 / period).

    Returns values in [0, 100].  NaN for the first ``period`` rows.
    When all gains in the window are zero (pure downtrend) RSI → 0.
    When all losses are zero (pure uptrend) RSI → 100.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # Guard: where avg_loss == 0, RSI should be 100 (no losing periods)
    rs = avg_gain / avg_loss.mask(avg_loss == 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0.0, other=100.0)


def _macd_lines(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, and histogram.

    Returns
    -------
    tuple
        ``(macd_line, signal_line, histogram)`` — all ``pd.Series``.
        NaN for the first ``slow + signal_span - 1`` rows.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False, min_periods=signal_span).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger upper band, lower band, and bandwidth.

    Bandwidth is defined as ``(upper − lower) / middle``.
    Returns NaN for the first ``window − 1`` rows.
    """
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=1)
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.where(sma != 0.0)
    return upper, lower, width


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR) using Wilder's EWM smoothing.

    True Range = max(High−Low, |High−PrevClose|, |Low−PrevClose|).
    Returns NaN for the first ``period`` rows.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).rename("hl"),
            (high - prev_close).abs().rename("hpc"),
            (low - prev_close).abs().rename("lpc"),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ── Private technical feature-group helpers ───────────────────────────────────

def _extract_at(
    series: pd.Series,
    dates: pd.Series,
    target_date: date,
) -> float | None:
    """
    Extract a scalar value from a computed indicator series at ``target_date``.

    Parameters
    ----------
    series : pd.Series
        Computed indicator values (same index as ``dates``).
    dates : pd.Series
        Trading dates corresponding to each row of ``series``.
    target_date : date
        The date to extract the value for.

    Returns
    -------
    float | None
        Rounded value, or ``None`` if the date is absent or the value is NaN.
    """
    mask = dates == target_date
    if not mask.any():
        return None
    val = series[mask].iloc[-1]
    if pd.isna(val):
        return None
    return round(float(val), 8)


def _compute_trend_features(
    prices_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | None]:
    """
    Compute SMA 10/20 and EMA 10/20 for ``target_date``.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV rows for a single ticker, sorted ascending by ``trading_date``.
        Must contain ``close_price`` and ``trading_date`` columns.
    target_date : date
        The date for which indicator values are extracted.

    Returns
    -------
    dict
        ``sma_10``, ``sma_20``, ``ema_10``, ``ema_20``.
    """
    close = prices_df["close_price"].astype(float)
    dates = prices_df["trading_date"]

    def ex(s: pd.Series) -> float | None:
        return _extract_at(s, dates, target_date)

    return {
        "sma_10": ex(_sma(close, 10)),
        "sma_20": ex(_sma(close, 20)),
        "ema_10": ex(_ema(close, 10)),
        "ema_20": ex(_ema(close, 20)),
    }


def _compute_momentum_features(
    prices_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | None]:
    """
    Compute RSI 14, MACD line, signal, and histogram for ``target_date``.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV rows for a single ticker, sorted ascending by ``trading_date``.
    target_date : date
        The date for which indicator values are extracted.

    Returns
    -------
    dict
        ``rsi_14``, ``macd``, ``macd_signal``, ``macd_histogram``.
    """
    close = prices_df["close_price"].astype(float)
    dates = prices_df["trading_date"]

    def ex(s: pd.Series) -> float | None:
        return _extract_at(s, dates, target_date)

    macd_line, signal_line, histogram = _macd_lines(close)
    return {
        "rsi_14":         ex(_rsi(close, 14)),
        "macd":           ex(macd_line),
        "macd_signal":    ex(signal_line),
        "macd_histogram": ex(histogram),
    }


def _compute_volatility_features(
    prices_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | None]:
    """
    Compute Bollinger Bands (upper/lower/width), ATR 14, and 20-day rolling
    volatility for ``target_date``.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV rows for a single ticker, sorted ascending by ``trading_date``.
        Must contain ``close_price``, ``high_price``, ``low_price``.
    target_date : date
        The date for which indicator values are extracted.

    Returns
    -------
    dict
        ``bb_upper``, ``bb_lower``, ``bb_width``, ``atr_14``, ``volatility_20d``.
    """
    close = prices_df["close_price"].astype(float)
    high  = prices_df["high_price"].astype(float)
    low   = prices_df["low_price"].astype(float)
    dates = prices_df["trading_date"]

    def ex(s: pd.Series) -> float | None:
        return _extract_at(s, dates, target_date)

    bb_upper, bb_lower, bb_width = _bollinger_bands(close, window=20)
    daily_ret = close.pct_change()
    vol_20d   = daily_ret.rolling(window=20, min_periods=20).std(ddof=1)

    return {
        "bb_upper":       ex(bb_upper),
        "bb_lower":       ex(bb_lower),
        "bb_width":       ex(bb_width),
        "atr_14":         ex(_atr(high, low, close, 14)),
        "volatility_20d": ex(vol_20d),
    }


def _compute_return_features(
    prices_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | None]:
    """
    Compute 1-day, 5-day, and 10-day percentage price changes for ``target_date``.

    Column names use the ``price_chg_`` prefix to avoid collision with the
    ``return_*`` ML label columns produced by Phase 6.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV rows for a single ticker, sorted ascending by ``trading_date``.
    target_date : date
        The date for which values are extracted.

    Returns
    -------
    dict
        ``price_chg_1d``, ``price_chg_5d``, ``price_chg_10d``.
    """
    close = prices_df["close_price"].astype(float)
    dates = prices_df["trading_date"]

    def ex(s: pd.Series) -> float | None:
        return _extract_at(s, dates, target_date)

    return {
        "price_chg_1d":  ex(close.pct_change(1)),
        "price_chg_5d":  ex(close.pct_change(5)),
        "price_chg_10d": ex(close.pct_change(10)),
    }


def _compute_volume_features(
    prices_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | None]:
    """
    Compute volume change (%), 5-day average volume, and volume/average ratio
    for ``target_date``.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV rows for a single ticker, sorted ascending by ``trading_date``.
        Must contain ``volume`` and ``trading_date`` columns.
    target_date : date
        The date for which values are extracted.

    Returns
    -------
    dict
        ``volume_change_pct``, ``volume_avg_5d``, ``volume_ratio``.
    """
    volume = prices_df["volume"].astype(float)
    dates  = prices_df["trading_date"]

    def ex(s: pd.Series) -> float | None:
        return _extract_at(s, dates, target_date)

    vol_avg_5d = volume.rolling(window=5, min_periods=5).mean()
    vol_ratio  = volume / vol_avg_5d.mask(vol_avg_5d == 0.0)

    return {
        "volume_change_pct": ex(volume.pct_change(1)),
        "volume_avg_5d":     ex(vol_avg_5d),
        "volume_ratio":      ex(vol_ratio),
    }


def _compute_technical_features(
    prices_df: pd.DataFrame,
    target_date: date,
) -> dict[str, float | None]:
    """
    Orchestrate all technical indicator feature groups for one ticker.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV rows for a **single ticker**, sorted ascending by
        ``trading_date``.  Must contain ``trading_date``, ``open_price``,
        ``high_price``, ``low_price``, ``close_price``, and ``volume``.
    target_date : date
        The date for which indicator values are extracted.

    Returns
    -------
    dict
        All 19 technical feature key-value pairs (values may be ``None``
        when the series contains insufficient history at ``target_date``).
    """
    result: dict[str, float | None] = {}
    result.update(_compute_trend_features(prices_df, target_date))
    result.update(_compute_momentum_features(prices_df, target_date))
    result.update(_compute_volatility_features(prices_df, target_date))
    result.update(_compute_return_features(prices_df, target_date))
    result.update(_compute_volume_features(prices_df, target_date))
    return result


# ── FeatureEngineer ───────────────────────────────────────────────────────────

#: Calendar days of price history to load when technical indicators are needed.
#: 90 days ≈ 63 trading days — sufficient for the slowest indicator (MACD
#: signal line, which needs ~35 trading days of warm-up).
_PRICE_LOOKBACK_DAYS: int = 90


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

    def load_price_data(
        self,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV price data from the ``stock_prices`` table.

        Parameters
        ----------
        tickers : list[str] | None
            Ticker symbols to load.  ``None`` loads all available tickers.
        start_date : date | None
            Inclusive lower bound on ``trading_date``.  ``None`` imposes no
            lower bound.
        end_date : date | None
            Inclusive upper bound on ``trading_date``.  ``None`` imposes no
            upper bound.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``trading_date``, ``open_price``,
            ``high_price``, ``low_price``, ``close_price``,
            ``adjusted_close``, ``volume``.
            Sorted ascending by ``(ticker, trading_date)``.
            Returns an empty DataFrame when no rows are found.

        Raises
        ------
        DataLoadError
            On any database connectivity or query failure.
        """
        try:
            db = self._get_db()

            stmt = select(
                StockPrice.ticker.label("ticker"),
                StockPrice.trading_date.label("trading_date"),
                StockPrice.open_price.label("open_price"),
                StockPrice.high_price.label("high_price"),
                StockPrice.low_price.label("low_price"),
                StockPrice.close_price.label("close_price"),
                StockPrice.adjusted_close.label("adjusted_close"),
                StockPrice.volume.label("volume"),
            ).order_by(StockPrice.ticker, StockPrice.trading_date)

            if tickers:
                stmt = stmt.where(
                    StockPrice.ticker.in_([t.upper() for t in tickers])
                )
            if start_date is not None:
                stmt = stmt.where(StockPrice.trading_date >= start_date)
            if end_date is not None:
                stmt = stmt.where(StockPrice.trading_date <= end_date)

            with db.engine.connect() as conn:
                df = pd.read_sql(stmt, conn)

        except FeatureEngineeringError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"Failed to load price data from database: {exc}"
            ) from exc

        if df.empty:
            logger.debug(
                f"No price data found for tickers={tickers or 'all'} "
                f"({start_date} → {end_date})"
            )
        else:
            logger.debug(
                f"Loaded {len(df)} price rows for "
                f"{df['ticker'].nunique()} ticker(s) "
                f"({start_date} → {end_date})"
            )
        return df

    def generate_features(
        self,
        raw_df: pd.DataFrame,
        target_date: date,
        prices_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute per-ticker feature vectors for the given target date.

        For each ticker present in ``raw_df``, the full feature set is
        computed.  Tickers that have no articles *on* ``target_date`` are
        silently skipped (rolling and time features can still use historical
        data from the window).

        When ``prices_df`` is provided, 19 technical indicator columns are
        appended to each row.  Values for indicators that cannot be computed
        (due to insufficient price history) are left as ``None`` / NaN.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Output of :meth:`load_data`.  Must contain the columns documented
            there.
        target_date : date
            The date for which same-day features are computed (sentiment,
            source).  Rolling and time features look back from this date.
        prices_df : pd.DataFrame | None
            Optional OHLCV price DataFrame (output of :meth:`load_price_data`).
            When provided, technical indicator features are merged into each
            row.  When ``None``, technical feature columns are included with
            ``None`` values (backward-compatible).

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

            # ── Technical indicators ─────────────────────────────────────────
            tech: dict[str, float | None] = {
                col: None for col in TECHNICAL_FEATURE_COLUMNS
            }
            if prices_df is not None and not prices_df.empty:
                ticker_prices = prices_df[prices_df["ticker"] == ticker].copy()
                if not ticker_prices.empty:
                    ticker_prices = (
                        ticker_prices
                        .sort_values("trading_date")
                        .reset_index(drop=True)
                    )
                    tech.update(_compute_technical_features(ticker_prices, target_date))
            row.update(tech)

            rows.append(row)

        if not rows:
            raise FeatureGenerationError(
                f"No features generated for target_date={target_date}. "
                "No tickers had articles on that date in the loaded window."
            )

        features_df = pd.DataFrame(rows)[FEATURE_COLUMNS]
        logger.info(
            f"Generated {len(FEATURE_COLUMNS) - 2} features "
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

        Technical indicators are computed from a single price data load
        spanning ``[start_date − _PRICE_LOOKBACK_DAYS, end_date]``.  If the
        ``stock_prices`` table contains no data, technical columns are ``None``
        for all rows.

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

        # Load price data once for the full range (with indicator warm-up buffer)
        prices_df: pd.DataFrame | None = None
        try:
            price_start = start_date - timedelta(days=_PRICE_LOOKBACK_DAYS)
            loaded = self.load_price_data(
                tickers=tickers,
                start_date=price_start,
                end_date=end_date,
            )
            if not loaded.empty:
                prices_df = loaded
        except DataLoadError as exc:
            logger.warning(
                f"run_range: could not load price data ({exc}) — "
                "technical indicator columns will be None for all rows"
            )

        all_frames: list[pd.DataFrame] = []
        dates_processed = 0
        dates_skipped = 0

        current = start_date
        while current <= end_date:
            try:
                features_df = self.generate_features(
                    raw_df, current, prices_df=prices_df
                )
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

        # Load price data for technical indicator computation
        prices_df: pd.DataFrame | None = None
        try:
            price_start = target_date - timedelta(days=_PRICE_LOOKBACK_DAYS)
            loaded = self.load_price_data(
                tickers=tickers,
                start_date=price_start,
                end_date=target_date,
            )
            if not loaded.empty:
                prices_df = loaded
        except DataLoadError as exc:
            logger.warning(
                f"Could not load price data ({exc}) — "
                "technical indicator columns will be None"
            )

        features_df = self.generate_features(
            raw_df, target_date, prices_df=prices_df
        )

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
