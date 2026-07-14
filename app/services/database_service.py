"""
Read-oriented data-access service for the Streamlit presentation layer.

This module never re-implements storage logic. It reuses:

* :class:`src.storage.database.DatabaseManager` for connections/sessions
* :class:`src.storage.models` ORM classes for schema-aware queries
* :class:`src.storage.repository.ArticleRepository`
* :class:`src.prices.price_repository.PriceRepository`
* :class:`src.realtime.prediction_repository.PredictionRepository`

Repositories cover simple counts/lookups. For the richer, filterable
DataFrame reads the dashboard needs (date ranges, multi-ticker, search,
sorting), this module builds SQLAlchemy ``select()`` statements directly
against the existing ORM models and loads them with ``pandas.read_sql`` —
this is a read-only presentation concern, not new business logic, and it
never touches the write path (upserts stay exclusively in the repositories).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import select

from src.prices.price_repository import PriceRepository
from src.realtime.prediction_repository import PredictionRepository
from src.storage.database import DatabaseConnectionError, DatabaseManager
from src.storage.models import NewsArticle, Prediction, SentimentResult
from src.storage.repository import ArticleRepository
from src.utils.config import settings

_CACHE_TTL = 60  # seconds — dashboard data doesn't need to be perfectly live


# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_db_manager() -> DatabaseManager:
    """Return a cached :class:`DatabaseManager` bound to ``settings.database_url``."""
    if not settings.database_url:
        raise DatabaseConnectionError(
            "DATABASE_URL is not configured. Set it in your .env file."
        )
    return DatabaseManager(settings.database_url)


def database_status() -> dict[str, Any]:
    """Return ``{"connected": bool, "error": str | None, "url": str | None}``."""
    try:
        db = get_db_manager()
        db.verify_connection()
        return {"connected": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"connected": False, "error": str(exc)}


# ── Counts (delegated straight to repositories) ────────────────────────────────

@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def count_articles(ticker: str | None = None) -> int:
    db = get_db_manager()
    with db.get_session() as session:
        return ArticleRepository(session, dialect_name=db.engine.dialect.name).count_articles(ticker)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def count_sentiment_results(model_name: str | None = None) -> int:
    db = get_db_manager()
    with db.get_session() as session:
        return ArticleRepository(
            session, dialect_name=db.engine.dialect.name
        ).count_sentiment_results(model_name)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def count_prices(ticker: str | None = None) -> int:
    db = get_db_manager()
    with db.get_session() as session:
        return PriceRepository(session, dialect_name=db.engine.dialect.name).count_prices(ticker)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def count_predictions(ticker: str | None = None) -> int:
    db = get_db_manager()
    with db.get_session() as session:
        return PredictionRepository(session).count_predictions(ticker)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_sentiment_distribution(
    ticker: str | None = None, model_name: str | None = None
) -> dict[str, int]:
    db = get_db_manager()
    with db.get_session() as session:
        return ArticleRepository(
            session, dialect_name=db.engine.dialect.name
        ).get_sentiment_distribution(ticker, model_name)


# ── DataFrame reads ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_latest_prediction(ticker: str | None = None) -> dict[str, Any] | None:
    db = get_db_manager()
    with db.get_session() as session:
        stmt = select(Prediction).order_by(Prediction.created_at.desc())
        if ticker:
            stmt = stmt.where(Prediction.ticker == ticker.upper())
        row = session.execute(stmt.limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return {
            "ticker": row.ticker,
            "prediction": row.prediction,
            "confidence": row.confidence,
            "buy_probability": row.buy_probability,
            "hold_probability": row.hold_probability,
            "sell_probability": row.sell_probability,
            "headline": row.headline,
            "published_at": row.published_at,
            "created_at": row.created_at,
        }


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_latest_article(ticker: str | None = None) -> dict[str, Any] | None:
    db = get_db_manager()
    with db.get_session() as session:
        stmt = (
            select(NewsArticle, SentimentResult)
            .outerjoin(SentimentResult, NewsArticle.id == SentimentResult.article_id)
            .order_by(NewsArticle.published_at.desc())
        )
        if ticker:
            stmt = stmt.where(NewsArticle.ticker == ticker.upper())
        row = session.execute(stmt.limit(1)).first()
        if row is None:
            return None
        article, sentiment = row
        return {
            "ticker": article.ticker,
            "title": article.title,
            "description": article.description,
            "source_name": article.source_name,
            "url": article.url,
            "published_at": article.published_at,
            "sentiment_label": sentiment.sentiment_label if sentiment else None,
            "sentiment_confidence": sentiment.sentiment_confidence if sentiment else None,
        }


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_ticker_distribution() -> pd.DataFrame:
    """Article count per ticker."""
    db = get_db_manager()
    with db.get_session() as session:
        stmt = select(NewsArticle.ticker, NewsArticle.id)
        df = pd.read_sql(stmt, session.bind)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "article_count"])
    out = df.groupby("ticker", as_index=False)["id"].count()
    out.columns = ["ticker", "article_count"]
    return out.sort_values("article_count", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_articles_df(
    ticker: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Recent articles joined with their FinBERT sentiment, newest first."""
    db = get_db_manager()
    with db.get_session() as session:
        stmt = (
            select(
                NewsArticle.ticker,
                NewsArticle.title,
                NewsArticle.description,
                NewsArticle.source_name,
                NewsArticle.url,
                NewsArticle.published_at,
                SentimentResult.sentiment_label,
                SentimentResult.sentiment_score,
                SentimentResult.sentiment_confidence,
            )
            .outerjoin(SentimentResult, NewsArticle.id == SentimentResult.article_id)
            .order_by(NewsArticle.published_at.desc())
            .limit(limit)
        )
        if ticker:
            stmt = stmt.where(NewsArticle.ticker == ticker.upper())
        return pd.read_sql(stmt, session.bind)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_sentiment_articles_df(
    tickers: tuple[str, ...] | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Articles + sentiment for analytics aggregation (Sentiment Analytics page)."""
    db = get_db_manager()
    with db.get_session() as session:
        stmt = (
            select(
                NewsArticle.ticker,
                NewsArticle.title,
                NewsArticle.published_at,
                SentimentResult.sentiment_label,
                SentimentResult.sentiment_score,
                SentimentResult.sentiment_confidence,
            )
            .join(SentimentResult, NewsArticle.id == SentimentResult.article_id)
            .order_by(NewsArticle.published_at.desc())
        )
        if tickers:
            stmt = stmt.where(NewsArticle.ticker.in_([t.upper() for t in tickers]))
        if start:
            stmt = stmt.where(NewsArticle.published_at >= datetime.combine(start, datetime.min.time()))
        if end:
            stmt = stmt.where(NewsArticle.published_at <= datetime.combine(end, datetime.max.time()))
        if limit:
            stmt = stmt.limit(limit)
        return pd.read_sql(stmt, session.bind)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_predictions_df(
    tickers: tuple[str, ...] | None = None,
    predictions: tuple[str, ...] | None = None,
    start: date | None = None,
    end: date | None = None,
    min_confidence: float = 0.0,
    search: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Filtered prediction history for the Historical Predictions page."""
    db = get_db_manager()
    with db.get_session() as session:
        stmt = select(Prediction).order_by(Prediction.created_at.desc())
        if tickers:
            stmt = stmt.where(Prediction.ticker.in_([t.upper() for t in tickers]))
        if predictions:
            stmt = stmt.where(Prediction.prediction.in_([p.upper() for p in predictions]))
        if start:
            stmt = stmt.where(Prediction.created_at >= datetime.combine(start, datetime.min.time()))
        if end:
            stmt = stmt.where(Prediction.created_at <= datetime.combine(end, datetime.max.time()))
        if min_confidence:
            stmt = stmt.where(Prediction.confidence >= min_confidence)
        if search:
            stmt = stmt.where(Prediction.headline.ilike(f"%{search}%"))
        if limit:
            stmt = stmt.limit(limit)
        df = pd.read_sql(stmt, session.bind)
    return df


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_prices_df(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """OHLCV history for *ticker*, oldest first."""
    db = get_db_manager()
    with db.get_session() as session:
        repo = PriceRepository(session, dialect_name=db.engine.dialect.name)
        if start and end:
            rows = repo.get_price_range(ticker, start, end)
        else:
            rows = repo.get_prices_by_ticker(ticker)
        return pd.DataFrame(
            [
                {
                    "trading_date": r.trading_date,
                    "open_price": r.open_price,
                    "high_price": r.high_price,
                    "low_price": r.low_price,
                    "close_price": r.close_price,
                    "adjusted_close": r.adjusted_close,
                    "volume": r.volume,
                }
                for r in rows
            ]
        )


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_recent_activity(limit: int = 25) -> pd.DataFrame:
    """Union of the most recent articles and predictions for a timeline widget."""
    db = get_db_manager()
    with db.get_session() as session:
        articles = pd.read_sql(
            select(
                NewsArticle.ticker,
                NewsArticle.title.label("detail"),
                NewsArticle.published_at.label("timestamp"),
            )
            .order_by(NewsArticle.published_at.desc())
            .limit(limit),
            session.bind,
        )
        articles["event_type"] = "News Article"

        preds = pd.read_sql(
            select(
                Prediction.ticker,
                Prediction.prediction.label("detail"),
                Prediction.created_at.label("timestamp"),
            )
            .order_by(Prediction.created_at.desc())
            .limit(limit),
            session.bind,
        )
        preds["event_type"] = "Prediction"

    combined = pd.concat([articles, preds], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    return combined.sort_values("timestamp", ascending=False).head(limit).reset_index(drop=True)


def clear_all_caches() -> None:
    """Invalidate every cached read in this module (used by manual refresh buttons)."""
    for fn in (
        count_articles,
        count_sentiment_results,
        count_prices,
        count_predictions,
        get_sentiment_distribution,
        get_latest_prediction,
        get_latest_article,
        get_ticker_distribution,
        get_articles_df,
        get_sentiment_articles_df,
        get_predictions_df,
        get_prices_df,
        get_recent_activity,
    ):
        fn.clear()
