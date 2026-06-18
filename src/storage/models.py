"""
SQLAlchemy ORM models for the financial-news-analytics database.

Tables
------
- ``news_articles``     — raw article metadata fetched from Finnhub (Phase 1)
- ``sentiment_results`` — FinBERT sentiment scores linked to articles (Phase 2)

Schema design
-------------
- ``news_articles`` is deduplicated on ``url`` (unique constraint).
  The same URL may appear in multiple tickers' newsfeeds; only the first
  fetch is retained in the canonical row. The ``ticker`` column stores the
  ticker that first triggered the fetch.

- ``sentiment_results`` is deduplicated on ``(article_id, model_name)``.
  Running Phase 2 with a different model produces a separate row per article;
  re-running with the same model updates the existing row in-place.

- Both tables carry a ``created_at`` server-side default so inserts never
  need to supply a timestamp explicitly.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ── news_articles ─────────────────────────────────────────────────────────────

class NewsArticle(Base):
    """
    One row per unique news article URL.

    Maps directly to the Phase 1 ingestion output schema with the addition
    of a surrogate primary key and an audit ``created_at`` column.
    """

    __tablename__ = "news_articles"

    id:           Mapped[int]                 = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True, autoincrement=True,
    )
    ticker:       Mapped[str]                 = mapped_column(String(10),  nullable=False)
    source_id:    Mapped[str]                 = mapped_column(String(64),  nullable=False)
    source_name:  Mapped[Optional[str]]       = mapped_column(String(255), nullable=True)
    author:       Mapped[Optional[str]]       = mapped_column(Text,        nullable=True)
    title:        Mapped[str]                 = mapped_column(Text,        nullable=False)
    description:  Mapped[Optional[str]]       = mapped_column(Text,        nullable=True)
    url:          Mapped[str]                 = mapped_column(Text,        nullable=False)
    published_at: Mapped[Optional[datetime]]  = mapped_column(DateTime(timezone=True), nullable=True)
    content:      Mapped[Optional[str]]       = mapped_column(Text,        nullable=True)
    fetched_at:   Mapped[Optional[datetime]]  = mapped_column(DateTime(timezone=True), nullable=True)
    created_at:   Mapped[datetime]            = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # One article → many sentiment results (one per model)
    sentiment_results: Mapped[List[SentimentResult]] = relationship(
        "SentimentResult",
        back_populates="article",
        cascade="all, delete-orphan",
        lazy="select",
    )

    __table_args__ = (
        UniqueConstraint("url", name="uq_news_articles_url"),
        Index("ix_news_articles_ticker",           "ticker"),
        Index("ix_news_articles_ticker_published", "ticker", "published_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<NewsArticle id={self.id} ticker={self.ticker!r} "
            f"source_id={self.source_id!r}>"
        )


# ── sentiment_results ─────────────────────────────────────────────────────────

class SentimentResult(Base):
    """
    One row per (article, model) pair.

    Stores the output of Phase 2 FinBERT inference. Re-running sentiment
    analysis with the same model updates the existing row; using a different
    model inserts a new row for the same article.
    """

    __tablename__ = "sentiment_results"

    id:                   Mapped[int]               = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True, autoincrement=True,
    )
    article_id:           Mapped[int]               = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name:           Mapped[str]               = mapped_column(String(255), nullable=False)
    sentiment_label:      Mapped[str]               = mapped_column(String(10),  nullable=False)
    sentiment_score:      Mapped[int]               = mapped_column(SmallInteger, nullable=False)
    sentiment_confidence: Mapped[float]             = mapped_column(Float,        nullable=False)
    analysed_at:          Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at:           Mapped[datetime]           = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    article: Mapped[NewsArticle] = relationship(
        "NewsArticle",
        back_populates="sentiment_results",
    )

    __table_args__ = (
        UniqueConstraint(
            "article_id", "model_name",
            name="uq_sentiment_article_model",
        ),
        Index("ix_sentiment_results_article_id", "article_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<SentimentResult id={self.id} article_id={self.article_id} "
            f"label={self.sentiment_label!r} score={self.sentiment_score}>"
        )
