"""
Data-access layer for financial-news-analytics.

``ArticleRepository`` owns all SQL reads and writes for ``news_articles``
and ``sentiment_results``. It is constructed with an active SQLAlchemy
``Session`` and a ``dialect_name`` so callers can swap in a SQLite-backed
session for testing without changing any business logic.

Upsert strategy
---------------
PostgreSQL (production)
    Uses ``INSERT … ON CONFLICT DO UPDATE … RETURNING id, url`` for true
    bulk upsert in a single round-trip per table.  The ``RETURNING`` clause
    provides the canonical ``article_id`` for every row — whether it was
    inserted or already existed — so sentiment rows can be linked
    immediately without a follow-up SELECT.

SQLite / other (tests)
    Falls back to a SELECT-then-INSERT/UPDATE loop.  Slightly slower but
    functionally identical and fully portable.

Usage
-----
    from src.storage.repository import ArticleRepository

    with db.get_session() as session:
        repo = ArticleRepository(session, dialect_name=db.engine.dialect.name)
        article_result, url_to_id = repo.upsert_articles(article_rows)
        sentiment_result = repo.upsert_sentiment_results(url_to_id, sentiment_rows, model_name)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.storage.models import NewsArticle, SentimentResult
from src.utils.logger import get_logger

UTC = timezone.utc
logger = get_logger(__name__)

# ── Column lists ──────────────────────────────────────────────────────────────

# Fields that belong to the news_articles table
_ARTICLE_FIELDS: tuple[str, ...] = (
    "ticker", "source_id", "source_name", "author",
    "title", "description", "url",
    "published_at", "content", "fetched_at",
)

# Fields that belong to the sentiment_results table (url is used as join key only)
_SENTIMENT_FIELDS: tuple[str, ...] = (
    "sentiment_label", "sentiment_score", "sentiment_confidence", "analysed_at",
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UpsertResult:
    """Summary of a single upsert operation."""

    inserted: int = 0
    updated:  int = 0
    skipped:  int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated

    def __str__(self) -> str:
        return (
            f"inserted={self.inserted} updated={self.updated} skipped={self.skipped}"
        )


# ── Type coercion helpers ─────────────────────────────────────────────────────

def _nan_to_none(value: Any) -> Any:
    """Convert NaN / pandas NA to None; leave everything else unchanged."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_datetime(value: Any) -> datetime | None:
    """
    Parse a datetime value from a CSV cell.

    Handles:
    - ``None`` / NaN               → ``None``
    - ``datetime`` objects         → returned as-is
    - ISO-8601 strings             → parsed to UTC-aware datetime
    - pandas ``Timestamp``         → converted to Python datetime
    """
    value = _nan_to_none(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _coerce_article_record(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise a raw CSV dict to an ``article``-insert-ready dict.

    - Keeps only columns that exist in ``news_articles``
    - Converts NaN → None for nullable text fields
    - Parses timestamp strings into timezone-aware ``datetime`` objects
    """
    return {
        "ticker":       str(_nan_to_none(raw.get("ticker")) or ""),
        "source_id":    str(_nan_to_none(raw.get("source_id")) or ""),
        "source_name":  _nan_to_none(raw.get("source_name")),
        "author":       _nan_to_none(raw.get("author")),
        "title":        str(_nan_to_none(raw.get("title")) or ""),
        "description":  _nan_to_none(raw.get("description")),
        "url":          str(_nan_to_none(raw.get("url")) or ""),
        "published_at": _to_datetime(raw.get("published_at")),
        "content":      _nan_to_none(raw.get("content")),
        "fetched_at":   _to_datetime(raw.get("fetched_at")),
    }


def _coerce_sentiment_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw CSV dict to a ``sentiment``-insert-ready dict."""
    return {
        "url":                  str(_nan_to_none(raw.get("url")) or ""),
        "sentiment_label":      str(_nan_to_none(raw.get("sentiment_label")) or "neutral"),
        "sentiment_score":      int(_nan_to_none(raw.get("sentiment_score")) or 0),
        "sentiment_confidence": float(_nan_to_none(raw.get("sentiment_confidence")) or 0.0),
        "analysed_at":          _to_datetime(raw.get("analysed_at")),
    }


# ── Repository ────────────────────────────────────────────────────────────────

class ArticleRepository:
    """
    Data-access object for ``news_articles`` and ``sentiment_results``.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.  The caller owns transaction management
        (commit / rollback) via ``DatabaseManager.get_session()``.
    dialect_name : str
        SQLAlchemy dialect name: ``"postgresql"`` or ``"sqlite"``.
        Used to select the appropriate upsert implementation.
        Default: ``"postgresql"``.
    """

    def __init__(
        self,
        session: Session,
        dialect_name: str = "postgresql",
    ) -> None:
        self._session = session
        self._dialect = dialect_name.lower()

    # ── Articles ──────────────────────────────────────────────────────────────

    def upsert_articles(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[UpsertResult, dict[str, int]]:
        """
        Upsert a batch of news-article rows into ``news_articles``.

        Deduplication is done on ``url``.  On conflict, the ``title``,
        ``description``, and ``fetched_at`` columns are refreshed with the
        latest values (the URL is the stable identity; everything else may
        be corrected over time).

        Parameters
        ----------
        records : list[dict]
            Raw rows from a Phase 2 processed CSV (all 14+ columns accepted;
            extra columns are silently ignored).

        Returns
        -------
        result : UpsertResult
            Insert / update / skip counts.
        url_to_id : dict[str, int]
            Mapping from article URL → ``news_articles.id``.
            Required to link sentiment results in the next step.
        """
        if not records:
            return UpsertResult(), {}

        clean = [_coerce_article_record(r) for r in records]
        # Drop rows without a URL — nothing useful can be stored
        valid = [r for r in clean if r["url"]]
        skipped = len(clean) - len(valid)

        if not valid:
            logger.warning("upsert_articles: all records have empty URLs — nothing inserted")
            return UpsertResult(skipped=len(clean)), {}

        logger.debug(
            f"Upserting {len(valid)} articles "
            f"(dialect={self._dialect}, skipped={skipped})"
        )

        if self._dialect == "postgresql":
            result, url_to_id = self._pg_upsert_articles(valid)
        else:
            result, url_to_id = self._generic_upsert_articles(valid)

        result = UpsertResult(
            inserted=result.inserted,
            updated=result.updated,
            skipped=result.skipped + skipped,
        )
        logger.debug(f"upsert_articles done: {result}")
        return result, url_to_id

    def _pg_upsert_articles(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[UpsertResult, dict[str, int]]:
        """PostgreSQL path — single-round-trip bulk upsert with RETURNING."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: PLC0415

        stmt = pg_insert(NewsArticle).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["url"],
            set_={
                "source_name":  stmt.excluded.source_name,
                "title":        stmt.excluded.title,
                "description":  stmt.excluded.description,
                "fetched_at":   stmt.excluded.fetched_at,
            },
        ).returning(NewsArticle.id, NewsArticle.url)

        rows = self._session.execute(stmt).fetchall()
        url_to_id = {row.url: row.id for row in rows}
        # PostgreSQL does not distinguish INSERT vs UPDATE in a plain ON CONFLICT;
        # report total as inserted for simplicity.
        return UpsertResult(inserted=len(rows)), url_to_id

    def _generic_upsert_articles(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[UpsertResult, dict[str, int]]:
        """Generic path — SELECT + INSERT/UPDATE per record (SQLite-safe)."""
        inserted = 0
        updated = 0
        url_to_id: dict[str, int] = {}

        for rec in records:
            url = rec["url"]
            existing: NewsArticle | None = (
                self._session.query(NewsArticle).filter_by(url=url).first()
            )
            if existing is None:
                article = NewsArticle(**rec)
                self._session.add(article)
                self._session.flush()   # populate auto-generated id
                url_to_id[url] = article.id
                inserted += 1
            else:
                # Refresh mutable fields only
                existing.source_name  = rec.get("source_name",  existing.source_name)
                existing.title        = rec.get("title",        existing.title)
                existing.description  = rec.get("description",  existing.description)
                existing.fetched_at   = rec.get("fetched_at",   existing.fetched_at)
                url_to_id[url] = existing.id
                updated += 1

        self._session.flush()
        return UpsertResult(inserted=inserted, updated=updated), url_to_id

    # ── Sentiment results ─────────────────────────────────────────────────────

    def upsert_sentiment_results(
        self,
        url_to_id: dict[str, int],
        records: list[dict[str, Any]],
        model_name: str,
    ) -> UpsertResult:
        """
        Upsert sentiment results linked to previously upserted articles.

        Deduplication is on ``(article_id, model_name)``.  On conflict the
        label, score, confidence, and ``analysed_at`` are refreshed.

        Parameters
        ----------
        url_to_id : dict[str, int]
            URL → article_id mapping returned by ``upsert_articles()``.
        records : list[dict]
            Raw Phase 2 rows (must contain ``url`` and the four sentiment cols).
        model_name : str
            Name of the FinBERT model used (e.g. ``"ProsusAI/finbert"``).

        Returns
        -------
        UpsertResult
        """
        if not records or not url_to_id:
            return UpsertResult(skipped=len(records) if records else 0)

        clean = [_coerce_sentiment_record(r) for r in records]
        # Attach article IDs; drop any row whose URL isn't in the mapping
        linked: list[dict[str, Any]] = []
        skipped = 0
        for rec in clean:
            article_id = url_to_id.get(rec["url"])
            if article_id is None:
                skipped += 1
                continue
            linked.append({
                "article_id":           article_id,
                "model_name":           model_name,
                "sentiment_label":      rec["sentiment_label"],
                "sentiment_score":      rec["sentiment_score"],
                "sentiment_confidence": rec["sentiment_confidence"],
                "analysed_at":          rec["analysed_at"],
            })

        if not linked:
            logger.warning("upsert_sentiment_results: no records could be linked to articles")
            return UpsertResult(skipped=skipped)

        logger.debug(
            f"Upserting {len(linked)} sentiment rows "
            f"(model={model_name!r}, dialect={self._dialect}, skipped={skipped})"
        )

        if self._dialect == "postgresql":
            result = self._pg_upsert_sentiment(linked)
        else:
            result = self._generic_upsert_sentiment(linked)

        result = UpsertResult(
            inserted=result.inserted,
            updated=result.updated,
            skipped=result.skipped + skipped,
        )
        logger.debug(f"upsert_sentiment_results done: {result}")
        return result

    def _pg_upsert_sentiment(
        self,
        records: list[dict[str, Any]],
    ) -> UpsertResult:
        from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: PLC0415

        stmt = pg_insert(SentimentResult).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_sentiment_article_model",
            set_={
                "sentiment_label":      stmt.excluded.sentiment_label,
                "sentiment_score":      stmt.excluded.sentiment_score,
                "sentiment_confidence": stmt.excluded.sentiment_confidence,
                "analysed_at":          stmt.excluded.analysed_at,
            },
        )
        self._session.execute(stmt)
        return UpsertResult(inserted=len(records))

    def _generic_upsert_sentiment(
        self,
        records: list[dict[str, Any]],
    ) -> UpsertResult:
        inserted = 0
        updated = 0

        for rec in records:
            existing: SentimentResult | None = (
                self._session.query(SentimentResult)
                .filter_by(
                    article_id=rec["article_id"],
                    model_name=rec["model_name"],
                )
                .first()
            )
            if existing is None:
                self._session.add(SentimentResult(**rec))
                inserted += 1
            else:
                existing.sentiment_label      = rec["sentiment_label"]
                existing.sentiment_score      = rec["sentiment_score"]
                existing.sentiment_confidence = rec["sentiment_confidence"]
                existing.analysed_at          = rec["analysed_at"]
                updated += 1

        self._session.flush()
        return UpsertResult(inserted=inserted, updated=updated)

    # ── Queries ───────────────────────────────────────────────────────────────

    def count_articles(self, ticker: str | None = None) -> int:
        """Return the total number of articles, optionally filtered by ticker."""
        q = self._session.query(NewsArticle)
        if ticker:
            q = q.filter(NewsArticle.ticker == ticker.upper())
        return q.count()

    def count_sentiment_results(self, model_name: str | None = None) -> int:
        """Return the total number of sentiment results, optionally filtered by model."""
        q = self._session.query(SentimentResult)
        if model_name:
            q = q.filter(SentimentResult.model_name == model_name)
        return q.count()

    def get_articles_by_ticker(self, ticker: str) -> list[NewsArticle]:
        """Return all articles for a given ticker, newest first."""
        return (
            self._session.query(NewsArticle)
            .filter(NewsArticle.ticker == ticker.upper())
            .order_by(NewsArticle.published_at.desc())
            .all()
        )

    def get_articles_without_sentiment(
        self,
        model_name: str | None = None,
    ) -> list[NewsArticle]:
        """
        Return articles that have no corresponding sentiment result.

        Useful for identifying articles that need to be re-run through
        Phase 2 after a model upgrade or partial failure.
        """
        q = (
            self._session.query(NewsArticle)
            .outerjoin(
                SentimentResult,
                NewsArticle.id == SentimentResult.article_id,
            )
        )
        if model_name:
            q = q.filter(
                (SentimentResult.id == None) |  # noqa: E711
                (SentimentResult.model_name != model_name)
            )
        else:
            q = q.filter(SentimentResult.id == None)  # noqa: E711
        return q.all()

    def get_sentiment_distribution(
        self,
        ticker: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, int]:
        """
        Return label → count distribution across all sentiment results.

        Parameters
        ----------
        ticker : str | None
            If provided, only articles for this ticker are counted.
        model_name : str | None
            If provided, only results from this model are counted.
        """
        from sqlalchemy import func as sa_func  # noqa: PLC0415

        q = (
            self._session.query(
                SentimentResult.sentiment_label,
                sa_func.count(SentimentResult.id).label("count"),
            )
            .join(NewsArticle, NewsArticle.id == SentimentResult.article_id)
        )
        if ticker:
            q = q.filter(NewsArticle.ticker == ticker.upper())
        if model_name:
            q = q.filter(SentimentResult.model_name == model_name)

        rows = q.group_by(SentimentResult.sentiment_label).all()
        return {row.sentiment_label: row.count for row in rows}
