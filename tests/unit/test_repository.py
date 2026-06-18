"""
Unit tests for src.storage — DatabaseManager, ORM models, and ArticleRepository.

All tests use an in-memory SQLite database so no PostgreSQL instance is needed.
SQLite exercises the generic (non-PostgreSQL) upsert path; the PostgreSQL path
is covered by patching the dialect name and mocking the pg_insert call.

Test organisation
-----------------
TestDatabaseManager        — engine creation, table DDL, session context manager
TestOrmModels              — NewsArticle / SentimentResult construction + repr
TestCoercionHelpers        — _nan_to_none, _to_datetime, _coerce_* helpers
TestUpsertResult           — UpsertResult dataclass properties
TestArticleRepositoryInit  — constructor / dialect wiring
TestUpsertArticles         — insert, update, skip, batch, empty URL guard
TestUpsertSentimentResults — insert, update, link to articles, skipped rows
TestQueries                — count_articles, count_sentiment_results,
                             get_articles_by_ticker,
                             get_articles_without_sentiment,
                             get_sentiment_distribution
TestPostgresDialectPath    — verify PostgreSQL upsert code path is invoked
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.storage.database import (
    DatabaseConnectionError,
    DatabaseManager,
    SchemaError,
)
from src.storage.models import Base, NewsArticle, SentimentResult
from src.storage.repository import (
    ArticleRepository,
    UpsertResult,
    _coerce_article_record,
    _coerce_sentiment_record,
    _nan_to_none,
    _to_datetime,
)

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def sqlite_engine():
    """
    Fresh in-memory SQLite engine per test.

    Using scope="function" + sqlite:///:memory: means each test starts with
    a brand-new, empty database — no cross-test contamination, no rollback
    management needed.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(sqlite_engine):
    """Plain session bound to the per-test in-memory engine."""
    session = Session(sqlite_engine)
    yield session
    session.close()


@pytest.fixture
def repo(db_session) -> ArticleRepository:
    return ArticleRepository(db_session, dialect_name="sqlite")


@pytest.fixture
def sample_article() -> dict:
    return {
        "ticker":      "AAPL",
        "source_id":   "12345",
        "source_name": "Reuters",
        "author":      None,
        "title":       "Apple beats Q2 earnings estimates",
        "description": "Record iPhone sales drove revenue above expectations.",
        "url":         "https://example.com/aapl-q2",
        "published_at": datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC),
        "content":     None,
        "fetched_at":  datetime(2026, 6, 15, 13, 0, 0, tzinfo=UTC),
        "sentiment_label":      "positive",
        "sentiment_score":      1,
        "sentiment_confidence": 0.97,
        "analysed_at": datetime(2026, 6, 15, 14, 0, 0, tzinfo=UTC),
    }


@pytest.fixture
def sample_articles() -> list[dict]:
    base = datetime(2026, 6, 15, tzinfo=UTC)
    return [
        {
            "ticker":       "AAPL",
            "source_id":    f"100{i}",
            "source_name":  "Yahoo",
            "author":       None,
            "title":        f"AAPL headline {i}",
            "description":  f"Description {i}",
            "url":          f"https://example.com/aapl-{i}",
            "published_at": base,
            "content":      None,
            "fetched_at":   base,
            "sentiment_label":      ["positive", "neutral", "negative"][i % 3],
            "sentiment_score":      [1, 0, -1][i % 3],
            "sentiment_confidence": 0.9 - i * 0.01,
            "analysed_at":  base,
        }
        for i in range(5)
    ]


# ── TestDatabaseManager ───────────────────────────────────────────────────────

class TestDatabaseManager:
    def test_empty_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="database_url must not be empty"):
            DatabaseManager("")

    def test_engine_is_lazy_and_cached(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        assert db._engine is None
        engine1 = db.engine
        engine2 = db.engine
        assert engine1 is engine2

    def test_session_factory_is_lazy_and_cached(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        sf1 = db.session_factory
        sf2 = db.session_factory
        assert sf1 is sf2

    def test_create_tables_creates_news_articles(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        db.create_tables()
        with db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = {row[0] for row in result}
        assert "news_articles" in tables
        assert "sentiment_results" in tables

    def test_create_tables_is_idempotent(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        db.create_tables()
        db.create_tables()  # must not raise

    def test_verify_connection_passes_for_sqlite(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        db.verify_connection()  # must not raise

    def test_verify_connection_raises_on_bad_url(self) -> None:
        db = DatabaseManager("postgresql://bad:bad@nonexistent-host:9999/db")
        with pytest.raises(DatabaseConnectionError):
            db.verify_connection()

    def test_get_session_commits_on_clean_exit(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        db.create_tables()
        with db.get_session() as session:
            session.add(NewsArticle(
                ticker="AAPL", source_id="X1", title="Test",
                url="https://x.com/1",
            ))
        with db.get_session() as session:
            count = session.query(NewsArticle).count()
        assert count == 1

    def test_get_session_rolls_back_on_exception(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        db.create_tables()
        with pytest.raises(RuntimeError, match="intentional"):
            with db.get_session() as session:
                session.add(NewsArticle(
                    ticker="AAPL", source_id="X2", title="Should rollback",
                    url="https://x.com/2",
                ))
                raise RuntimeError("intentional")
        with db.get_session() as session:
            count = session.query(NewsArticle).count()
        assert count == 0

    def test_dispose_does_not_raise(self) -> None:
        db = DatabaseManager("sqlite:///:memory:")
        _ = db.engine
        db.dispose()  # must not raise

    def test_safe_url_redacts_password(self) -> None:
        db = DatabaseManager("postgresql://user:secret@localhost:5432/db")
        safe = db._safe_url()
        assert "secret" not in safe
        assert "user" in safe


# ── TestOrmModels ─────────────────────────────────────────────────────────────

class TestOrmModels:
    def test_news_article_repr(self) -> None:
        a = NewsArticle(id=1, ticker="TSLA", source_id="abc")
        assert "TSLA" in repr(a)
        assert "abc" in repr(a)

    def test_sentiment_result_repr(self) -> None:
        s = SentimentResult(
            id=1, article_id=42,
            sentiment_label="positive", sentiment_score=1,
        )
        assert "42" in repr(s)
        assert "positive" in repr(s)

    def test_news_article_persists_to_sqlite(self, db_session) -> None:
        article = NewsArticle(
            ticker="NVDA", source_id="99",
            title="NVIDIA beats estimates",
            url="https://example.com/nvda",
        )
        db_session.add(article)
        db_session.flush()
        assert article.id is not None

    def test_sentiment_result_links_to_article(self, db_session) -> None:
        article = NewsArticle(
            ticker="MSFT", source_id="88",
            title="Microsoft Azure growth",
            url="https://example.com/msft",
        )
        db_session.add(article)
        db_session.flush()

        sentiment = SentimentResult(
            article_id=article.id,
            model_name="ProsusAI/finbert",
            sentiment_label="positive",
            sentiment_score=1,
            sentiment_confidence=0.95,
        )
        db_session.add(sentiment)
        db_session.flush()
        assert sentiment.id is not None
        assert sentiment.article_id == article.id


# ── TestCoercionHelpers ───────────────────────────────────────────────────────

class TestNanToNone:
    def test_none_returns_none(self) -> None:
        assert _nan_to_none(None) is None

    def test_float_nan_returns_none(self) -> None:
        assert _nan_to_none(float("nan")) is None

    def test_pandas_nan_returns_none(self) -> None:
        assert _nan_to_none(pd.NA) is None

    def test_numpy_nan_returns_none(self) -> None:
        import numpy as np
        assert _nan_to_none(np.nan) is None

    def test_zero_preserved(self) -> None:
        assert _nan_to_none(0) == 0

    def test_empty_string_preserved(self) -> None:
        assert _nan_to_none("") == ""

    def test_valid_string_preserved(self) -> None:
        assert _nan_to_none("hello") == "hello"

    def test_valid_int_preserved(self) -> None:
        assert _nan_to_none(42) == 42


class TestToDatetime:
    def test_none_returns_none(self) -> None:
        assert _to_datetime(None) is None

    def test_float_nan_returns_none(self) -> None:
        assert _to_datetime(float("nan")) is None

    def test_datetime_object_returned_as_is(self) -> None:
        dt = datetime(2026, 6, 15, tzinfo=UTC)
        result = _to_datetime(dt)
        assert result == dt

    def test_iso_string_parsed(self) -> None:
        result = _to_datetime("2026-06-15 12:00:00+00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 6

    def test_pandas_timestamp_converted(self) -> None:
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        result = _to_datetime(ts)
        assert isinstance(result, datetime)

    def test_invalid_string_returns_none(self) -> None:
        result = _to_datetime("not-a-date")
        assert result is None


class TestCoerceArticleRecord:
    def test_extracts_article_fields(self, sample_article) -> None:
        result = _coerce_article_record(sample_article)
        assert result["ticker"] == "AAPL"
        assert result["source_id"] == "12345"
        assert result["title"] == "Apple beats Q2 earnings estimates"

    def test_sentiment_fields_not_in_output(self, sample_article) -> None:
        result = _coerce_article_record(sample_article)
        assert "sentiment_label" not in result
        assert "sentiment_score" not in result

    def test_nan_author_becomes_none(self) -> None:
        record = {"ticker": "X", "source_id": "1", "title": "t",
                  "url": "u", "author": float("nan")}
        result = _coerce_article_record(record)
        assert result["author"] is None

    def test_timestamp_string_parsed(self) -> None:
        record = {
            "ticker": "X", "source_id": "1", "title": "t", "url": "u",
            "published_at": "2026-06-15 12:00:00+00:00",
        }
        result = _coerce_article_record(record)
        assert isinstance(result["published_at"], datetime)


class TestCoerceSentimentRecord:
    def test_extracts_sentiment_fields(self, sample_article) -> None:
        result = _coerce_sentiment_record(sample_article)
        assert result["sentiment_label"] == "positive"
        assert result["sentiment_score"] == 1
        assert result["sentiment_confidence"] == pytest.approx(0.97)
        assert result["url"] == "https://example.com/aapl-q2"

    def test_nan_confidence_defaults_to_zero(self) -> None:
        record = {"url": "u", "sentiment_label": "neutral",
                  "sentiment_score": 0, "sentiment_confidence": float("nan")}
        result = _coerce_sentiment_record(record)
        assert result["sentiment_confidence"] == 0.0

    def test_missing_label_defaults_to_neutral(self) -> None:
        record = {"url": "u"}
        result = _coerce_sentiment_record(record)
        assert result["sentiment_label"] == "neutral"


# ── TestUpsertResult ──────────────────────────────────────────────────────────

class TestUpsertResult:
    def test_total_is_inserted_plus_updated(self) -> None:
        r = UpsertResult(inserted=3, updated=2, skipped=1)
        assert r.total == 5

    def test_defaults_are_zero(self) -> None:
        r = UpsertResult()
        assert r.inserted == 0
        assert r.updated == 0
        assert r.skipped == 0
        assert r.total == 0

    def test_is_immutable(self) -> None:
        r = UpsertResult(inserted=1)
        with pytest.raises(Exception):
            r.inserted = 99  # type: ignore[misc]

    def test_str_representation(self) -> None:
        r = UpsertResult(inserted=2, updated=1, skipped=0)
        s = str(r)
        assert "inserted=2" in s
        assert "updated=1" in s


# ── TestArticleRepositoryInit ─────────────────────────────────────────────────

class TestArticleRepositoryInit:
    def test_defaults_to_postgresql_dialect(self, db_session) -> None:
        repo = ArticleRepository(db_session)
        assert repo._dialect == "postgresql"

    def test_accepts_sqlite_dialect(self, db_session) -> None:
        repo = ArticleRepository(db_session, dialect_name="sqlite")
        assert repo._dialect == "sqlite"

    def test_dialect_stored_lowercase(self, db_session) -> None:
        repo = ArticleRepository(db_session, dialect_name="SQLITE")
        assert repo._dialect == "sqlite"


# ── TestUpsertArticles ────────────────────────────────────────────────────────

class TestUpsertArticles:
    def test_insert_single_article(self, repo, sample_article) -> None:
        result, url_to_id = repo.upsert_articles([sample_article])
        assert result.inserted == 1
        assert result.updated == 0
        assert "https://example.com/aapl-q2" in url_to_id

    def test_insert_returns_article_id(self, repo, db_session, sample_article) -> None:
        _, url_to_id = repo.upsert_articles([sample_article])
        article_id = url_to_id["https://example.com/aapl-q2"]
        db_article = db_session.query(NewsArticle).filter_by(id=article_id).first()
        assert db_article is not None
        assert db_article.ticker == "AAPL"

    def test_insert_batch_of_articles(self, repo, sample_articles) -> None:
        result, url_to_id = repo.upsert_articles(sample_articles)
        assert result.inserted == 5
        assert len(url_to_id) == 5

    def test_update_existing_article_on_conflict(self, repo, sample_article) -> None:
        repo.upsert_articles([sample_article])

        updated = {**sample_article, "title": "Updated headline"}
        result, _ = repo.upsert_articles([updated])

        assert result.updated == 1
        assert result.inserted == 0

    def test_updated_title_persisted(self, repo, db_session, sample_article) -> None:
        _, id_map = repo.upsert_articles([sample_article])
        article_id = id_map[sample_article["url"]]

        updated = {**sample_article, "title": "New title"}
        repo.upsert_articles([updated])

        article = db_session.query(NewsArticle).filter_by(id=article_id).first()
        assert article.title == "New title"

    def test_url_to_id_mapping_consistent_across_upserts(self, repo, sample_article) -> None:
        _, id_map1 = repo.upsert_articles([sample_article])
        _, id_map2 = repo.upsert_articles([sample_article])
        assert id_map1[sample_article["url"]] == id_map2[sample_article["url"]]

    def test_empty_records_returns_empty_result(self, repo) -> None:
        result, url_to_id = repo.upsert_articles([])
        assert result.total == 0
        assert url_to_id == {}

    def test_records_with_empty_url_are_skipped(self, repo) -> None:
        bad = {"ticker": "X", "source_id": "1", "title": "No URL", "url": ""}
        result, url_to_id = repo.upsert_articles([bad])
        assert result.skipped == 1
        assert url_to_id == {}

    def test_mixed_valid_and_invalid_urls(self, repo, sample_article) -> None:
        bad = {"ticker": "X", "source_id": "2", "title": "Bad", "url": ""}
        result, url_to_id = repo.upsert_articles([sample_article, bad])
        assert result.inserted == 1
        assert result.skipped == 1
        assert len(url_to_id) == 1

    def test_nan_values_coerced_to_none(self, repo, db_session) -> None:
        record = {
            "ticker": "NVDA", "source_id": "77", "title": "NaN test",
            "url": "https://example.com/nan-test",
            "author": float("nan"),
            "description": float("nan"),
        }
        repo.upsert_articles([record])
        article = db_session.query(NewsArticle).filter_by(url=record["url"]).first()
        assert article.author is None
        assert article.description is None


# ── TestUpsertSentimentResults ────────────────────────────────────────────────

class TestUpsertSentimentResults:
    MODEL = "ProsusAI/finbert"

    def _insert_articles_and_get_mapping(
        self,
        repo: ArticleRepository,
        articles: list[dict],
    ) -> dict[str, int]:
        _, url_to_id = repo.upsert_articles(articles)
        return url_to_id

    def test_insert_single_sentiment_row(self, repo, sample_article) -> None:
        url_to_id = self._insert_articles_and_get_mapping(repo, [sample_article])
        result = repo.upsert_sentiment_results(url_to_id, [sample_article], self.MODEL)
        assert result.inserted == 1
        assert result.updated == 0

    def test_update_existing_sentiment_on_conflict(self, repo, sample_article) -> None:
        url_to_id = self._insert_articles_and_get_mapping(repo, [sample_article])
        repo.upsert_sentiment_results(url_to_id, [sample_article], self.MODEL)

        updated = {**sample_article, "sentiment_label": "negative", "sentiment_score": -1}
        result = repo.upsert_sentiment_results(url_to_id, [updated], self.MODEL)
        assert result.updated == 1

    def test_updated_sentiment_label_persisted(
        self, repo, db_session, sample_article
    ) -> None:
        url_to_id = self._insert_articles_and_get_mapping(repo, [sample_article])
        repo.upsert_sentiment_results(url_to_id, [sample_article], self.MODEL)

        updated = {**sample_article, "sentiment_label": "negative", "sentiment_score": -1}
        repo.upsert_sentiment_results(url_to_id, [updated], self.MODEL)

        article_id = url_to_id[sample_article["url"]]
        row = (
            db_session.query(SentimentResult)
            .filter_by(article_id=article_id, model_name=self.MODEL)
            .first()
        )
        assert row.sentiment_label == "negative"
        assert row.sentiment_score == -1

    def test_different_models_create_separate_rows(
        self, repo, db_session, sample_article
    ) -> None:
        url_to_id = self._insert_articles_and_get_mapping(repo, [sample_article])
        repo.upsert_sentiment_results(url_to_id, [sample_article], "model-A")
        repo.upsert_sentiment_results(url_to_id, [sample_article], "model-B")
        count = db_session.query(SentimentResult).count()
        assert count == 2

    def test_row_without_matching_url_is_skipped(self, repo, sample_article) -> None:
        url_to_id: dict[str, int] = {}  # empty — no mapped articles
        result = repo.upsert_sentiment_results(url_to_id, [sample_article], self.MODEL)
        assert result.skipped == 1

    def test_empty_url_to_id_returns_empty_result(self, repo, sample_article) -> None:
        result = repo.upsert_sentiment_results({}, [sample_article], self.MODEL)
        assert result.total == 0
        assert result.skipped == 1

    def test_empty_records_returns_empty_result(self, repo, sample_article) -> None:
        url_to_id = self._insert_articles_and_get_mapping(repo, [sample_article])
        result = repo.upsert_sentiment_results(url_to_id, [], self.MODEL)
        assert result.total == 0

    def test_batch_insert_five_rows(self, repo, sample_articles) -> None:
        url_to_id = self._insert_articles_and_get_mapping(repo, sample_articles)
        result = repo.upsert_sentiment_results(url_to_id, sample_articles, self.MODEL)
        assert result.inserted == 5


# ── TestQueries ───────────────────────────────────────────────────────────────

class TestQueries:
    MODEL = "ProsusAI/finbert"

    @pytest.fixture(autouse=True)
    def _seed(self, repo, sample_articles) -> None:
        """Insert 5 AAPL articles + their sentiment results before each test."""
        url_to_id: dict
        _, url_to_id = repo.upsert_articles(sample_articles)
        repo.upsert_sentiment_results(url_to_id, sample_articles, self.MODEL)

    def test_count_articles_total(self, repo) -> None:
        assert repo.count_articles() == 5

    def test_count_articles_by_ticker(self, repo) -> None:
        assert repo.count_articles(ticker="AAPL") == 5
        assert repo.count_articles(ticker="TSLA") == 0

    def test_count_articles_ticker_case_insensitive(self, repo) -> None:
        assert repo.count_articles(ticker="aapl") == 5

    def test_count_sentiment_results_total(self, repo) -> None:
        assert repo.count_sentiment_results() == 5

    def test_count_sentiment_results_by_model(self, repo) -> None:
        assert repo.count_sentiment_results(model_name=self.MODEL) == 5
        assert repo.count_sentiment_results(model_name="other-model") == 0

    def test_get_articles_by_ticker_returns_correct_count(self, repo) -> None:
        articles = repo.get_articles_by_ticker("AAPL")
        assert len(articles) == 5

    def test_get_articles_by_ticker_empty_result(self, repo) -> None:
        assert repo.get_articles_by_ticker("GOOG") == []

    def test_get_articles_without_sentiment_is_empty_after_seed(self, repo) -> None:
        # All articles have sentiment → result should be empty
        result = repo.get_articles_without_sentiment(model_name=self.MODEL)
        assert result == []

    def test_get_articles_without_sentiment_finds_unscoerd_articles(
        self, repo, db_session
    ) -> None:
        # Add a new article without sentiment
        db_session.add(NewsArticle(
            ticker="TSLA", source_id="orphan", title="No sentiment",
            url="https://example.com/orphan",
        ))
        db_session.flush()
        result = repo.get_articles_without_sentiment(model_name=self.MODEL)
        assert len(result) == 1
        assert result[0].url == "https://example.com/orphan"

    def test_get_sentiment_distribution_sums_correctly(self, repo) -> None:
        distribution = repo.get_sentiment_distribution()
        total = sum(distribution.values())
        assert total == 5

    def test_get_sentiment_distribution_by_ticker(self, repo) -> None:
        dist = repo.get_sentiment_distribution(ticker="AAPL")
        assert sum(dist.values()) == 5

    def test_get_sentiment_distribution_by_model(self, repo) -> None:
        dist = repo.get_sentiment_distribution(model_name=self.MODEL)
        assert sum(dist.values()) == 5

    def test_get_sentiment_distribution_unknown_model_is_empty(self, repo) -> None:
        dist = repo.get_sentiment_distribution(model_name="ghost-model")
        assert dist == {}


# ── TestPostgresDialectPath ───────────────────────────────────────────────────

class TestPostgresDialectPath:
    """
    Verify that the PostgreSQL-specific upsert code path is invoked when
    dialect_name="postgresql".  We don't need a real PG instance; we just
    confirm the correct internal method is called.
    """

    MODEL = "ProsusAI/finbert"

    def test_pg_upsert_articles_method_called(self, db_session, sample_article) -> None:
        repo = ArticleRepository(db_session, dialect_name="postgresql")
        with patch.object(
            repo, "_pg_upsert_articles",
            wraps=repo._generic_upsert_articles,   # delegate to working impl
        ) as mock_pg:
            repo.upsert_articles([sample_article])
            mock_pg.assert_called_once()

    def test_pg_upsert_sentiment_method_called(
        self, db_session, sample_article
    ) -> None:
        # First insert the article via the generic path so we have an id
        sqlite_repo = ArticleRepository(db_session, dialect_name="sqlite")
        _, url_to_id = sqlite_repo.upsert_articles([sample_article])

        pg_repo = ArticleRepository(db_session, dialect_name="postgresql")
        with patch.object(
            pg_repo, "_pg_upsert_sentiment",
            wraps=pg_repo._generic_upsert_sentiment,
        ) as mock_pg:
            pg_repo.upsert_sentiment_results(
                url_to_id, [sample_article], self.MODEL
            )
            mock_pg.assert_called_once()

    def test_generic_path_not_called_for_postgresql_dialect(
        self, db_session, sample_article
    ) -> None:
        repo = ArticleRepository(db_session, dialect_name="postgresql")
        # Supply a valid return value so upsert_articles can unpack cleanly
        fake_pg_result = (UpsertResult(inserted=1), {sample_article["url"]: 1})
        with patch.object(repo, "_generic_upsert_articles") as mock_generic, \
             patch.object(repo, "_pg_upsert_articles", return_value=fake_pg_result):
            repo.upsert_articles([sample_article])
            mock_generic.assert_not_called()
