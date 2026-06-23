"""
Integration tests for the GDELT ingestion pipeline.

Validates that GDELT-sourced articles flow end-to-end through:

    GDELT Client (mocked HTTP)
        → ArticleRepository (SQLite)
        → SentimentResult (SQLite)
        → FeatureEngineer
        → MLDatasetBuilder

No live GDELT API calls are made — the HTTP layer is mocked.
No PostgreSQL instance is required — all DB tests use SQLite.

Test classes
------------
  TestGDELTClientSchema       — fetch_articles returns correct schema
  TestGDELTArticlesInDB       — GDELT articles load into SQLite via ArticleRepository
  TestGDELTSentimentInDB      — Sentiment results linked to GDELT articles
  TestGDELTFeaturePipeline    — FeatureEngineer produces features from GDELT data
  TestGDELTMLPipeline         — MLDatasetBuilder produces labels from GDELT features
  TestGDELTSchemaCompatibility — GDELT DataFrame is a drop-in for Finnhub DataFrame
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.features.feature_engineer import FEATURE_COLUMNS, FeatureEngineer
from src.ingestion.gdelt_client import (
    GDELT_ARTICLE_COLUMNS,
    GDELTAPIError,
    fetch_articles,
    summarise_results,
)
from src.ml.dataset_builder import LABEL_COLUMNS, MLDatasetBuilder
from src.storage.models import Base, NewsArticle, SentimentResult, StockPrice
from src.storage.repository import ArticleRepository

UTC = timezone.utc

# ── Test constants ────────────────────────────────────────────────────────────

START_DATE = date(2025, 3, 3)   # Monday
END_DATE   = date(2025, 3, 7)   # Friday
TICKERS    = ["AAPL", "MSFT"]

_SEENDATE_FMT = "%Y%m%dT%H%M%SZ"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _gdelt_article_dict(
    ticker: str,
    target_date: date,
    idx: int,
) -> dict[str, Any]:
    """Return a raw GDELT-shaped article dict for seeding mocks."""
    return {
        "url":          f"https://reuters.com/{ticker}-{target_date}-{idx}",
        "url_mobile":   "",
        "title":        f"{ticker} financial update on {target_date} #{idx}",
        "seendate":     datetime.combine(
            target_date, time(9 + idx, 0), UTC
        ).strftime(_SEENDATE_FMT),
        "domain":       ["reuters.com", "bloomberg.com", "cnbc.com"][idx % 3],
        "language":     "English",
        "sourcecountry": "United States",
    }


def _make_gdelt_response(articles: list[dict[str, Any]]) -> MagicMock:
    import json
    mock = MagicMock(spec=requests.Response)
    mock.status_code = 200
    mock.ok = True
    mock.json.return_value = {"articles": articles}
    mock.text = json.dumps({"articles": articles})
    return mock


# ── GDELT DataFrame fixture (no DB) ──────────────────────────────────────────

@pytest.fixture(scope="module")
def gdelt_aapl_df() -> pd.DataFrame:
    """
    A GDELT DataFrame for AAPL, simulating what ``fetch_articles`` returns.

    Produced by mocking the HTTP layer so we exercise the full parsing path
    without hitting the live GDELT API.
    """
    articles = [
        _gdelt_article_dict("AAPL", START_DATE + timedelta(days=d), i)
        for d in range((END_DATE - START_DATE).days + 1)
        for i in range(3)
    ]

    with patch("src.ingestion.gdelt_client.requests.Session") as mock_session_cls:
        session_inst = MagicMock()
        mock_session_cls.return_value = session_inst
        session_inst.get.return_value = _make_gdelt_response(articles)

        from_dt = datetime.combine(START_DATE, time(0, 0), UTC)
        to_dt   = datetime.combine(END_DATE,   time(23, 59, 59), UTC)

        # Patch _build_session to return our controlled mock
        with patch("src.ingestion.gdelt_client._build_session") as mock_build:
            mock_build.return_value = session_inst
            df = fetch_articles(
                "AAPL",
                from_date=from_dt,
                to_date=to_dt,
                session=session_inst,
            )
    return df


# ── GDELT + DB fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gdelt_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    File-backed SQLite DB seeded with:
    - GDELT-sourced articles + sentiment for AAPL and MSFT across [START, END]
    - Stock prices for a window including lookahead labels

    The articles are built from GDELT-shaped data (domain-based source_name,
    hash-based source_id) to validate the full pipeline handles GDELT provenance.
    """
    tmp = tmp_path_factory.mktemp("gdelt_integration")
    db_file = tmp / "gdelt_test.db"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for day_offset in range((END_DATE - START_DATE).days + 1):
            target = START_DATE + timedelta(days=day_offset)
            for ticker in TICKERS:
                for art_idx in range(3):
                    raw = _gdelt_article_dict(ticker, target, art_idx)
                    # source_id mimics what _url_to_source_id produces
                    import hashlib
                    source_id = hashlib.sha256(
                        raw["url"].encode("utf-8")
                    ).hexdigest()[:16]

                    article = NewsArticle(
                        ticker=ticker,
                        source_id=source_id,
                        source_name=raw["domain"],
                        title=raw["title"],
                        url=raw["url"],
                        published_at=datetime.combine(target, time(9 + art_idx, 0), UTC),
                    )
                    session.add(article)
                    session.flush()

                    labels = ["positive", "neutral", "negative"]
                    scores = [1, 0, -1]
                    session.add(SentimentResult(
                        article_id=article.id,
                        model_name="ProsusAI/finbert",
                        sentiment_label=labels[art_idx % 3],
                        sentiment_score=scores[art_idx % 3],
                        sentiment_confidence=0.88 - art_idx * 0.04,
                        analysed_at=datetime.now(UTC),
                    ))

        # Stock prices: START_DATE − 7d to END_DATE + 21d
        price_start = START_DATE - timedelta(days=7)
        price_end   = END_DATE + timedelta(days=21)
        current = price_start
        price_idx = 0
        while current <= price_end:
            for ticker in TICKERS:
                base = 170.0 if ticker == "AAPL" else 380.0
                session.add(StockPrice(
                    ticker=ticker,
                    trading_date=current,
                    open_price=base + price_idx,
                    high_price=base + price_idx + 3,
                    low_price=base + price_idx - 1,
                    close_price=base + price_idx + 1,
                    volume=2_000_000 + price_idx * 500,
                ))
            current += timedelta(days=1)
            price_idx += 1

        session.commit()

    engine.dispose()
    yield url


# ── Phase 4 fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gdelt_features_df(gdelt_db_url: str) -> pd.DataFrame:
    """Run Phase 4 on the GDELT-seeded DB and return the feature DataFrame."""
    eng = FeatureEngineer(database_url=gdelt_db_url)
    df = eng.run_range(
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=TICKERS,
    )
    eng.dispose()
    return df


@pytest.fixture(scope="module")
def gdelt_features_csv(
    tmp_path_factory: pytest.TempPathFactory,
    gdelt_db_url: str,
) -> Path:
    """Run Phase 4 and save to a temp CSV; return its path."""
    out_dir = tmp_path_factory.mktemp("gdelt_features")
    eng = FeatureEngineer(database_url=gdelt_db_url)
    eng.run_range(
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=TICKERS,
        output_dir=out_dir,
    )
    eng.dispose()
    s = START_DATE.strftime("%Y-%m-%d")
    e = END_DATE.strftime("%Y-%m-%d")
    return out_dir / f"feature_dataset_{s}_{e}.csv"


# ── TestGDELTClientSchema ─────────────────────────────────────────────────────

class TestGDELTClientSchema:
    """Validate that fetch_articles output has the correct pipeline schema."""

    def test_dataframe_has_correct_columns(
        self, gdelt_aapl_df: pd.DataFrame
    ) -> None:
        assert list(gdelt_aapl_df.columns) == GDELT_ARTICLE_COLUMNS

    def test_published_at_is_utc_aware(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(gdelt_aapl_df["published_at"])
        assert gdelt_aapl_df["published_at"].dt.tz is not None

    def test_all_rows_have_aapl_ticker(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert (gdelt_aapl_df["ticker"] == "AAPL").all()

    def test_source_name_is_domain(self, gdelt_aapl_df: pd.DataFrame) -> None:
        valid_domains = {"reuters.com", "bloomberg.com", "cnbc.com"}
        for val in gdelt_aapl_df["source_name"].dropna():
            assert val in valid_domains

    def test_source_id_is_16_chars(self, gdelt_aapl_df: pd.DataFrame) -> None:
        for sid in gdelt_aapl_df["source_id"]:
            assert len(str(sid)) == 16

    def test_author_column_is_all_none(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert gdelt_aapl_df["author"].isna().all()

    def test_description_column_is_all_none(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert gdelt_aapl_df["description"].isna().all()

    def test_content_column_is_all_none(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert gdelt_aapl_df["content"].isna().all()

    def test_url_column_has_no_duplicates(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert gdelt_aapl_df["url"].nunique() == len(gdelt_aapl_df)

    def test_sorted_descending_by_published_at(
        self, gdelt_aapl_df: pd.DataFrame
    ) -> None:
        dates = gdelt_aapl_df["published_at"].tolist()
        assert dates == sorted(dates, reverse=True)

    def test_dataframe_not_empty(self, gdelt_aapl_df: pd.DataFrame) -> None:
        assert not gdelt_aapl_df.empty


# ── TestGDELTArticlesInDB ─────────────────────────────────────────────────────

class TestGDELTArticlesInDB:
    """Validate ArticleRepository handles GDELT-sourced records correctly."""

    def test_gdelt_articles_upsert_into_sqlite(
        self, gdelt_db_url: str
    ) -> None:
        engine = create_engine(
            gdelt_db_url, connect_args={"check_same_thread": False}
        )
        with Session(engine) as session:
            repo = ArticleRepository(session, dialect_name="sqlite")
            count = repo.count_articles()
        engine.dispose()
        expected = len(TICKERS) * ((END_DATE - START_DATE).days + 1) * 3
        assert count == expected

    def test_articles_have_domain_as_source_name(
        self, gdelt_db_url: str
    ) -> None:
        engine = create_engine(
            gdelt_db_url, connect_args={"check_same_thread": False}
        )
        with Session(engine) as session:
            articles = (
                session.query(NewsArticle)
                .filter(NewsArticle.ticker == "AAPL")
                .all()
            )
        engine.dispose()
        valid_domains = {"reuters.com", "bloomberg.com", "cnbc.com"}
        for art in articles:
            assert art.source_name in valid_domains

    def test_articles_have_16_char_source_id(
        self, gdelt_db_url: str
    ) -> None:
        engine = create_engine(
            gdelt_db_url, connect_args={"check_same_thread": False}
        )
        with Session(engine) as session:
            articles = session.query(NewsArticle).limit(10).all()
        engine.dispose()
        for art in articles:
            assert len(art.source_id) == 16

    def test_upsert_respects_url_uniqueness(self) -> None:
        # Use a dedicated in-memory DB so this test does not pollute the
        # module-scoped gdelt_db_url fixture shared by other test classes.
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        import hashlib
        dup_url = "https://example.com/gdelt-dedup-test"
        dup_record = {
            "ticker":       "AAPL",
            "source_id":    hashlib.sha256(dup_url.encode()).hexdigest()[:16],
            "source_name":  "example.com",
            "author":       None,
            "title":        "Test article",
            "description":  None,
            "url":          dup_url,
            "published_at": datetime(2025, 3, 3, 12, 0, tzinfo=UTC),
            "content":      None,
            "fetched_at":   datetime.now(UTC),
        }
        with Session(engine) as session:
            repo = ArticleRepository(session, dialect_name="sqlite")
            r1, _ = repo.upsert_articles([dup_record])
            r2, _ = repo.upsert_articles([dup_record])
            session.commit()
        engine.dispose()
        assert r1.inserted == 1
        assert r2.updated == 1


# ── TestGDELTSentimentInDB ────────────────────────────────────────────────────

class TestGDELTSentimentInDB:
    """Validate sentiment results are correctly linked to GDELT articles."""

    def test_all_articles_have_sentiment(
        self, gdelt_db_url: str
    ) -> None:
        engine = create_engine(
            gdelt_db_url, connect_args={"check_same_thread": False}
        )
        with Session(engine) as session:
            repo = ArticleRepository(session, dialect_name="sqlite")
            unsentimented = repo.get_articles_without_sentiment(
                model_name="ProsusAI/finbert"
            )
        engine.dispose()
        assert unsentimented == []

    def test_sentiment_distribution_has_all_labels(
        self, gdelt_db_url: str
    ) -> None:
        engine = create_engine(
            gdelt_db_url, connect_args={"check_same_thread": False}
        )
        with Session(engine) as session:
            repo = ArticleRepository(session, dialect_name="sqlite")
            dist = repo.get_sentiment_distribution(
                ticker="AAPL",
                model_name="ProsusAI/finbert",
            )
        engine.dispose()
        assert "positive" in dist
        assert "neutral" in dist
        assert "negative" in dist

    def test_sentiment_count_matches_article_count(
        self, gdelt_db_url: str
    ) -> None:
        engine = create_engine(
            gdelt_db_url, connect_args={"check_same_thread": False}
        )
        with Session(engine) as session:
            repo = ArticleRepository(session, dialect_name="sqlite")
            article_count = repo.count_articles()
            sentiment_count = repo.count_sentiment_results(
                model_name="ProsusAI/finbert"
            )
        engine.dispose()
        assert sentiment_count == article_count


# ── TestGDELTFeaturePipeline ──────────────────────────────────────────────────

class TestGDELTFeaturePipeline:
    """Validate Phase 4 feature engineering works on GDELT-seeded data."""

    def test_features_dataframe_not_empty(
        self, gdelt_features_df: pd.DataFrame
    ) -> None:
        assert not gdelt_features_df.empty

    def test_feature_columns_match_schema(
        self, gdelt_features_df: pd.DataFrame
    ) -> None:
        assert list(gdelt_features_df.columns) == FEATURE_COLUMNS

    def test_both_tickers_present(
        self, gdelt_features_df: pd.DataFrame
    ) -> None:
        assert "AAPL" in set(gdelt_features_df["ticker"])
        assert "MSFT" in set(gdelt_features_df["ticker"])

    def test_all_dates_covered(
        self, gdelt_features_df: pd.DataFrame
    ) -> None:
        dates_found = set(gdelt_features_df["date"].unique())
        expected = {
            START_DATE + timedelta(days=i)
            for i in range((END_DATE - START_DATE).days + 1)
        }
        assert dates_found == expected

    def test_article_count_reflects_gdelt_seeding(
        self, gdelt_features_df: pd.DataFrame
    ) -> None:
        for _, row in gdelt_features_df.iterrows():
            assert row["article_count"] == 3

    def test_csv_saved_with_range_filename(
        self, gdelt_features_csv: Path
    ) -> None:
        assert gdelt_features_csv.exists()
        assert START_DATE.strftime("%Y-%m-%d") in gdelt_features_csv.name
        assert END_DATE.strftime("%Y-%m-%d") in gdelt_features_csv.name

    def test_csv_row_count_matches_dataframe(
        self,
        gdelt_features_df: pd.DataFrame,
        gdelt_features_csv: Path,
    ) -> None:
        loaded = pd.read_csv(gdelt_features_csv)
        assert len(loaded) == len(gdelt_features_df)


# ── TestGDELTMLPipeline ───────────────────────────────────────────────────────

class TestGDELTMLPipeline:
    """Validate Phase 6 ML dataset builder works on GDELT-derived features."""

    @pytest.fixture(scope="class")
    def gdelt_ml_df(
        self,
        gdelt_db_url: str,
        gdelt_features_csv: Path,
    ) -> pd.DataFrame:
        builder = MLDatasetBuilder(database_url=gdelt_db_url)
        df = builder.run_range(
            features_path=gdelt_features_csv,
            lookahead_days=21,
        )
        builder.dispose()
        return df

    def test_ml_dataframe_not_empty(
        self, gdelt_ml_df: pd.DataFrame
    ) -> None:
        assert not gdelt_ml_df.empty

    def test_all_label_columns_present(
        self, gdelt_ml_df: pd.DataFrame
    ) -> None:
        for col in LABEL_COLUMNS:
            assert col in gdelt_ml_df.columns

    def test_feature_columns_preserved(
        self, gdelt_ml_df: pd.DataFrame
    ) -> None:
        assert "ticker" in gdelt_ml_df.columns
        assert "date" in gdelt_ml_df.columns
        assert "article_count" in gdelt_ml_df.columns

    def test_label_direction_valid_values(
        self, gdelt_ml_df: pd.DataFrame
    ) -> None:
        valid = {"BUY", "SELL", "HOLD"}
        for val in gdelt_ml_df["label_direction"].dropna():
            assert val in valid

    def test_binary_labels_are_zero_or_one(
        self, gdelt_ml_df: pd.DataFrame
    ) -> None:
        for col in ["label_up_1d", "label_up_3d", "label_up_5d"]:
            non_null = gdelt_ml_df[col].dropna()
            assert set(non_null.unique()).issubset({0, 1})

    def test_covers_all_feature_dates(
        self,
        gdelt_features_df: pd.DataFrame,
        gdelt_ml_df: pd.DataFrame,
    ) -> None:
        assert set(gdelt_ml_df["date"].unique()) == set(
            gdelt_features_df["date"].unique()
        )

    def test_row_count_equals_feature_row_count(
        self,
        gdelt_features_df: pd.DataFrame,
        gdelt_ml_df: pd.DataFrame,
    ) -> None:
        assert len(gdelt_ml_df) == len(gdelt_features_df)


# ── TestGDELTSchemaCompatibility ──────────────────────────────────────────────

class TestGDELTSchemaCompatibility:
    """Validate GDELT DataFrames are schema-compatible with Finnhub DataFrames."""

    def test_gdelt_columns_match_article_columns(self) -> None:
        from src.ingestion.news_client import ARTICLE_COLUMNS
        assert GDELT_ARTICLE_COLUMNS == ARTICLE_COLUMNS

    def test_empty_gdelt_df_has_same_columns_as_empty_finnhub_df(self) -> None:
        from src.ingestion.gdelt_client import _empty_dataframe as gdelt_empty  # noqa: PLC0415
        from src.ingestion.news_client import ARTICLE_COLUMNS  # noqa: PLC0415
        from src.ingestion.news_client import _empty_dataframe as finnhub_empty  # noqa: PLC0415

        assert list(gdelt_empty().columns) == list(finnhub_empty().columns)
        assert list(gdelt_empty().columns) == ARTICLE_COLUMNS

    def test_gdelt_df_passes_article_repository_upsert(
        self, gdelt_aapl_df: pd.DataFrame
    ) -> None:
        """GDELT DataFrame rows can be upserted via ArticleRepository."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        rows = gdelt_aapl_df.head(5).to_dict(orient="records")
        with Session(engine) as session:
            repo = ArticleRepository(session, dialect_name="sqlite")
            result, url_to_id = repo.upsert_articles(rows)
            session.commit()

        engine.dispose()
        assert result.inserted == 5
        assert len(url_to_id) == 5

    def test_summarise_results_compatible_with_finnhub_summarise(self) -> None:
        """Both summarise_results functions accept the same DataFrame shape."""
        from src.ingestion.news_client import summarise_results as fh_summarise

        df = pd.DataFrame({
            "ticker":       ["AAPL"],
            "published_at": pd.to_datetime(
                [datetime(2025, 3, 3, 12, 0, tzinfo=UTC)], utc=True
            ),
            "source_name":  ["reuters.com"],
        })
        # Both should produce a summary without errors
        gdelt_summary = summarise_results({"AAPL": df})
        fh_summary    = fh_summarise({"AAPL": df})
        assert list(gdelt_summary.columns) == list(fh_summary.columns)

    def test_gdelt_error_hierarchy_independent_of_finnhub(self) -> None:
        """GDELT exceptions do not inherit from Finnhub exceptions."""
        from src.ingestion.news_client import FinnhubError
        assert not issubclass(GDELTAPIError, FinnhubError)
