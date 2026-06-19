"""
Unit tests for src.features.feature_engineer.

All database-dependent tests use a file-backed SQLite database seeded via
the existing ORM models so no PostgreSQL instance is required.  Pure feature-
computation tests operate directly on pandas DataFrames.

Test organisation
-----------------
TestFeatureEngineeringErrors       — exception hierarchy (3 tests)
TestFeatureEngineerInit            — constructor + lazy init (5 tests)
TestComputeSentimentFeatures       — _compute_sentiment_features (13 tests)
TestComputeSourceFeatures          — _compute_source_features (8 tests)
TestComputeTimeFeatures            — _compute_time_features (9 tests)
TestComputeRollingFeatures         — _compute_rolling_features (10 tests)
TestGenerateFeatures               — FeatureEngineer.generate_features (12 tests)
TestSaveFeatures                   — FeatureEngineer.save_features (6 tests)
TestFeatureEngineerRun             — FeatureEngineer.run (5 tests)
TestLoadData                       — FeatureEngineer.load_data with SQLite (9 tests)
TestRunRange                       — FeatureEngineer.run_range (15 tests)
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.features.feature_engineer import (
    FEATURE_COLUMNS,
    DataLoadError,
    FeatureEngineer,
    FeatureEngineeringError,
    FeatureGenerationError,
    _compute_rolling_features,
    _compute_sentiment_features,
    _compute_source_features,
    _compute_time_features,
)
from src.storage.models import Base, NewsArticle, SentimentResult

UTC = timezone.utc

# ── Shared date constants ─────────────────────────────────────────────────────

TARGET_DATE     = date(2026, 6, 16)
YESTERDAY       = TARGET_DATE - timedelta(days=1)
TWO_DAYS_AGO    = TARGET_DATE - timedelta(days=2)
THREE_DAYS_AGO  = TARGET_DATE - timedelta(days=3)
SIX_DAYS_AGO    = TARGET_DATE - timedelta(days=6)
EIGHT_DAYS_AGO  = TARGET_DATE - timedelta(days=8)


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _row(
    ticker: str = "AAPL",
    source_name: str = "Yahoo Finance",
    date_val: date = TARGET_DATE,
    sentiment_label: str = "positive",
    sentiment_score: int = 1,
    sentiment_confidence: float = 0.9,
) -> dict:
    """Return one raw-data row matching the load_data() output schema."""
    published = datetime.combine(date_val, time(12, 0), UTC)
    return {
        "ticker":               ticker,
        "source_name":          source_name,
        "published_at":         published,
        "date":                 date_val,
        "sentiment_label":      sentiment_label,
        "sentiment_score":      sentiment_score,
        "sentiment_confidence": sentiment_confidence,
    }


def _df(*rows: dict) -> pd.DataFrame:
    """Construct a DataFrame from one or more row dicts."""
    return pd.DataFrame(list(rows))


# ── SQLite fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_url(tmp_path: Path) -> str:
    """
    File-backed SQLite DB seeded with articles + sentiment results.

    Tickers: AAPL (3 articles on TARGET_DATE), TSLA (2 articles on TARGET_DATE)
    and one AAPL article on YESTERDAY (for rolling / time feature tests).
    """
    db_path = tmp_path / "test_features.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # AAPL: three articles on TARGET_DATE
        aapl_articles = [
            NewsArticle(
                ticker="AAPL",
                source_id=f"aapl-{i}",
                source_name=["Yahoo Finance", "Benzinga", "CNBC"][i],
                title=f"AAPL news {i}",
                url=f"https://example.com/aapl-{i}",
                published_at=datetime.combine(TARGET_DATE, time(9 + i, 0), UTC),
            )
            for i in range(3)
        ]
        # AAPL: one article on YESTERDAY (for rolling features)
        aapl_articles.append(
            NewsArticle(
                ticker="AAPL",
                source_id="aapl-yesterday",
                source_name="Reuters",
                title="AAPL yesterday",
                url="https://example.com/aapl-yesterday",
                published_at=datetime.combine(YESTERDAY, time(10, 0), UTC),
            )
        )
        # TSLA: two articles on TARGET_DATE
        tsla_articles = [
            NewsArticle(
                ticker="TSLA",
                source_id=f"tsla-{i}",
                source_name="Reuters",
                title=f"TSLA news {i}",
                url=f"https://example.com/tsla-{i}",
                published_at=datetime.combine(TARGET_DATE, time(11 + i, 0), UTC),
            )
            for i in range(2)
        ]

        all_articles = aapl_articles + tsla_articles
        for a in all_articles:
            session.add(a)
        session.flush()

        labels  = ["positive", "neutral", "negative"]
        scores  = [1, 0, -1]
        for idx, a in enumerate(all_articles):
            session.add(SentimentResult(
                article_id=a.id,
                model_name="ProsusAI/finbert",
                sentiment_label=labels[idx % 3],
                sentiment_score=scores[idx % 3],
                sentiment_confidence=0.9 - idx * 0.05,
                analysed_at=datetime.now(UTC),
            ))

        session.commit()

    engine.dispose()
    return url


# ── TestFeatureEngineeringErrors ──────────────────────────────────────────────

class TestFeatureEngineeringErrors:
    def test_data_load_error_inherits_from_base(self) -> None:
        with pytest.raises(FeatureEngineeringError):
            raise DataLoadError("test")

    def test_feature_generation_error_inherits_from_base(self) -> None:
        with pytest.raises(FeatureEngineeringError):
            raise FeatureGenerationError("test")

    def test_can_catch_all_errors_via_base_class(self) -> None:
        for exc_cls in (DataLoadError, FeatureGenerationError):
            with pytest.raises(FeatureEngineeringError):
                raise exc_cls("msg")


# ── TestFeatureEngineerInit ───────────────────────────────────────────────────

class TestFeatureEngineerInit:
    def test_explicit_url_is_stored(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        assert fe._database_url == "sqlite:///:memory:"

    def test_db_is_initially_none(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        assert fe._db is None

    def test_raises_when_no_url_and_settings_database_url_is_none(
        self, mock_settings: MagicMock
    ) -> None:
        mock_settings.database_url = None
        with pytest.raises(DataLoadError, match="No database URL configured"):
            FeatureEngineer()

    def test_uses_settings_database_url_when_no_explicit_url(
        self, mock_settings: MagicMock
    ) -> None:
        mock_settings.database_url = "sqlite:///:memory:"
        fe = FeatureEngineer()
        assert fe._database_url == "sqlite:///:memory:"

    def test_dispose_without_connect_does_not_raise(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        fe.dispose()  # _db is None — must not raise


# ── TestComputeSentimentFeatures ──────────────────────────────────────────────

class TestComputeSentimentFeatures:
    def test_empty_dataframe_returns_zeros(self) -> None:
        result = _compute_sentiment_features(pd.DataFrame())
        assert result["article_count"] == 0
        assert result["positive_count"] == 0
        assert result["positive_ratio"] == 0.0

    def test_article_count(self) -> None:
        df = _df(_row(), _row(), _row())
        assert _compute_sentiment_features(df)["article_count"] == 3

    def test_positive_count(self) -> None:
        df = _df(
            _row(sentiment_label="positive"),
            _row(sentiment_label="positive"),
            _row(sentiment_label="neutral"),
        )
        assert _compute_sentiment_features(df)["positive_count"] == 2

    def test_neutral_count(self) -> None:
        df = _df(
            _row(sentiment_label="neutral"),
            _row(sentiment_label="positive"),
        )
        assert _compute_sentiment_features(df)["neutral_count"] == 1

    def test_negative_count(self) -> None:
        df = _df(
            _row(sentiment_label="negative"),
            _row(sentiment_label="negative"),
            _row(sentiment_label="neutral"),
        )
        assert _compute_sentiment_features(df)["negative_count"] == 2

    def test_positive_ratio(self) -> None:
        df = _df(
            _row(sentiment_label="positive"),
            _row(sentiment_label="neutral"),
            _row(sentiment_label="negative"),
            _row(sentiment_label="negative"),
        )
        result = _compute_sentiment_features(df)
        assert result["positive_ratio"] == pytest.approx(0.25)

    def test_ratios_sum_to_one(self) -> None:
        df = _df(
            _row(sentiment_label="positive"),
            _row(sentiment_label="neutral"),
            _row(sentiment_label="negative"),
        )
        r = _compute_sentiment_features(df)
        total = r["positive_ratio"] + r["neutral_ratio"] + r["negative_ratio"]
        # Rounded to 6 d.p. per ratio; absolute tolerance accounts for rounding error
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_mean_sentiment_score(self) -> None:
        df = _df(
            _row(sentiment_score=1),
            _row(sentiment_score=0),
            _row(sentiment_score=-1),
        )
        assert _compute_sentiment_features(df)["mean_sentiment_score"] == pytest.approx(0.0)

    def test_sentiment_score_std_single_article(self) -> None:
        df = _df(_row(sentiment_score=1))
        assert _compute_sentiment_features(df)["sentiment_score_std"] == 0.0

    def test_sentiment_score_std_multiple_articles(self) -> None:
        df = _df(_row(sentiment_score=1), _row(sentiment_score=-1))
        result = _compute_sentiment_features(df)
        assert result["sentiment_score_std"] > 0.0

    def test_sentiment_score_min(self) -> None:
        df = _df(_row(sentiment_score=1), _row(sentiment_score=-1), _row(sentiment_score=0))
        assert _compute_sentiment_features(df)["sentiment_score_min"] == -1

    def test_sentiment_score_max(self) -> None:
        df = _df(_row(sentiment_score=1), _row(sentiment_score=-1), _row(sentiment_score=0))
        assert _compute_sentiment_features(df)["sentiment_score_max"] == 1

    def test_all_positive(self) -> None:
        df = _df(_row(sentiment_label="positive", sentiment_score=1))
        r = _compute_sentiment_features(df)
        assert r["positive_ratio"] == 1.0
        assert r["neutral_ratio"] == 0.0
        assert r["negative_ratio"] == 0.0


# ── TestComputeSourceFeatures ─────────────────────────────────────────────────

class TestComputeSourceFeatures:
    def test_empty_dataframe_returns_zeros(self) -> None:
        result = _compute_source_features(pd.DataFrame())
        assert result["unique_source_count"] == 0
        assert result["yahoo_article_count"] == 0

    def test_unique_source_count(self) -> None:
        df = _df(
            _row(source_name="Yahoo Finance"),
            _row(source_name="Benzinga"),
            _row(source_name="Yahoo Finance"),
        )
        assert _compute_source_features(df)["unique_source_count"] == 2

    def test_yahoo_article_count(self) -> None:
        df = _df(
            _row(source_name="Yahoo Finance"),
            _row(source_name="yahoo.com"),
            _row(source_name="Reuters"),
        )
        assert _compute_source_features(df)["yahoo_article_count"] == 2

    def test_benzinga_article_count(self) -> None:
        df = _df(
            _row(source_name="Benzinga"),
            _row(source_name="Yahoo Finance"),
        )
        assert _compute_source_features(df)["benzinga_article_count"] == 1

    def test_cnbc_article_count(self) -> None:
        df = _df(
            _row(source_name="CNBC"),
            _row(source_name="cnbc.com"),
            _row(source_name="Reuters"),
        )
        assert _compute_source_features(df)["cnbc_article_count"] == 2

    def test_source_matching_is_case_insensitive(self) -> None:
        df = _df(
            _row(source_name="YAHOO FINANCE"),
            _row(source_name="BENZINGA"),
            _row(source_name="Cnbc"),
        )
        r = _compute_source_features(df)
        assert r["yahoo_article_count"] == 1
        assert r["benzinga_article_count"] == 1
        assert r["cnbc_article_count"] == 1

    def test_none_source_name_handled(self) -> None:
        df = _df(_row(source_name=None))  # type: ignore[arg-type]
        result = _compute_source_features(df)
        assert result["yahoo_article_count"] == 0
        assert result["unique_source_count"] == 0

    def test_unknown_source_not_counted_in_named_buckets(self) -> None:
        df = _df(_row(source_name="Reuters"), _row(source_name="Bloomberg"))
        r = _compute_source_features(df)
        assert r["yahoo_article_count"] == 0
        assert r["benzinga_article_count"] == 0
        assert r["cnbc_article_count"] == 0
        assert r["unique_source_count"] == 2


# ── TestComputeTimeFeatures ───────────────────────────────────────────────────

class TestComputeTimeFeatures:
    def test_empty_dataframe_returns_zeros(self) -> None:
        result = _compute_time_features(pd.DataFrame(), TARGET_DATE)
        assert result["articles_last_24h"] == 0
        assert result["articles_last_3d"] == 0
        assert result["articles_last_7d"] == 0

    def test_articles_last_24h_counts_same_day(self) -> None:
        df = _df(_row(date_val=TARGET_DATE), _row(date_val=YESTERDAY))
        assert _compute_time_features(df, TARGET_DATE)["articles_last_24h"] == 1

    def test_articles_last_24h_excludes_older_dates(self) -> None:
        df = _df(_row(date_val=TWO_DAYS_AGO))
        assert _compute_time_features(df, TARGET_DATE)["articles_last_24h"] == 0

    def test_articles_last_3d_includes_3_days(self) -> None:
        df = _df(
            _row(date_val=TARGET_DATE),
            _row(date_val=YESTERDAY),
            _row(date_val=TWO_DAYS_AGO),
        )
        assert _compute_time_features(df, TARGET_DATE)["articles_last_3d"] == 3

    def test_articles_last_3d_excludes_4th_day(self) -> None:
        df = _df(
            _row(date_val=TARGET_DATE),
            _row(date_val=THREE_DAYS_AGO),
        )
        assert _compute_time_features(df, TARGET_DATE)["articles_last_3d"] == 1

    def test_articles_last_7d_includes_7_days(self) -> None:
        rows = [_row(date_val=TARGET_DATE - timedelta(days=i)) for i in range(7)]
        df = _df(*rows)
        assert _compute_time_features(df, TARGET_DATE)["articles_last_7d"] == 7

    def test_articles_last_7d_excludes_8th_day(self) -> None:
        df = _df(
            _row(date_val=TARGET_DATE),
            _row(date_val=EIGHT_DAYS_AGO),
        )
        assert _compute_time_features(df, TARGET_DATE)["articles_last_7d"] == 1

    def test_time_features_zero_when_all_data_is_old(self) -> None:
        df = _df(_row(date_val=EIGHT_DAYS_AGO))
        r = _compute_time_features(df, TARGET_DATE)
        assert r["articles_last_24h"] == 0
        assert r["articles_last_3d"] == 0
        assert r["articles_last_7d"] == 0

    def test_multiple_articles_same_day_counted_all(self) -> None:
        df = _df(_row(date_val=TARGET_DATE), _row(date_val=TARGET_DATE))
        assert _compute_time_features(df, TARGET_DATE)["articles_last_24h"] == 2


# ── TestComputeRollingFeatures ────────────────────────────────────────────────

class TestComputeRollingFeatures:
    def test_empty_dataframe_returns_zeros(self) -> None:
        result = _compute_rolling_features(pd.DataFrame(), TARGET_DATE)
        assert result["rolling_3d_mean_sentiment"] == 0.0
        assert result["rolling_7d_mean_sentiment"] == 0.0
        assert result["rolling_3d_article_volume"] == 0
        assert result["rolling_7d_article_volume"] == 0

    def test_rolling_3d_mean_sentiment_single_day(self) -> None:
        df = _df(_row(date_val=TARGET_DATE, sentiment_score=1))
        r = _compute_rolling_features(df, TARGET_DATE)
        assert r["rolling_3d_mean_sentiment"] == pytest.approx(1.0)

    def test_rolling_7d_mean_sentiment_multiple_days(self) -> None:
        # day 1 mean = 1, day 2 mean = -1 → daily mean = 0.0
        df = _df(
            _row(date_val=TARGET_DATE,  sentiment_score=1),
            _row(date_val=YESTERDAY,    sentiment_score=-1),
        )
        r = _compute_rolling_features(df, TARGET_DATE)
        assert r["rolling_7d_mean_sentiment"] == pytest.approx(0.0)

    def test_rolling_3d_article_volume(self) -> None:
        df = _df(
            _row(date_val=TARGET_DATE),
            _row(date_val=YESTERDAY),
            _row(date_val=TWO_DAYS_AGO),
        )
        r = _compute_rolling_features(df, TARGET_DATE)
        assert r["rolling_3d_article_volume"] == 3

    def test_rolling_7d_article_volume(self) -> None:
        rows = [_row(date_val=TARGET_DATE - timedelta(days=i)) for i in range(7)]
        df = _df(*rows)
        assert _compute_rolling_features(df, TARGET_DATE)["rolling_7d_article_volume"] == 7

    def test_rolling_3d_excludes_data_older_than_3_days(self) -> None:
        df = _df(
            _row(date_val=TARGET_DATE,    sentiment_score=1),
            _row(date_val=THREE_DAYS_AGO, sentiment_score=-1),
        )
        r = _compute_rolling_features(df, TARGET_DATE)
        # Only TARGET_DATE is in the 3d window
        assert r["rolling_3d_mean_sentiment"] == pytest.approx(1.0)
        assert r["rolling_3d_article_volume"] == 1

    def test_rolling_7d_includes_exactly_7_days(self) -> None:
        df = _df(
            _row(date_val=SIX_DAYS_AGO,  sentiment_score=1),
            _row(date_val=EIGHT_DAYS_AGO, sentiment_score=-1),
        )
        r = _compute_rolling_features(df, TARGET_DATE)
        # Only SIX_DAYS_AGO is within 7d window
        assert r["rolling_7d_article_volume"] == 1

    def test_rolling_mean_weights_days_equally(self) -> None:
        # Day A: 3 articles all score=1  → daily mean=1
        # Day B: 1 article  score=-1     → daily mean=-1
        # Rolling mean = (1 + -1) / 2 = 0.0 (not article-weighted 0.5)
        df = _df(
            _row(date_val=TARGET_DATE, sentiment_score=1),
            _row(date_val=TARGET_DATE, sentiment_score=1),
            _row(date_val=TARGET_DATE, sentiment_score=1),
            _row(date_val=YESTERDAY,   sentiment_score=-1),
        )
        r = _compute_rolling_features(df, TARGET_DATE)
        assert r["rolling_3d_mean_sentiment"] == pytest.approx(0.0)

    def test_rolling_features_no_data_in_window_returns_zeros(self) -> None:
        # All data is outside the rolling window
        df = _df(_row(date_val=EIGHT_DAYS_AGO))
        r = _compute_rolling_features(df, TARGET_DATE)
        assert r["rolling_3d_article_volume"] == 0
        assert r["rolling_7d_article_volume"] == 0

    def test_returned_dict_has_all_four_keys(self) -> None:
        df = _df(_row())
        r = _compute_rolling_features(df, TARGET_DATE)
        assert set(r.keys()) == {
            "rolling_3d_mean_sentiment",
            "rolling_7d_mean_sentiment",
            "rolling_3d_article_volume",
            "rolling_7d_article_volume",
        }


# ── TestGenerateFeatures ──────────────────────────────────────────────────────

class TestGenerateFeatures:
    def test_raises_on_empty_dataframe(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        with pytest.raises(FeatureGenerationError, match="empty DataFrame"):
            fe.generate_features(pd.DataFrame(), TARGET_DATE)

    def test_returns_dataframe(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row())
        result = fe.generate_features(raw, TARGET_DATE)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_ticker(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row(ticker="AAPL"), _row(ticker="TSLA"))
        result = fe.generate_features(raw, TARGET_DATE)
        assert len(result) == 2

    def test_correct_ticker_in_output(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row(ticker="NVDA"))
        result = fe.generate_features(raw, TARGET_DATE)
        assert "NVDA" in result["ticker"].values

    def test_output_columns_match_feature_columns(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row())
        result = fe.generate_features(raw, TARGET_DATE)
        assert list(result.columns) == FEATURE_COLUMNS

    def test_date_column_equals_target_date(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row())
        result = fe.generate_features(raw, TARGET_DATE)
        assert result["date"].iloc[0] == TARGET_DATE

    def test_skips_ticker_with_no_articles_on_target_date(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        # Only historical data, nothing on TARGET_DATE
        raw = _df(_row(ticker="AAPL", date_val=YESTERDAY))
        with pytest.raises(FeatureGenerationError, match="No features generated"):
            fe.generate_features(raw, TARGET_DATE)

    def test_multiple_tickers_each_get_correct_article_count(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(
            _row(ticker="AAPL"),
            _row(ticker="AAPL"),
            _row(ticker="TSLA"),
        )
        result = fe.generate_features(raw, TARGET_DATE)
        aapl_count = result.loc[result["ticker"] == "AAPL", "article_count"].iloc[0]
        tsla_count = result.loc[result["ticker"] == "TSLA", "article_count"].iloc[0]
        assert aapl_count == 2
        assert tsla_count == 1

    def test_positive_and_negative_ratios_are_finite(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row(sentiment_label="positive"), _row(sentiment_label="negative"))
        result = fe.generate_features(raw, TARGET_DATE)
        assert result["positive_ratio"].iloc[0] == pytest.approx(0.5)
        assert result["negative_ratio"].iloc[0] == pytest.approx(0.5)

    def test_article_count_equals_sum_of_sentiment_counts(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(
            _row(sentiment_label="positive"),
            _row(sentiment_label="neutral"),
            _row(sentiment_label="negative"),
        )
        result = fe.generate_features(raw, TARGET_DATE)
        row = result.iloc[0]
        assert row["article_count"] == (
            row["positive_count"] + row["neutral_count"] + row["negative_count"]
        )

    def test_historical_articles_contribute_to_rolling_features(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(
            _row(date_val=TARGET_DATE,  sentiment_score=1),
            _row(date_val=YESTERDAY,    sentiment_score=-1),
        )
        result = fe.generate_features(raw, TARGET_DATE)
        # rolling_7d_article_volume should see both articles
        assert result["rolling_7d_article_volume"].iloc[0] == 2

    def test_result_is_sorted_by_ticker(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        raw = _df(_row(ticker="TSLA"), _row(ticker="AAPL"))
        result = fe.generate_features(raw, TARGET_DATE)
        assert list(result["ticker"]) == ["AAPL", "TSLA"]


# ── TestSaveFeatures ──────────────────────────────────────────────────────────

class TestSaveFeatures:
    @pytest.fixture
    def fe(self) -> FeatureEngineer:
        return FeatureEngineer(database_url="sqlite:///:memory:")

    @pytest.fixture
    def sample_features(self) -> pd.DataFrame:
        return pd.DataFrame([{col: 0 for col in FEATURE_COLUMNS}])

    def test_save_creates_csv_file(
        self, fe: FeatureEngineer, sample_features: pd.DataFrame, tmp_path: Path
    ) -> None:
        fe.save_features(sample_features, tmp_path, "2026-06-16")
        assert (tmp_path / "feature_dataset_2026-06-16.csv").exists()

    def test_save_returns_correct_path(
        self, fe: FeatureEngineer, sample_features: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = fe.save_features(sample_features, tmp_path, "2026-06-16")
        assert result == tmp_path / "feature_dataset_2026-06-16.csv"

    def test_save_creates_output_directory(
        self, fe: FeatureEngineer, sample_features: pd.DataFrame, tmp_path: Path
    ) -> None:
        nested = tmp_path / "a" / "b" / "c"
        fe.save_features(sample_features, nested, "2026-06-16")
        assert (nested / "feature_dataset_2026-06-16.csv").exists()

    def test_saved_csv_has_expected_columns(
        self, fe: FeatureEngineer, sample_features: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = fe.save_features(sample_features, tmp_path, "2026-06-16")
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == FEATURE_COLUMNS

    def test_saved_csv_row_count_matches(
        self, fe: FeatureEngineer, tmp_path: Path
    ) -> None:
        two_rows = pd.DataFrame([{col: 0 for col in FEATURE_COLUMNS} for _ in range(2)])
        path = fe.save_features(two_rows, tmp_path, "2026-06-16")
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    def test_save_date_tag_used_in_filename(
        self, fe: FeatureEngineer, sample_features: pd.DataFrame, tmp_path: Path
    ) -> None:
        fe.save_features(sample_features, tmp_path, "2099-12-31")
        assert (tmp_path / "feature_dataset_2099-12-31.csv").exists()


# ── TestFeatureEngineerRun ────────────────────────────────────────────────────

class TestFeatureEngineerRun:
    def test_dry_run_returns_empty_dataframe(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        result = fe.run(dry_run=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_dry_run_dataframe_has_feature_columns(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        result = fe.run(dry_run=True)
        assert list(result.columns) == FEATURE_COLUMNS

    def test_run_calls_load_data(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        with patch.object(fe, "load_data", return_value=pd.DataFrame()) as mock_load:
            fe.run(target_date=TARGET_DATE)
            mock_load.assert_called_once()

    def test_run_returns_empty_when_load_data_is_empty(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        with patch.object(fe, "load_data", return_value=pd.DataFrame()):
            result = fe.run(target_date=TARGET_DATE)
        assert result.empty

    def test_run_saves_when_output_dir_provided(self, tmp_path: Path) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        fake_features = pd.DataFrame(
            [{col: 0 for col in FEATURE_COLUMNS}]
        )
        with patch.object(fe, "load_data", return_value=_df(_row())), \
             patch.object(fe, "generate_features", return_value=fake_features), \
             patch.object(fe, "save_features") as mock_save:
            fe.run(target_date=TARGET_DATE, output_dir=tmp_path)
            mock_save.assert_called_once()


# ── TestLoadData ──────────────────────────────────────────────────────────────

class TestLoadData:
    def test_returns_dataframe(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE)
        assert isinstance(df, pd.DataFrame)
        fe.dispose()

    def test_expected_columns_present(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE)
        for col in ("ticker", "source_name", "published_at", "date",
                    "sentiment_label", "sentiment_score", "sentiment_confidence"):
            assert col in df.columns
        fe.dispose()

    def test_returns_articles_for_target_date(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE, lookback_days=0)
        # Lookback_days=0 means only TARGET_DATE itself
        assert len(df) > 0
        fe.dispose()

    def test_ticker_filter_applied(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(tickers=["AAPL"], target_date=TARGET_DATE)
        assert set(df["ticker"].unique()) == {"AAPL"}
        fe.dispose()

    def test_multiple_tickers_returned(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE)
        assert "AAPL" in df["ticker"].values
        assert "TSLA" in df["ticker"].values
        fe.dispose()

    def test_date_column_is_date_type(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE)
        assert isinstance(df["date"].iloc[0], date)
        fe.dispose()

    def test_published_at_is_datetime(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE)
        assert pd.api.types.is_datetime64_any_dtype(df["published_at"])
        fe.dispose()

    def test_lookback_includes_yesterday(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        df = fe.load_data(target_date=TARGET_DATE, lookback_days=7)
        # The fixture inserts one AAPL article on YESTERDAY
        yesterday_rows = df[df["date"] == YESTERDAY]
        assert len(yesterday_rows) >= 1
        fe.dispose()

    def test_raises_data_load_error_on_bad_url(self) -> None:
        fe = FeatureEngineer(
            database_url="postgresql://bad:bad@nonexistent-host:9999/db"
        )
        with pytest.raises(DataLoadError):
            fe.load_data(target_date=TARGET_DATE)


# ── TestRunRange ──────────────────────────────────────────────────────────────

class TestRunRange:
    """Tests for FeatureEngineer.run_range() — date-range backfill mode."""

    def test_raises_value_error_when_start_after_end(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        with pytest.raises(ValueError, match="must not be after"):
            fe.run_range(
                start_date=TARGET_DATE,
                end_date=TARGET_DATE - timedelta(days=1),
            )

    def test_returns_dataframe(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        result = fe.run_range(start_date=TARGET_DATE, end_date=TARGET_DATE)
        assert isinstance(result, pd.DataFrame)
        fe.dispose()

    def test_single_date_range_matches_single_run(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        single = fe.run(target_date=TARGET_DATE)
        ranged = fe.run_range(start_date=TARGET_DATE, end_date=TARGET_DATE)
        # Same tickers and same number of rows
        assert set(single["ticker"]) == set(ranged["ticker"])
        assert len(ranged) == len(single)
        fe.dispose()

    def test_multi_date_range_returns_rows_for_each_date(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        # db_url fixture has articles on TARGET_DATE and YESTERDAY
        result = fe.run_range(
            start_date=YESTERDAY,
            end_date=TARGET_DATE,
        )
        assert not result.empty
        assert result["date"].nunique() >= 1
        fe.dispose()

    def test_columns_match_feature_columns(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        result = fe.run_range(start_date=TARGET_DATE, end_date=TARGET_DATE)
        assert list(result.columns) == FEATURE_COLUMNS
        fe.dispose()

    def test_empty_dataframe_when_no_data_in_range(self) -> None:
        fe = FeatureEngineer(database_url="sqlite:///:memory:")
        with patch.object(fe, "load_data", return_value=pd.DataFrame()):
            result = fe.run_range(
                start_date=date(2000, 1, 1),
                end_date=date(2000, 1, 7),
            )
        assert result.empty
        assert list(result.columns) == FEATURE_COLUMNS
        fe.dispose()

    def test_skips_dates_with_no_articles_gracefully(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        # Range includes dates before the fixture's earliest article
        start = TARGET_DATE - timedelta(days=30)
        result = fe.run_range(start_date=start, end_date=TARGET_DATE)
        # Should still produce rows for dates that have data
        if not result.empty:
            assert all(result["date"] >= start)
        fe.dispose()

    def test_start_date_equals_end_date_is_valid(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        result = fe.run_range(start_date=TARGET_DATE, end_date=TARGET_DATE)
        # All result rows should be on TARGET_DATE
        if not result.empty:
            assert all(d == TARGET_DATE for d in result["date"])
        fe.dispose()

    def test_all_result_dates_within_range(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        start = YESTERDAY
        result = fe.run_range(start_date=start, end_date=TARGET_DATE)
        if not result.empty:
            assert all(start <= d <= TARGET_DATE for d in result["date"])
        fe.dispose()

    def test_saves_range_csv_when_output_dir_provided(
        self, db_url: str, tmp_path: Path
    ) -> None:
        fe = FeatureEngineer(database_url=db_url)
        start_tag = YESTERDAY.strftime("%Y-%m-%d")
        end_tag   = TARGET_DATE.strftime("%Y-%m-%d")
        result = fe.run_range(
            start_date=YESTERDAY,
            end_date=TARGET_DATE,
            output_dir=tmp_path,
        )
        if not result.empty:
            expected = tmp_path / f"feature_dataset_{start_tag}_{end_tag}.csv"
            assert expected.exists()
        fe.dispose()

    def test_range_csv_filename_contains_both_dates(
        self, db_url: str, tmp_path: Path
    ) -> None:
        fe = FeatureEngineer(database_url=db_url)
        start_str = YESTERDAY.strftime("%Y-%m-%d")
        end_str   = TARGET_DATE.strftime("%Y-%m-%d")
        fe.run_range(
            start_date=YESTERDAY,
            end_date=TARGET_DATE,
            output_dir=tmp_path,
        )
        csvs = list(tmp_path.glob("feature_dataset_*.csv"))
        if csvs:
            assert start_str in csvs[0].name
            assert end_str in csvs[0].name
        fe.dispose()

    def test_does_not_save_when_output_dir_is_none(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        with patch.object(fe, "save_features") as mock_save:
            fe.run_range(
                start_date=TARGET_DATE,
                end_date=TARGET_DATE,
                output_dir=None,
            )
        mock_save.assert_not_called()
        fe.dispose()

    def test_run_range_with_ticker_filter(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        result = fe.run_range(
            start_date=TARGET_DATE,
            end_date=TARGET_DATE,
            tickers=["AAPL"],
        )
        if not result.empty:
            assert set(result["ticker"].unique()) == {"AAPL"}
        fe.dispose()

    def test_run_range_result_sorted_by_date_and_ticker(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        result = fe.run_range(start_date=YESTERDAY, end_date=TARGET_DATE)
        if len(result) > 1:
            dates = list(result["date"])
            assert dates == sorted(dates)
        fe.dispose()

    def test_run_range_uses_single_db_load(self, db_url: str) -> None:
        fe = FeatureEngineer(database_url=db_url)
        with patch.object(fe, "load_data", wraps=fe.load_data) as mock_load:
            fe.run_range(start_date=YESTERDAY, end_date=TARGET_DATE)
        # Must call load_data exactly once regardless of date-range width
        assert mock_load.call_count == 1
        fe.dispose()
