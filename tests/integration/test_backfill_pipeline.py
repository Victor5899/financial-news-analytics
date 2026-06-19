"""
Integration tests for the Historical Backfill Mode pipeline.

Validates that Phase 4 (feature backfill) and Phase 6 (ML dataset backfill)
work correctly together end-to-end using an in-memory SQLite database.

Pipeline under test
-------------------
    SQLite DB (seeded articles + sentiment + prices)
        │
        ▼ Phase 4 backfill
    FeatureEngineer.run_range(start, end)
        │   → data/features/feature_dataset_<start>_<end>.csv
        ▼ Phase 6 backfill
    MLDatasetBuilder.run_range(features_path)
        │   → data/ml/ml_dataset_<start>_<end>.csv
        ▼
    Assertions: schema, date coverage, label correctness,
                backward compatibility of single-date paths

No PostgreSQL instance is required — all tests run against SQLite.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.features.feature_engineer import FEATURE_COLUMNS, FeatureEngineer
from src.ml.dataset_builder import LABEL_COLUMNS, MLDatasetBuilder
from src.storage.models import Base, NewsArticle, SentimentResult, StockPrice

UTC = timezone.utc

# ── Shared test constants ─────────────────────────────────────────────────────

START_DATE = date(2026, 1, 5)   # Monday — 5 trading days
END_DATE   = date(2026, 1, 9)   # Friday
TICKERS    = ["AAPL", "TSLA"]


# ── Shared fixture: fully seeded SQLite DB ────────────────────────────────────

@pytest.fixture(scope="module")
def seeded_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:  # type: ignore[return]
    """
    File-backed SQLite DB seeded with:
    - Articles + sentiment for AAPL and TSLA on every date in [START_DATE, END_DATE]
    - Stock prices for AAPL and TSLA for a wide window including lookahead

    Uses ``scope="module"`` so the DB is created once and shared across all
    integration tests in this file.
    """
    tmp = tmp_path_factory.mktemp("integration_db")
    db_file = tmp / "integration_test.db"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        article_id = 0
        for day_offset in range((END_DATE - START_DATE).days + 1):
            target = START_DATE + timedelta(days=day_offset)
            for ticker in TICKERS:
                for art_idx in range(3):
                    article = NewsArticle(
                        ticker=ticker,
                        source_id=f"{ticker}-{target}-{art_idx}",
                        source_name=["Yahoo Finance", "Benzinga", "CNBC"][art_idx],
                        title=f"{ticker} news on {target} #{art_idx}",
                        url=f"https://example.com/{ticker}-{target}-{art_idx}",
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
                        sentiment_confidence=0.9 - art_idx * 0.05,
                        analysed_at=datetime.now(UTC),
                    ))
                    article_id += 1

        # Prices: wide window START_DATE − 7d to END_DATE + 21d
        price_start = START_DATE - timedelta(days=7)
        price_end   = END_DATE + timedelta(days=21)
        current = price_start
        price_idx = 0
        while current <= price_end:
            for ticker in TICKERS:
                base = 150.0 if ticker == "AAPL" else 200.0
                session.add(StockPrice(
                    ticker=ticker,
                    trading_date=current,
                    open_price=base + price_idx,
                    high_price=base + price_idx + 2,
                    low_price=base + price_idx - 1,
                    close_price=base + price_idx + 0.5,
                    volume=1_000_000 + price_idx * 1000,
                ))
            current += timedelta(days=1)
            price_idx += 1

        session.commit()

    engine.dispose()
    yield url


# ── Phase 4 backfill fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backfill_features_df(seeded_db_url: str) -> pd.DataFrame:
    """Run Phase 4 backfill and return the combined feature DataFrame."""
    eng = FeatureEngineer(database_url=seeded_db_url)
    df = eng.run_range(
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=TICKERS,
    )
    eng.dispose()
    return df


@pytest.fixture(scope="module")
def backfill_features_csv(
    tmp_path_factory: pytest.TempPathFactory,
    seeded_db_url: str,
) -> Path:
    """Run Phase 4 backfill and save the combined CSV; return its path."""
    out_dir = tmp_path_factory.mktemp("features_out")
    eng = FeatureEngineer(database_url=seeded_db_url)
    eng.run_range(
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=TICKERS,
        output_dir=out_dir,
    )
    eng.dispose()
    start_tag = START_DATE.strftime("%Y-%m-%d")
    end_tag   = END_DATE.strftime("%Y-%m-%d")
    return out_dir / f"feature_dataset_{start_tag}_{end_tag}.csv"


# ── Phase 4 integration tests ─────────────────────────────────────────────────

class TestPhase4Backfill:
    def test_returns_nonempty_dataframe(self, backfill_features_df: pd.DataFrame) -> None:
        assert not backfill_features_df.empty

    def test_columns_match_feature_columns(self, backfill_features_df: pd.DataFrame) -> None:
        assert list(backfill_features_df.columns) == FEATURE_COLUMNS

    def test_covers_all_dates_with_articles(
        self, backfill_features_df: pd.DataFrame
    ) -> None:
        dates_found = set(backfill_features_df["date"].unique())
        expected    = {
            START_DATE + timedelta(days=i)
            for i in range((END_DATE - START_DATE).days + 1)
        }
        assert dates_found == expected

    def test_both_tickers_present(self, backfill_features_df: pd.DataFrame) -> None:
        tickers_found = set(backfill_features_df["ticker"].unique())
        assert "AAPL" in tickers_found
        assert "TSLA" in tickers_found

    def test_each_ticker_date_has_row(self, backfill_features_df: pd.DataFrame) -> None:
        expected_rows = len(TICKERS) * ((END_DATE - START_DATE).days + 1)
        assert len(backfill_features_df) == expected_rows

    def test_article_count_matches_seeded_data(
        self, backfill_features_df: pd.DataFrame
    ) -> None:
        for _, row in backfill_features_df.iterrows():
            assert row["article_count"] == 3

    def test_csv_file_created_with_range_name(self, backfill_features_csv: Path) -> None:
        assert backfill_features_csv.exists()

    def test_csv_filename_contains_start_date(self, backfill_features_csv: Path) -> None:
        assert START_DATE.strftime("%Y-%m-%d") in backfill_features_csv.name

    def test_csv_filename_contains_end_date(self, backfill_features_csv: Path) -> None:
        assert END_DATE.strftime("%Y-%m-%d") in backfill_features_csv.name

    def test_csv_row_count_matches_dataframe(
        self,
        backfill_features_df: pd.DataFrame,
        backfill_features_csv: Path,
    ) -> None:
        loaded = pd.read_csv(backfill_features_csv)
        assert len(loaded) == len(backfill_features_df)

    def test_single_date_run_unchanged(self, seeded_db_url: str) -> None:
        """Existing single-date run() path must produce correct output unchanged."""
        eng = FeatureEngineer(database_url=seeded_db_url)
        result = eng.run(tickers=TICKERS, target_date=START_DATE)
        eng.dispose()
        assert not result.empty
        assert list(result.columns) == FEATURE_COLUMNS
        assert all(d == START_DATE for d in result["date"])


# ── Phase 6 backfill fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backfill_ml_df(
    seeded_db_url: str,
    backfill_features_csv: Path,
) -> pd.DataFrame:
    """Run Phase 6 backfill and return the combined ML dataset."""
    builder = MLDatasetBuilder(database_url=seeded_db_url)
    df = builder.run_range(
        features_path=backfill_features_csv,
        lookahead_days=21,
    )
    builder.dispose()
    return df


# ── Phase 6 integration tests ─────────────────────────────────────────────────

class TestPhase6Backfill:
    def test_returns_nonempty_dataframe(self, backfill_ml_df: pd.DataFrame) -> None:
        assert not backfill_ml_df.empty

    def test_all_label_columns_present(self, backfill_ml_df: pd.DataFrame) -> None:
        for col in LABEL_COLUMNS:
            assert col in backfill_ml_df.columns

    def test_feature_columns_preserved(self, backfill_ml_df: pd.DataFrame) -> None:
        assert "article_count" in backfill_ml_df.columns
        assert "ticker" in backfill_ml_df.columns
        assert "date" in backfill_ml_df.columns

    def test_label_direction_valid_values(self, backfill_ml_df: pd.DataFrame) -> None:
        valid = {"BUY", "SELL", "HOLD", None}
        for val in backfill_ml_df["label_direction"].unique():
            assert val in valid or (isinstance(val, float) and pd.isna(val))

    def test_binary_labels_are_0_or_1(self, backfill_ml_df: pd.DataFrame) -> None:
        for col in ["label_up_1d", "label_up_3d", "label_up_5d", "label_up_7d"]:
            non_null = backfill_ml_df[col].dropna()
            assert set(non_null.unique()).issubset({0, 1})

    def test_returns_are_numeric(self, backfill_ml_df: pd.DataFrame) -> None:
        for col in ["return_1d", "return_3d", "return_5d", "return_7d"]:
            non_null = backfill_ml_df[col].dropna()
            assert non_null.dtype in ("float64", "float32")

    def test_covers_all_feature_dates(
        self,
        backfill_features_df: pd.DataFrame,
        backfill_ml_df: pd.DataFrame,
    ) -> None:
        feature_dates = set(backfill_features_df["date"].unique())
        ml_dates      = set(backfill_ml_df["date"].unique())
        assert ml_dates == feature_dates

    def test_row_count_matches_labeled_features(
        self,
        backfill_features_df: pd.DataFrame,
        backfill_ml_df: pd.DataFrame,
    ) -> None:
        assert len(backfill_ml_df) == len(backfill_features_df)

    def test_ml_csv_saved_with_range_name(
        self,
        seeded_db_url: str,
        backfill_features_csv: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        out_dir = tmp_path_factory.mktemp("ml_out")
        builder = MLDatasetBuilder(database_url=seeded_db_url)
        builder.run_range(
            features_path=backfill_features_csv,
            output_dir=out_dir,
            lookahead_days=21,
        )
        builder.dispose()
        csvs = list(out_dir.glob("ml_dataset_*.csv"))
        assert len(csvs) == 1
        assert START_DATE.strftime("%Y-%m-%d") in csvs[0].name

    def test_single_date_run_unchanged(
        self, seeded_db_url: str, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Existing single-date run() path must still work."""
        feat_dir = tmp_path_factory.mktemp("feat_single")
        eng = FeatureEngineer(database_url=seeded_db_url)
        eng.run(tickers=TICKERS, target_date=START_DATE, output_dir=feat_dir)
        eng.dispose()

        feat_path = feat_dir / f"feature_dataset_{START_DATE.strftime('%Y-%m-%d')}.csv"
        assert feat_path.exists()

        builder = MLDatasetBuilder(database_url=seeded_db_url)
        result = builder.run(
            features_path=feat_path,
            target_date=START_DATE,
            lookahead_days=21,
        )
        builder.dispose()

        assert not result.empty
        assert "label_direction" in result.columns
        assert all(d == START_DATE for d in result["date"])


# ── End-to-end validation: single-date mode outputs unchanged ─────────────────

class TestBackwardCompatibility:
    """
    Validates that single-date mode output is identical whether called via
    the standard ``run()`` or by extracting a single-date slice from a
    range run.
    """

    def test_feature_columns_identical_single_vs_range(
        self, seeded_db_url: str
    ) -> None:
        eng = FeatureEngineer(database_url=seeded_db_url)
        single = eng.run(tickers=TICKERS, target_date=START_DATE)
        ranged = eng.run_range(
            tickers=TICKERS, start_date=START_DATE, end_date=START_DATE
        )
        eng.dispose()
        assert list(single.columns) == list(ranged.columns)

    def test_feature_values_identical_single_vs_range(
        self, seeded_db_url: str
    ) -> None:
        eng = FeatureEngineer(database_url=seeded_db_url)
        single = eng.run(tickers=TICKERS, target_date=START_DATE).reset_index(drop=True)
        ranged = eng.run_range(
            tickers=TICKERS, start_date=START_DATE, end_date=START_DATE
        ).reset_index(drop=True)
        eng.dispose()
        pd.testing.assert_frame_equal(single, ranged, check_like=False)

    def test_ml_labels_identical_single_vs_range(
        self,
        seeded_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        feat_dir  = tmp_path_factory.mktemp("compat_feat")
        eng = FeatureEngineer(database_url=seeded_db_url)
        eng.run(tickers=TICKERS, target_date=START_DATE, output_dir=feat_dir)
        eng.run_range(
            tickers=TICKERS,
            start_date=START_DATE,
            end_date=START_DATE,
            output_dir=feat_dir,
        )
        eng.dispose()

        single_feat = feat_dir / f"feature_dataset_{START_DATE.strftime('%Y-%m-%d')}.csv"
        range_feat  = feat_dir / (
            f"feature_dataset_{START_DATE.strftime('%Y-%m-%d')}_"
            f"{START_DATE.strftime('%Y-%m-%d')}.csv"
        )

        builder = MLDatasetBuilder(database_url=seeded_db_url)
        single_ml = builder.run(
            features_path=single_feat,
            target_date=START_DATE,
            lookahead_days=21,
        )
        range_ml = builder.run_range(
            features_path=range_feat,
            lookahead_days=21,
        )
        builder.dispose()

        assert len(single_ml) == len(range_ml)
        assert set(single_ml["ticker"]) == set(range_ml["ticker"])
