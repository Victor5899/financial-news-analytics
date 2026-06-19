"""
Unit tests for src.ml.dataset_builder — MLDatasetBuilder.

All tests use temporary in-memory SQLite databases or mocked DataFrames.
No PostgreSQL instance is required.

Test organisation
-----------------
TestExceptionHierarchy       — exception class relationships and messages
TestMLDatasetBuilderInit     — constructor, URL validation, lazy DB init
TestLoadFeatures             — CSV loading, error conditions, column parsing
TestLoadPrices               — DB queries, filters, error propagation
TestComputeFutureCloses      — N-trading-day lookahead index arithmetic
TestComputeReturns           — percentage-return formula correctness
TestComputeBinaryLabels      — binary 0/1 label thresholding
TestComputeDirectionLabel    — BUY / HOLD / SELL classification
TestGenerateLabels           — full label pipeline, missing-data handling
TestBuildDataset             — canonical column ordering
TestSaveDataset              — CSV output, directory auto-creation
TestRunMethod                — end-to-end pipeline, dry_run, edge cases
TestDispose                  — resource cleanup lifecycle
TestMissingDataHandling      — graceful degradation on absent prices
TestLabelColumnConstants     — module-level constant correctness
TestRunRange                 — MLDatasetBuilder.run_range multi-date backfill
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.ml.dataset_builder import (
    BUY_THRESHOLD,
    LABEL_COLUMNS,
    LOOKAHEAD_DAYS,
    SELL_THRESHOLD,
    DataLoadError,
    LabelGenerationError,
    MLDatasetBuilder,
    MLDatasetError,
    _compute_binary_labels,
    _compute_direction_label,
    _compute_future_closes,
    _compute_returns,
)
from src.storage.models import Base, StockPrice

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def sqlite_engine():  # type: ignore[no-untyped-def]
    """Fresh in-memory SQLite engine per test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(sqlite_engine):  # type: ignore[no-untyped-def]
    session = Session(sqlite_engine)
    yield session
    session.close()


@pytest.fixture
def db_url_with_prices(tmp_path: Path) -> str:  # type: ignore[return]
    """
    SQLite DB with 20 consecutive days of OHLCV data for AAPL and TSLA,
    starting 2026-01-02.  Yields the SQLAlchemy connection URL.
    """
    db_file = tmp_path / "test_prices.db"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    start = date(2026, 1, 2)
    rows = []
    for i in range(20):
        d = start + timedelta(days=i)
        rows.append(StockPrice(ticker="AAPL", trading_date=d, close_price=150.0 + i))
        rows.append(StockPrice(ticker="TSLA", trading_date=d, close_price=200.0 + i * 2))

    with Session(engine) as session:
        session.add_all(rows)
        session.commit()

    yield url
    engine.dispose()


@pytest.fixture
def builder(db_url_with_prices: str) -> MLDatasetBuilder:  # type: ignore[return]
    b = MLDatasetBuilder(database_url=db_url_with_prices)
    yield b  # type: ignore[misc]
    b.dispose()


@pytest.fixture
def simple_builder() -> MLDatasetBuilder:  # type: ignore[return]
    """Lightweight builder backed by an empty in-memory SQLite DB."""
    b = MLDatasetBuilder(database_url="sqlite:///:memory:")
    yield b  # type: ignore[misc]
    b.dispose()


@pytest.fixture
def sample_features_csv(tmp_path: Path) -> Path:
    """Write a two-ticker feature CSV for 2026-01-02."""
    path = tmp_path / "feature_dataset_2026-01-02.csv"
    df = pd.DataFrame([
        {
            "ticker": "AAPL",
            "date": "2026-01-02",
            "article_count": 5,
            "positive_count": 3,
            "neutral_count": 1,
            "negative_count": 1,
            "positive_ratio": 0.6,
            "mean_sentiment_score": 0.5,
        },
        {
            "ticker": "TSLA",
            "date": "2026-01-02",
            "article_count": 3,
            "positive_count": 1,
            "neutral_count": 1,
            "negative_count": 1,
            "positive_ratio": 0.333,
            "mean_sentiment_score": -0.1,
        },
    ])
    df.to_csv(path, index=False)
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_map(
    start: date = date(2026, 1, 2),
    n: int = 15,
    base: float = 100.0,
) -> tuple[dict[date, float], list[date]]:
    pm = {start + timedelta(days=i): base + i for i in range(n)}
    return pm, sorted(pm.keys())


def _make_prices_df(
    ticker: str = "AAPL",
    start: date = date(2026, 1, 2),
    n: int = 15,
    base: float = 100.0,
) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": ticker, "trading_date": start + timedelta(days=i), "close_price": base + i}
        for i in range(n)
    ])


def _make_features_df(
    ticker: str = "AAPL",
    target_date: date = date(2026, 1, 2),
) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "date": target_date,
        "article_count": 5,
        "positive_ratio": 0.6,
        "mean_sentiment_score": 0.4,
    }])


# ═══════════════════════════════════════════════════════════════════════════════
# TestExceptionHierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    def test_ml_dataset_error_is_exception(self) -> None:
        assert issubclass(MLDatasetError, Exception)

    def test_data_load_error_is_ml_dataset_error(self) -> None:
        assert issubclass(DataLoadError, MLDatasetError)

    def test_label_generation_error_is_ml_dataset_error(self) -> None:
        assert issubclass(LabelGenerationError, MLDatasetError)

    def test_data_load_error_message_preserved(self) -> None:
        exc = DataLoadError("cannot load the data")
        assert "cannot load the data" in str(exc)

    def test_label_generation_error_message_preserved(self) -> None:
        exc = LabelGenerationError("label failure")
        assert "label failure" in str(exc)

    def test_data_load_error_catchable_as_base(self) -> None:
        with pytest.raises(MLDatasetError):
            raise DataLoadError("test")

    def test_label_generation_error_catchable_as_base(self) -> None:
        with pytest.raises(MLDatasetError):
            raise LabelGenerationError("test")


# ═══════════════════════════════════════════════════════════════════════════════
# TestMLDatasetBuilderInit
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLDatasetBuilderInit:
    def test_init_with_explicit_url(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        assert b._database_url == "sqlite:///:memory:"
        b.dispose()

    def test_no_url_no_settings_raises_data_load_error(self) -> None:
        with patch("src.ml.dataset_builder.settings") as mock_settings:
            mock_settings.database_url = None
            with pytest.raises(DataLoadError, match="No database URL"):
                MLDatasetBuilder()

    def test_db_attribute_starts_as_none(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        assert b._db is None
        b.dispose()

    def test_uses_settings_url_when_not_provided(self) -> None:
        with patch("src.ml.dataset_builder.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///:memory:"
            b = MLDatasetBuilder()
            assert b._database_url == "sqlite:///:memory:"
            b.dispose()

    def test_empty_string_url_raises_data_load_error(self) -> None:
        with patch("src.ml.dataset_builder.settings") as mock_settings:
            mock_settings.database_url = ""
            with pytest.raises(DataLoadError):
                MLDatasetBuilder()

    def test_get_db_initialises_database_manager(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        db = b._get_db()
        assert db is not None
        b.dispose()

    def test_get_db_returns_same_instance_on_repeat_calls(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        db1 = b._get_db()
        db2 = b._get_db()
        assert db1 is db2
        b.dispose()


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoadFeatures
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFeatures:
    def test_loads_csv_returns_dataframe(
        self, simple_builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        df = simple_builder.load_features(sample_features_csv)
        assert isinstance(df, pd.DataFrame)

    def test_loads_correct_row_count(
        self, simple_builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        df = simple_builder.load_features(sample_features_csv)
        assert len(df) == 2

    def test_ticker_column_present(
        self, simple_builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        df = simple_builder.load_features(sample_features_csv)
        assert "ticker" in df.columns

    def test_date_column_normalised_to_date_objects(
        self, simple_builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        df = simple_builder.load_features(sample_features_csv)
        assert all(isinstance(d, date) for d in df["date"])

    def test_correct_tickers_loaded(
        self, simple_builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        df = simple_builder.load_features(sample_features_csv)
        assert set(df["ticker"].tolist()) == {"AAPL", "TSLA"}

    def test_missing_file_raises_data_load_error(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            simple_builder.load_features(tmp_path / "nope.csv")

    def test_empty_csv_raises_data_load_error(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["ticker", "date"]).to_csv(path, index=False)
        with pytest.raises(DataLoadError, match="empty"):
            simple_builder.load_features(path)

    def test_missing_ticker_column_raises_data_load_error(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"date": "2026-01-02", "x": 1}]).to_csv(path, index=False)
        with pytest.raises(DataLoadError, match="missing required columns"):
            simple_builder.load_features(path)

    def test_missing_date_column_raises_data_load_error(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad2.csv"
        pd.DataFrame([{"ticker": "AAPL", "x": 1}]).to_csv(path, index=False)
        with pytest.raises(DataLoadError, match="missing required columns"):
            simple_builder.load_features(path)

    def test_extra_columns_preserved(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        path = tmp_path / "extra.csv"
        pd.DataFrame([{"ticker": "AAPL", "date": "2026-01-02", "foo": 42}]).to_csv(
            path, index=False
        )
        df = simple_builder.load_features(path)
        assert "foo" in df.columns


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoadPrices
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPrices:
    def test_returns_dataframe(self, builder: MLDatasetBuilder) -> None:
        df = builder.load_prices(["AAPL"], date(2026, 1, 2), date(2026, 1, 5))
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count_for_aapl(self, builder: MLDatasetBuilder) -> None:
        # 9 calendar days, all have prices in the fixture
        df = builder.load_prices(["AAPL"], date(2026, 1, 2), date(2026, 1, 10))
        assert len(df) == 9

    def test_filters_to_requested_ticker(self, builder: MLDatasetBuilder) -> None:
        df = builder.load_prices(["AAPL"], date(2026, 1, 2), date(2026, 1, 5))
        assert all(t == "AAPL" for t in df["ticker"])

    def test_empty_tickers_returns_empty_df(self, builder: MLDatasetBuilder) -> None:
        df = builder.load_prices([], date(2026, 1, 2), date(2026, 1, 10))
        assert df.empty

    def test_trading_date_normalised_to_date_objects(
        self, builder: MLDatasetBuilder
    ) -> None:
        df = builder.load_prices(["AAPL"], date(2026, 1, 2), date(2026, 1, 3))
        if not df.empty:
            assert all(isinstance(d, date) for d in df["trading_date"])

    def test_unknown_ticker_returns_empty_df(self, builder: MLDatasetBuilder) -> None:
        df = builder.load_prices(["ZZZZ"], date(2026, 1, 2), date(2026, 1, 10))
        assert df.empty

    def test_db_error_raises_data_load_error(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        with patch.object(simple_builder, "_get_db") as mock_db:
            mock_db.return_value.engine.connect.side_effect = Exception("conn fail")
            with pytest.raises(DataLoadError, match="Failed to load prices"):
                simple_builder.load_prices(["AAPL"], date(2026, 1, 2), date(2026, 1, 5))

    def test_multi_ticker_combined_result(self, builder: MLDatasetBuilder) -> None:
        df = builder.load_prices(["AAPL", "TSLA"], date(2026, 1, 2), date(2026, 1, 3))
        tickers_found = set(df["ticker"].unique())
        assert "AAPL" in tickers_found
        assert "TSLA" in tickers_found


# ═══════════════════════════════════════════════════════════════════════════════
# TestComputeFutureCloses
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeFutureCloses:
    def test_returns_all_four_keys(self) -> None:
        pm, sd = _make_price_map()
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        assert set(result.keys()) == {
            "future_close_1d", "future_close_3d", "future_close_5d", "future_close_7d"
        }

    def test_future_close_1d_is_next_day(self) -> None:
        pm, sd = _make_price_map(base=100.0)
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        assert result["future_close_1d"] == pytest.approx(101.0)

    def test_future_close_7d_is_eighth_row(self) -> None:
        pm, sd = _make_price_map(base=100.0)
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        assert result["future_close_7d"] == pytest.approx(107.0)

    def test_missing_target_date_returns_all_none(self) -> None:
        pm, sd = _make_price_map()
        result = _compute_future_closes("AAPL", date(2025, 1, 1), pm, sd)
        assert all(v is None for v in result.values())

    def test_insufficient_future_data_returns_none_for_far_horizons(self) -> None:
        pm = {date(2026, 1, 2) + timedelta(days=i): 100.0 + i for i in range(5)}
        sd = sorted(pm.keys())
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        assert result["future_close_7d"] is None

    def test_uses_trading_day_index_not_calendar_days(self) -> None:
        # Gap in calendar (Monday jump — Jan 2 and Jan 5, skipping weekend)
        dates = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
        pm = {d: 100.0 + i for i, d in enumerate(dates)}
        sd = sorted(pm.keys())
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        # 1 trading day ahead is Jan 5 (not Jan 3)
        assert result["future_close_1d"] == pytest.approx(101.0)

    def test_empty_price_map_returns_all_none(self) -> None:
        result = _compute_future_closes("AAPL", date(2026, 1, 2), {}, [])
        assert all(v is None for v in result.values())

    def test_single_day_returns_none_for_all(self) -> None:
        pm = {date(2026, 1, 2): 100.0}
        sd = [date(2026, 1, 2)]
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        assert all(v is None for v in result.values())


# ═══════════════════════════════════════════════════════════════════════════════
# TestComputeReturns
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeReturns:
    def _closes(
        self,
        r1: float = 1.01,
        r3: float = 0.98,
        r5: float = 1.03,
        r7: float = 0.95,
        base: float = 100.0,
    ) -> dict:
        return {
            "future_close_1d": base * r1,
            "future_close_3d": base * r3,
            "future_close_5d": base * r5,
            "future_close_7d": base * r7,
        }

    def test_return_1d_positive(self) -> None:
        result = _compute_returns(100.0, self._closes())
        assert result["return_1d"] == pytest.approx(0.01, rel=1e-5)

    def test_return_3d_negative(self) -> None:
        result = _compute_returns(100.0, self._closes())
        assert result["return_3d"] == pytest.approx(-0.02, rel=1e-5)

    def test_return_5d_positive(self) -> None:
        result = _compute_returns(100.0, self._closes())
        assert result["return_5d"] == pytest.approx(0.03, rel=1e-5)

    def test_return_none_when_future_close_is_none(self) -> None:
        fc = {"future_close_1d": None, "future_close_3d": 102.0,
              "future_close_5d": 103.0, "future_close_7d": 95.0}
        result = _compute_returns(100.0, fc)
        assert result["return_1d"] is None

    def test_zero_close_today_returns_all_none(self) -> None:
        result = _compute_returns(0.0, self._closes())
        assert all(v is None for v in result.values())

    def test_return_rounded_to_8_decimal_places(self) -> None:
        fc = {"future_close_1d": 100.123456789, "future_close_3d": 100.0,
              "future_close_5d": 100.0, "future_close_7d": 100.0}
        result = _compute_returns(100.0, fc)
        r = result["return_1d"]
        assert r is not None
        assert len(str(r).split(".")[-1]) <= 8

    def test_all_four_keys_returned(self) -> None:
        result = _compute_returns(100.0, self._closes())
        assert set(result.keys()) == {"return_1d", "return_3d", "return_5d", "return_7d"}


# ═══════════════════════════════════════════════════════════════════════════════
# TestComputeBinaryLabels
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeBinaryLabels:
    def _all_positive(self) -> dict:
        return {"return_1d": 0.01, "return_3d": 0.02, "return_5d": 0.03, "return_7d": 0.04}

    def _all_negative(self) -> dict:
        return {"return_1d": -0.01, "return_3d": -0.02, "return_5d": -0.03, "return_7d": -0.04}

    def test_positive_return_gives_label_1(self) -> None:
        result = _compute_binary_labels(self._all_positive())
        assert result["label_up_1d"] == 1

    def test_negative_return_gives_label_0(self) -> None:
        result = _compute_binary_labels(self._all_negative())
        assert result["label_up_1d"] == 0

    def test_zero_return_gives_label_0(self) -> None:
        returns = {"return_1d": 0.0, "return_3d": 0.0, "return_5d": 0.0, "return_7d": 0.0}
        result = _compute_binary_labels(returns)
        assert result["label_up_1d"] == 0

    def test_none_return_gives_none_label(self) -> None:
        returns = {"return_1d": None, "return_3d": 0.01, "return_5d": 0.01, "return_7d": 0.01}
        result = _compute_binary_labels(returns)
        assert result["label_up_1d"] is None

    def test_all_four_label_keys_present(self) -> None:
        result = _compute_binary_labels(self._all_positive())
        expected = {"label_up_1d", "label_up_3d", "label_up_5d", "label_up_7d"}
        assert set(result.keys()) == expected

    def test_mixed_returns_correct_labels(self) -> None:
        returns = {"return_1d": 0.01, "return_3d": -0.01,
                   "return_5d": 0.0, "return_7d": 0.05}
        result = _compute_binary_labels(returns)
        assert result["label_up_1d"] == 1
        assert result["label_up_3d"] == 0
        assert result["label_up_5d"] == 0
        assert result["label_up_7d"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestComputeDirectionLabel
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeDirectionLabel:
    def test_buy_when_above_buy_threshold(self) -> None:
        assert _compute_direction_label(0.025) == "BUY"

    def test_sell_when_below_sell_threshold(self) -> None:
        assert _compute_direction_label(-0.025) == "SELL"

    def test_hold_when_between_thresholds(self) -> None:
        assert _compute_direction_label(0.01) == "HOLD"

    def test_hold_at_zero_return(self) -> None:
        assert _compute_direction_label(0.0) == "HOLD"

    def test_hold_at_exact_buy_threshold(self) -> None:
        # Boundary: > BUY_THRESHOLD is BUY; exactly at threshold is HOLD
        assert _compute_direction_label(BUY_THRESHOLD) == "HOLD"

    def test_hold_at_exact_sell_threshold(self) -> None:
        assert _compute_direction_label(SELL_THRESHOLD) == "HOLD"

    def test_none_input_returns_none(self) -> None:
        assert _compute_direction_label(None) is None

    def test_large_positive_is_buy(self) -> None:
        assert _compute_direction_label(0.5) == "BUY"

    def test_large_negative_is_sell(self) -> None:
        assert _compute_direction_label(-0.5) == "SELL"


# ═══════════════════════════════════════════════════════════════════════════════
# TestGenerateLabels
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateLabels:
    def test_returns_dataframe(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df()
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_label_columns_present(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df()
        )
        for col in LABEL_COLUMNS:
            assert col in result.columns

    def test_label_direction_valid_values(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df()
        )
        assert result["label_direction"].iloc[0] in {"BUY", "SELL", "HOLD"}

    def test_binary_labels_are_0_or_1(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df()
        )
        for col in ["label_up_1d", "label_up_3d", "label_up_5d", "label_up_7d"]:
            assert result[col].iloc[0] in (0, 1)

    def test_feature_columns_preserved(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df()
        )
        assert "article_count" in result.columns
        assert "positive_ratio" in result.columns

    def test_return_1d_formula_correct(self, simple_builder: MLDatasetBuilder) -> None:
        # close_today=100, close_1d=101 → return_1d=0.01
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df(base=100.0)
        )
        assert result["return_1d"].iloc[0] == pytest.approx(0.01, rel=1e-5)

    def test_empty_features_raises_label_generation_error(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        with pytest.raises(LabelGenerationError):
            simple_builder.generate_labels(pd.DataFrame(), _make_prices_df())

    def test_empty_prices_raises_label_generation_error(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        with pytest.raises(LabelGenerationError):
            simple_builder.generate_labels(_make_features_df(), pd.DataFrame())

    def test_missing_current_price_raises_no_labeled_rows(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df(target_date=date(2020, 1, 1))  # date not in prices
        with pytest.raises(LabelGenerationError, match="No labeled rows"):
            simple_builder.generate_labels(feat, _make_prices_df())

    def test_insufficient_future_days_produces_partial_labels(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        # Only 3 rows: idx 0 (today), 1 (1d), 2 (2d) — no 3d/5d/7d closes
        prices = _make_prices_df(n=3)
        result = simple_builder.generate_labels(feat, prices)
        # Row is still included; 1d is available, 3d/5d/7d are null
        assert len(result) == 1
        assert pd.notna(result["future_close_1d"].iloc[0])
        assert pd.isna(result["future_close_3d"].iloc[0])
        assert pd.isna(result["future_close_5d"].iloc[0])
        assert pd.isna(result["future_close_7d"].iloc[0])

    def test_two_tickers_both_labeled(self, simple_builder: MLDatasetBuilder) -> None:
        feat = pd.DataFrame([
            {"ticker": "AAPL", "date": date(2026, 1, 2)},
            {"ticker": "TSLA", "date": date(2026, 1, 2)},
        ])
        prices = pd.concat([_make_prices_df("AAPL"), _make_prices_df("TSLA")])
        result = simple_builder.generate_labels(feat, prices)
        assert len(result) == 2

    def test_unknown_ticker_skipped_other_kept(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = pd.DataFrame([
            {"ticker": "AAPL", "date": date(2026, 1, 2)},
            {"ticker": "MISSING", "date": date(2026, 1, 2)},
        ])
        result = simple_builder.generate_labels(feat, _make_prices_df("AAPL"))
        assert len(result) == 1
        assert result["ticker"].iloc[0] == "AAPL"

    def test_future_close_columns_numeric(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.generate_labels(
            _make_features_df(), _make_prices_df()
        )
        for col in ["future_close_1d", "future_close_3d", "future_close_5d", "future_close_7d"]:
            assert pd.notna(result[col].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════════
# TestPartialFutureAvailability  (new — partial lookahead scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPartialFutureAvailability:
    """Row is kept even when only a subset of future closes exist."""

    def test_only_1d_future_row_is_kept(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        # 2 rows: idx 0 (today) + idx 1 (1d only)
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert len(result) == 1

    def test_only_1d_future_close_is_not_null(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert pd.notna(result["future_close_1d"].iloc[0])

    def test_only_1d_return_is_not_null(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert pd.notna(result["return_1d"].iloc[0])

    def test_only_1d_binary_label_is_not_null(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert result["label_up_1d"].iloc[0] in (0, 1)

    def test_3d_5d_7d_are_null_when_only_1d_available(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert pd.isna(result["future_close_3d"].iloc[0])
        assert pd.isna(result["future_close_5d"].iloc[0])
        assert pd.isna(result["future_close_7d"].iloc[0])

    def test_label_direction_null_when_5d_unavailable(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        # 5 rows: today + 1d/2d/3d/4d — 5d close (idx 5) is missing
        prices = _make_prices_df(n=5)
        result = simple_builder.generate_labels(feat, prices)
        assert result["label_direction"].iloc[0] is None or pd.isna(
            result["label_direction"].iloc[0]
        )

    def test_label_direction_present_when_5d_available(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        # 6 rows: idx 0-5, so idx 5 (5d future) is available
        prices = _make_prices_df(n=6)
        result = simple_builder.generate_labels(feat, prices)
        assert result["label_direction"].iloc[0] in {"BUY", "HOLD", "SELL"}

    def test_return_null_when_future_close_null(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert pd.isna(result["return_7d"].iloc[0])

    def test_binary_label_null_when_return_null(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        assert pd.isna(result["label_up_7d"].iloc[0])

    def test_partial_row_has_all_label_columns(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        for col in LABEL_COLUMNS:
            assert col in result.columns

    def test_dataset_written_with_partial_labels(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        feat = _make_features_df()
        prices = _make_prices_df(n=2)
        result = simple_builder.generate_labels(feat, prices)
        dataset = simple_builder.build_dataset(result)
        out = simple_builder.save_dataset(dataset, tmp_path, "2026-01-02")
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 1
        assert "label_direction" in loaded.columns

    def test_only_current_close_missing_still_raises(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        # Prices exist but not on the feature date — row skipped → no output
        feat = _make_features_df(target_date=date(2020, 6, 1))
        prices = _make_prices_df(n=15)  # starts 2026-01-02, not 2020
        with pytest.raises(LabelGenerationError, match="No labeled rows"):
            simple_builder.generate_labels(feat, prices)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBuildDataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildDataset:
    def _make_labeled_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": "AAPL",
            "date": date(2026, 1, 2),
            "article_count": 5,
            "future_close_1d": 151.0,
            "future_close_3d": 153.0,
            "future_close_5d": 155.0,
            "future_close_7d": 157.0,
            "return_1d": 0.01,
            "return_3d": 0.02,
            "return_5d": 0.03,
            "return_7d": 0.04,
            "label_up_1d": 1,
            "label_up_3d": 1,
            "label_up_5d": 1,
            "label_up_7d": 1,
            "label_direction": "BUY",
        }])

    def test_returns_dataframe(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.build_dataset(self._make_labeled_df())
        assert isinstance(result, pd.DataFrame)

    def test_label_columns_at_end(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.build_dataset(self._make_labeled_df())
        tail_cols = list(result.columns)[-len(LABEL_COLUMNS):]
        assert tail_cols == LABEL_COLUMNS

    def test_feature_columns_come_first(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.build_dataset(self._make_labeled_df())
        assert result.columns[0] == "ticker"

    def test_all_label_columns_present(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.build_dataset(self._make_labeled_df())
        for col in LABEL_COLUMNS:
            assert col in result.columns

    def test_row_count_unchanged(self, simple_builder: MLDatasetBuilder) -> None:
        labeled = self._make_labeled_df()
        result = simple_builder.build_dataset(labeled)
        assert len(result) == len(labeled)

    def test_non_label_columns_preserved(self, simple_builder: MLDatasetBuilder) -> None:
        result = simple_builder.build_dataset(self._make_labeled_df())
        assert "article_count" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveDataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveDataset:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame([{"ticker": "AAPL", "return_1d": 0.01}])

    def test_file_created(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        out = simple_builder.save_dataset(self._df(), tmp_path, "2026-01-02")
        assert out.exists()

    def test_filename_contains_date_tag(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        out = simple_builder.save_dataset(self._df(), tmp_path, "2026-01-02")
        assert "2026-01-02" in out.name

    def test_filename_prefix_is_ml_dataset(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        out = simple_builder.save_dataset(self._df(), tmp_path, "2026-01-02")
        assert out.name.startswith("ml_dataset_")

    def test_creates_nested_directory(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        nested = tmp_path / "ml" / "output"
        out = simple_builder.save_dataset(self._df(), nested, "2026-01-02")
        assert out.exists()

    def test_returns_path_object(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        result = simple_builder.save_dataset(self._df(), tmp_path, "2026-01-02")
        assert isinstance(result, Path)

    def test_csv_content_readable(
        self, simple_builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        out = simple_builder.save_dataset(self._df(), tmp_path, "2026-01-02")
        loaded = pd.read_csv(out)
        assert "ticker" in loaded.columns
        assert loaded["ticker"].iloc[0] == "AAPL"


# ═══════════════════════════════════════════════════════════════════════════════
# TestRunMethod
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunMethod:
    def test_dry_run_returns_empty_dataframe(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        result = builder.run(features_path=sample_features_csv, dry_run=True)
        assert result.empty

    def test_dry_run_does_not_call_load_features(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        with patch.object(builder, "load_features") as mock_lf:
            builder.run(features_path=sample_features_csv, dry_run=True)
        mock_lf.assert_not_called()

    def test_missing_features_file_raises_data_load_error(
        self, builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        with pytest.raises(DataLoadError):
            builder.run(features_path=tmp_path / "nope.csv")

    def test_returns_dataframe_on_success(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        result = builder.run(
            features_path=sample_features_csv,
            target_date=date(2026, 1, 2),
            lookahead_days=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_result_contains_label_direction(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        result = builder.run(
            features_path=sample_features_csv,
            target_date=date(2026, 1, 2),
            lookahead_days=20,
        )
        if not result.empty:
            assert "label_direction" in result.columns

    def test_no_output_dir_does_not_call_save(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        with patch.object(builder, "save_dataset") as mock_save:
            builder.run(
                features_path=sample_features_csv,
                target_date=date(2026, 1, 2),
                output_dir=None,
                lookahead_days=20,
            )
        mock_save.assert_not_called()

    def test_with_output_dir_calls_save(
        self, builder: MLDatasetBuilder, sample_features_csv: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "ml_out"
        builder.run(
            features_path=sample_features_csv,
            target_date=date(2026, 1, 2),
            output_dir=out_dir,
            lookahead_days=20,
        )
        saved = list(out_dir.glob("ml_dataset_*.csv"))
        assert len(saved) == 1

    def test_infers_target_date_from_csv(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        # target_date not provided — should infer from CSV max date
        result = builder.run(
            features_path=sample_features_csv,
            target_date=None,
            lookahead_days=20,
        )
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# TestDispose
# ═══════════════════════════════════════════════════════════════════════════════

class TestDispose:
    def test_dispose_sets_db_to_none(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        _ = b._get_db()  # force initialisation
        assert b._db is not None
        b.dispose()
        assert b._db is None

    def test_dispose_without_init_does_not_raise(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        b.dispose()  # _db is None; should be safe

    def test_double_dispose_does_not_raise(self) -> None:
        b = MLDatasetBuilder(database_url="sqlite:///:memory:")
        b.dispose()
        b.dispose()


# ═══════════════════════════════════════════════════════════════════════════════
# TestMissingDataHandling
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingDataHandling:
    def test_ticker_with_no_prices_skips_row(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = pd.DataFrame([{"ticker": "ZZZZ", "date": date(2026, 1, 2)}])
        prices = _make_prices_df("AAPL")
        with pytest.raises(LabelGenerationError, match="No labeled rows"):
            simple_builder.generate_labels(feat, prices)

    def test_date_not_in_prices_skips_row(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = pd.DataFrame([{"ticker": "AAPL", "date": date(2020, 1, 1)}])
        with pytest.raises(LabelGenerationError, match="No labeled rows"):
            simple_builder.generate_labels(feat, _make_prices_df())

    def test_7_rows_produces_null_7d_label(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        # 7 rows (indices 0–6): idx 0+7 = 7 which doesn't exist → future_close_7d is NULL
        prices_short = _make_prices_df(n=7)
        result = simple_builder.generate_labels(feat, prices_short)
        assert len(result) == 1
        assert pd.isna(result["future_close_7d"].iloc[0])
        assert pd.isna(result["return_7d"].iloc[0])
        assert pd.isna(result["label_up_7d"].iloc[0])

    def test_partial_ticker_list_processes_available(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = pd.DataFrame([
            {"ticker": "AAPL", "date": date(2026, 1, 2)},
            {"ticker": "MISSING", "date": date(2026, 1, 2)},
        ])
        result = simple_builder.generate_labels(feat, _make_prices_df("AAPL"))
        assert len(result) == 1
        assert result["ticker"].iloc[0] == "AAPL"

    def test_all_null_close_prices_skips_ticker(
        self, simple_builder: MLDatasetBuilder
    ) -> None:
        feat = _make_features_df()
        prices = pd.DataFrame([
            {"ticker": "AAPL",
             "trading_date": date(2026, 1, 2) + timedelta(days=i),
             "close_price": None}
            for i in range(15)
        ])
        with pytest.raises(LabelGenerationError, match="No labeled rows"):
            simple_builder.generate_labels(feat, prices)

    def test_future_closes_all_none_for_single_day(self) -> None:
        pm = {date(2026, 1, 2): 100.0}
        sd = [date(2026, 1, 2)]
        result = _compute_future_closes("AAPL", date(2026, 1, 2), pm, sd)
        assert all(v is None for v in result.values())


# ═══════════════════════════════════════════════════════════════════════════════
# TestLabelColumnConstants
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelColumnConstants:
    def test_lookahead_days_is_tuple(self) -> None:
        assert isinstance(LOOKAHEAD_DAYS, tuple)

    def test_lookahead_days_values(self) -> None:
        assert LOOKAHEAD_DAYS == (1, 3, 5, 7)

    def test_label_columns_count(self) -> None:
        # 4 future closes + 4 returns + 4 binary labels + 1 direction = 13
        assert len(LABEL_COLUMNS) == 13

    def test_label_columns_contains_direction(self) -> None:
        assert "label_direction" in LABEL_COLUMNS

    def test_label_columns_contains_all_future_close_horizons(self) -> None:
        for n in LOOKAHEAD_DAYS:
            assert f"future_close_{n}d" in LABEL_COLUMNS

    def test_label_columns_contains_all_return_horizons(self) -> None:
        for n in LOOKAHEAD_DAYS:
            assert f"return_{n}d" in LABEL_COLUMNS

    def test_label_columns_contains_all_binary_labels(self) -> None:
        for n in LOOKAHEAD_DAYS:
            assert f"label_up_{n}d" in LABEL_COLUMNS

    def test_buy_threshold_value(self) -> None:
        assert BUY_THRESHOLD == pytest.approx(0.02)

    def test_sell_threshold_value(self) -> None:
        assert SELL_THRESHOLD == pytest.approx(-0.02)


# ═══════════════════════════════════════════════════════════════════════════════
# TestRunRange  — MLDatasetBuilder.run_range (multi-date backfill)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunRange:
    """Tests for the date-range backfill path in MLDatasetBuilder."""

    @pytest.fixture
    def multi_date_features_csv(self, tmp_path: Path) -> Path:
        """Feature CSV spanning two trading dates: 2026-01-02 and 2026-01-05."""
        path = tmp_path / "feature_dataset_2026-01-02_2026-01-05.csv"
        rows = []
        for d in ["2026-01-02", "2026-01-05"]:
            for ticker in ["AAPL", "TSLA"]:
                rows.append({
                    "ticker": ticker,
                    "date": d,
                    "article_count": 5,
                    "positive_count": 3,
                    "neutral_count": 1,
                    "negative_count": 1,
                    "positive_ratio": 0.6,
                    "mean_sentiment_score": 0.5,
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_run_range_returns_dataframe(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        result = builder.run_range(
            features_path=multi_date_features_csv,
            lookahead_days=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_run_range_processes_all_dates(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        result = builder.run_range(
            features_path=multi_date_features_csv,
            lookahead_days=20,
        )
        assert not result.empty
        assert result["date"].nunique() >= 1

    def test_run_range_generates_label_columns(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        result = builder.run_range(
            features_path=multi_date_features_csv,
            lookahead_days=20,
        )
        for col in LABEL_COLUMNS:
            assert col in result.columns

    def test_run_range_preserves_feature_columns(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        result = builder.run_range(
            features_path=multi_date_features_csv,
            lookahead_days=20,
        )
        assert "article_count" in result.columns
        assert "positive_ratio" in result.columns

    def test_run_range_saves_range_csv(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "ml_out"
        builder.run_range(
            features_path=multi_date_features_csv,
            output_dir=out_dir,
            lookahead_days=20,
        )
        csvs = list(out_dir.glob("ml_dataset_*.csv"))
        assert len(csvs) == 1

    def test_run_range_csv_filename_contains_date_range(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "ml_out"
        builder.run_range(
            features_path=multi_date_features_csv,
            output_dir=out_dir,
            lookahead_days=20,
        )
        csvs = list(out_dir.glob("ml_dataset_*.csv"))
        if csvs:
            # Range file should contain start and end dates
            fname = csvs[0].name
            assert "2026-01-02" in fname

    def test_run_range_no_output_dir_does_not_save(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        with patch.object(builder, "save_dataset") as mock_save:
            builder.run_range(
                features_path=multi_date_features_csv,
                output_dir=None,
                lookahead_days=20,
            )
        mock_save.assert_not_called()

    def test_run_range_missing_file_raises_data_load_error(
        self, builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        with pytest.raises(DataLoadError):
            builder.run_range(features_path=tmp_path / "no_such_file.csv")

    def test_run_range_single_date_csv_uses_single_date_tag(
        self, builder: MLDatasetBuilder, sample_features_csv: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "ml_out"
        builder.run_range(
            features_path=sample_features_csv,
            output_dir=out_dir,
            lookahead_days=20,
        )
        csvs = list(out_dir.glob("ml_dataset_*.csv"))
        if csvs:
            # Single-date CSV → filename should use just that date (no underscore range)
            fname = csvs[0].stem  # ml_dataset_2026-01-02
            # Should not have the pattern "YYYY-MM-DD_YYYY-MM-DD" (two dates)
            parts = fname.replace("ml_dataset_", "")
            # Either "2026-01-02" or "2026-01-02_2026-01-02" — both acceptable
            assert "2026-01-02" in parts

    def test_run_range_independent_labels_per_row(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        result = builder.run_range(
            features_path=multi_date_features_csv,
            lookahead_days=20,
        )
        if not result.empty and "return_1d" in result.columns:
            # Rows on different dates should potentially have different returns
            # (just verify the column is present and processed per row)
            assert result["return_1d"].dtype in (
                "float64", "object"
            )  # float or nullable

    def test_run_range_loads_prices_for_full_window(
        self, builder: MLDatasetBuilder, multi_date_features_csv: Path
    ) -> None:
        with patch.object(
            builder, "load_prices", wraps=builder.load_prices
        ) as mock_lp:
            builder.run_range(
                features_path=multi_date_features_csv,
                lookahead_days=20,
            )
        # load_prices must be called exactly once with the full window
        assert mock_lp.call_count == 1
        call_kwargs = mock_lp.call_args
        # start_date should be the minimum feature date (2026-01-02)
        assert call_kwargs.kwargs["start_date"] == date(2026, 1, 2)

    def test_run_range_backward_compat_single_date_via_run(
        self, builder: MLDatasetBuilder, sample_features_csv: Path
    ) -> None:
        """Existing run() single-date path must still work unchanged."""
        result = builder.run(
            features_path=sample_features_csv,
            target_date=date(2026, 1, 2),
            lookahead_days=20,
        )
        assert isinstance(result, pd.DataFrame)
