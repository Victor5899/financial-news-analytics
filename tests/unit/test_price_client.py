"""
Unit tests for src.prices.price_client — YFinancePriceClient.

All tests mock ``yfinance`` so no real network calls are made.

Test organisation
-----------------
TestExceptionHierarchy       — PriceIngestionError, PriceFetchError,
                               PriceValidationError inheritance
TestYFinancePriceClientInit  — constructor, required columns, defaults
TestFetchPricesValidation    — empty ticker, invalid date ranges
TestFetchPricesDateResolution — lookback_days, default end_date, explicit dates
TestFetchPricesSuccess       — returned record shape, types, adj-close logic
TestFetchPricesYfinanceErrors — yfinance exception → PriceFetchError
TestFetchPricesEmptyData     — empty DataFrame, missing columns
TestFetchMultipleTickers     — empty list, partial success, result shape
TestIsNanHelper              — edge cases for _is_nan()
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prices.price_client import (
    PriceFetchError,
    PriceIngestionError,
    PriceValidationError,
    YFinancePriceClient,
    _is_nan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_df(
    n_rows: int = 5,
    include_adj_close: bool = True,
    base_close: float = 150.0,
) -> pd.DataFrame:
    """Build a realistic yfinance-style DataFrame for testing."""
    today = date.today()
    dates = [today - timedelta(days=n_rows - 1 - i) for i in range(n_rows)]
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])

    data: dict[str, list] = {
        "Open":   [base_close - 1.0 + i * 0.5 for i in range(n_rows)],
        "High":   [base_close + 2.0 + i * 0.5 for i in range(n_rows)],
        "Low":    [base_close - 3.0 + i * 0.5 for i in range(n_rows)],
        "Close":  [base_close + 0.0 + i * 0.5 for i in range(n_rows)],
        "Volume": [1_000_000 + i * 50_000 for i in range(n_rows)],
    }
    if include_adj_close:
        data["Adj Close"] = [base_close - 0.5 + i * 0.5 for i in range(n_rows)]

    return pd.DataFrame(data, index=index)


def _make_yf_mock(df: pd.DataFrame) -> MagicMock:
    """Return a mock ``yf`` module whose Ticker().history() returns *df*."""
    mock_yf = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df
    mock_yf.Ticker.return_value = mock_ticker
    return mock_yf


# ── TestExceptionHierarchy ────────────────────────────────────────────────────

class TestExceptionHierarchy:
    def test_price_ingestion_error_is_exception(self) -> None:
        assert issubclass(PriceIngestionError, Exception)

    def test_price_fetch_error_is_price_ingestion_error(self) -> None:
        assert issubclass(PriceFetchError, PriceIngestionError)

    def test_price_validation_error_is_price_ingestion_error(self) -> None:
        assert issubclass(PriceValidationError, PriceIngestionError)

    def test_price_fetch_error_can_be_raised(self) -> None:
        with pytest.raises(PriceFetchError, match="test"):
            raise PriceFetchError("test")

    def test_price_validation_error_can_be_raised(self) -> None:
        with pytest.raises(PriceValidationError, match="invalid"):
            raise PriceValidationError("invalid")

    def test_price_fetch_error_caught_as_base(self) -> None:
        with pytest.raises(PriceIngestionError):
            raise PriceFetchError("network down")

    def test_price_validation_error_caught_as_base(self) -> None:
        with pytest.raises(PriceIngestionError):
            raise PriceValidationError("bad input")


# ── TestYFinancePriceClientInit ───────────────────────────────────────────────

class TestYFinancePriceClientInit:
    def test_default_timeout(self) -> None:
        client = YFinancePriceClient()
        assert client._timeout == 30

    def test_custom_timeout(self) -> None:
        client = YFinancePriceClient(request_timeout=60)
        assert client._timeout == 60

    def test_required_columns_defined(self) -> None:
        client = YFinancePriceClient()
        assert "Open" in client._REQUIRED_COLUMNS
        assert "High" in client._REQUIRED_COLUMNS
        assert "Low" in client._REQUIRED_COLUMNS
        assert "Close" in client._REQUIRED_COLUMNS
        assert "Volume" in client._REQUIRED_COLUMNS

    def test_adj_close_candidates_defined(self) -> None:
        client = YFinancePriceClient()
        assert "Adj Close" in client._ADJ_CLOSE_CANDIDATES


# ── TestFetchPricesValidation ─────────────────────────────────────────────────

class TestFetchPricesValidation:
    def test_empty_ticker_raises_validation_error(self) -> None:
        client = YFinancePriceClient()
        with pytest.raises(PriceValidationError, match="empty"):
            client.fetch_prices("")

    def test_whitespace_only_ticker_raises_validation_error(self) -> None:
        client = YFinancePriceClient()
        with pytest.raises(PriceValidationError, match="empty"):
            client.fetch_prices("   ")

    def test_start_equals_end_date_raises_validation_error(self) -> None:
        client = YFinancePriceClient()
        d = date(2026, 1, 15)
        with pytest.raises(PriceValidationError, match="before"):
            client.fetch_prices("AAPL", start_date=d, end_date=d)

    def test_start_after_end_date_raises_validation_error(self) -> None:
        client = YFinancePriceClient()
        with pytest.raises(PriceValidationError, match="before"):
            client.fetch_prices(
                "AAPL",
                start_date=date(2026, 6, 10),
                end_date=date(2026, 6, 1),
            )

    def test_valid_ticker_and_dates_do_not_raise_before_yf(self) -> None:
        """Validation passes; failure comes later from yfinance (mocked ok)."""
        mock_yf = _make_yf_mock(_make_mock_df())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_prices(
                "AAPL",
                start_date=date(2026, 1, 1),
                end_date=date(2026, 6, 1),
            )
        assert isinstance(result, list)


# ── TestFetchPricesDateResolution ─────────────────────────────────────────────

class TestFetchPricesDateResolution:
    def test_default_end_date_is_today(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            client.fetch_prices("AAPL", lookback_days=30)

        mock_ticker = mock_yf.Ticker.return_value
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs["end"] == date.today().isoformat()

    def test_lookback_days_applied_when_no_start_date(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            client.fetch_prices("AAPL", lookback_days=90)

        mock_ticker = mock_yf.Ticker.return_value
        call_kwargs = mock_ticker.history.call_args.kwargs
        expected_start = (date.today() - timedelta(days=90)).isoformat()
        assert call_kwargs["start"] == expected_start

    def test_explicit_start_date_overrides_lookback(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        explicit_start = date(2025, 1, 1)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            client.fetch_prices(
                "AAPL",
                start_date=explicit_start,
                lookback_days=365,
            )

        call_kwargs = mock_yf.Ticker.return_value.history.call_args.kwargs
        assert call_kwargs["start"] == explicit_start.isoformat()

    def test_explicit_end_date_passed_to_yfinance(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        explicit_end = date(2026, 3, 1)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            client.fetch_prices(
                "AAPL",
                start_date=date(2025, 1, 1),
                end_date=explicit_end,
            )

        call_kwargs = mock_yf.Ticker.return_value.history.call_args.kwargs
        assert call_kwargs["end"] == explicit_end.isoformat()

    def test_auto_adjust_false_passed_to_yfinance(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            client.fetch_prices("AAPL", lookback_days=30)

        call_kwargs = mock_yf.Ticker.return_value.history.call_args.kwargs
        assert call_kwargs.get("auto_adjust") is False

    def test_ticker_uppercased_in_yfinance_call(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            client.fetch_prices("aapl", lookback_days=30)

        mock_yf.Ticker.assert_called_with("AAPL")


# ── TestFetchPricesSuccess ────────────────────────────────────────────────────

class TestFetchPricesSuccess:
    def _fetch(self, n_rows: int = 5, **kwargs) -> list[dict]:  # type: ignore[no-untyped-def]
        df = _make_mock_df(n_rows=n_rows, **kwargs)
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            return client.fetch_prices("AAPL", lookback_days=30)

    def test_returns_list(self) -> None:
        result = self._fetch()
        assert isinstance(result, list)

    def test_correct_number_of_rows(self) -> None:
        result = self._fetch(n_rows=10)
        assert len(result) == 10

    def test_record_has_all_required_keys(self) -> None:
        result = self._fetch(n_rows=1)
        expected_keys = {
            "ticker", "trading_date",
            "open_price", "high_price", "low_price",
            "close_price", "adjusted_close", "volume",
        }
        assert expected_keys.issubset(result[0].keys())

    def test_ticker_is_uppercase(self) -> None:
        result = self._fetch()
        assert all(r["ticker"] == "AAPL" for r in result)

    def test_trading_date_is_date_object(self) -> None:
        result = self._fetch()
        for rec in result:
            assert isinstance(rec["trading_date"], date)

    def test_open_price_is_float(self) -> None:
        result = self._fetch()
        assert all(isinstance(r["open_price"], float) for r in result)

    def test_high_price_is_float(self) -> None:
        result = self._fetch()
        assert all(isinstance(r["high_price"], float) for r in result)

    def test_low_price_is_float(self) -> None:
        result = self._fetch()
        assert all(isinstance(r["low_price"], float) for r in result)

    def test_close_price_is_float(self) -> None:
        result = self._fetch()
        assert all(isinstance(r["close_price"], float) for r in result)

    def test_volume_is_int(self) -> None:
        result = self._fetch()
        assert all(isinstance(r["volume"], int) for r in result)

    def test_adjusted_close_used_when_available(self) -> None:
        df = _make_mock_df(n_rows=3, include_adj_close=True)
        # Set Adj Close clearly different from Close
        df["Close"] = 200.0
        df["Adj Close"] = 190.0
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_prices("AAPL", lookback_days=30)
        assert all(r["adjusted_close"] == pytest.approx(190.0) for r in result)

    def test_adjusted_close_falls_back_to_close_when_absent(self) -> None:
        df = _make_mock_df(n_rows=3, include_adj_close=False)
        df["Close"] = 175.0
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_prices("AAPL", lookback_days=30)
        assert all(r["adjusted_close"] == pytest.approx(175.0) for r in result)

    def test_dates_are_ordered_by_df_index(self) -> None:
        result = self._fetch(n_rows=5)
        dates = [r["trading_date"] for r in result]
        assert dates == sorted(dates)

    def test_nan_price_stored_as_none(self) -> None:
        df = _make_mock_df(n_rows=2)
        df.iloc[0, df.columns.get_loc("Open")] = float("nan")
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_prices("AAPL", lookback_days=30)
        assert result[0]["open_price"] is None

    def test_nan_volume_stored_as_none(self) -> None:
        df = _make_mock_df(n_rows=2)
        df.iloc[0, df.columns.get_loc("Volume")] = float("nan")
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_prices("AAPL", lookback_days=30)
        assert result[0]["volume"] is None


# ── TestFetchPricesYfinanceErrors ─────────────────────────────────────────────

class TestFetchPricesYfinanceErrors:
    def test_yfinance_exception_raises_price_fetch_error(self) -> None:
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("connection refused")
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            with pytest.raises(PriceFetchError, match="connection refused"):
                client.fetch_prices("AAPL", lookback_days=30)

    def test_history_exception_raises_price_fetch_error(self) -> None:
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = ConnectionError("timeout")
        mock_yf.Ticker.return_value = mock_ticker
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            with pytest.raises(PriceFetchError, match="timeout"):
                client.fetch_prices("AAPL", lookback_days=30)

    def test_price_fetch_error_wraps_original_exception(self) -> None:
        original = ValueError("yfinance crashed")
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = original
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            with pytest.raises(PriceFetchError) as exc_info:
                client.fetch_prices("AAPL", lookback_days=30)
        assert exc_info.value.__cause__ is original


# ── TestFetchPricesEmptyData ──────────────────────────────────────────────────

class TestFetchPricesEmptyData:
    def test_empty_dataframe_raises_price_validation_error(self) -> None:
        mock_yf = _make_yf_mock(pd.DataFrame())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            with pytest.raises(PriceValidationError, match="No data returned"):
                client.fetch_prices("INVALID_TICKER_XYZ", lookback_days=30)

    def test_missing_open_column_raises_price_validation_error(self) -> None:
        df = _make_mock_df()
        df = df.drop(columns=["Open"])
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            with pytest.raises(PriceValidationError, match="missing required columns"):
                client.fetch_prices("AAPL", lookback_days=30)

    def test_missing_volume_column_raises_price_validation_error(self) -> None:
        df = _make_mock_df()
        df = df.drop(columns=["Volume"])
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            with pytest.raises(PriceValidationError, match="missing required columns"):
                client.fetch_prices("AAPL", lookback_days=30)

    def test_multiindex_columns_flattened(self) -> None:
        """MultiIndex DataFrames (from yf.download) should be normalised."""
        df = _make_mock_df(n_rows=3)
        # Wrap in MultiIndex mimicking yf.download output
        df.columns = pd.MultiIndex.from_tuples(
            [(col, "AAPL") for col in df.columns]
        )
        mock_yf = _make_yf_mock(df)
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_prices("AAPL", lookback_days=30)
        assert len(result) == 3


# ── TestFetchMultipleTickers ──────────────────────────────────────────────────

class TestFetchMultipleTickers:
    def test_empty_tickers_raises_price_validation_error(self) -> None:
        client = YFinancePriceClient()
        with pytest.raises(PriceValidationError, match="empty"):
            client.fetch_multiple_tickers([])

    def test_single_ticker_success(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df(n_rows=5))
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(["AAPL"], lookback_days=30)
        assert "AAPL" in result
        assert len(result["AAPL"]) == 5

    def test_multiple_tickers_success(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df(n_rows=3))
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(
                ["AAPL", "TSLA"], lookback_days=30
            )
        assert set(result.keys()) == {"AAPL", "TSLA"}

    def test_result_keys_are_uppercase(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df(n_rows=2))
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(["aapl", "tsla"], lookback_days=30)
        assert "AAPL" in result
        assert "TSLA" in result

    def test_partial_failure_returns_successful_tickers(self) -> None:
        """One bad ticker should not prevent other tickers from being returned."""

        def ticker_side_effect(symbol: str) -> MagicMock:
            mock = MagicMock()
            if symbol == "BADSYM":
                mock.history.side_effect = RuntimeError("not found")
            else:
                mock.history.return_value = _make_mock_df(n_rows=3)
            return mock

        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = ticker_side_effect

        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(
                ["AAPL", "BADSYM", "TSLA"], lookback_days=30
            )

        assert "AAPL" in result
        assert "TSLA" in result
        assert "BADSYM" not in result

    def test_empty_string_ticker_skipped(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df(n_rows=2))
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(
                ["AAPL", "", "  "], lookback_days=30
            )
        assert "AAPL" in result
        assert "" not in result

    def test_returns_dict(self) -> None:
        mock_yf = _make_yf_mock(_make_mock_df())
        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(["AAPL"], lookback_days=30)
        assert isinstance(result, dict)

    def test_validation_error_per_ticker_logged_not_raised(self) -> None:
        """PriceValidationError for one ticker should not propagate."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # triggers PriceValidationError
        mock_yf.Ticker.return_value = mock_ticker

        with patch("src.prices.price_client.yf", mock_yf):
            client = YFinancePriceClient()
            result = client.fetch_multiple_tickers(["EMPTY_TICKER"], lookback_days=30)

        assert "EMPTY_TICKER" not in result


# ── TestIsNanHelper ───────────────────────────────────────────────────────────

class TestIsNanHelper:
    def test_float_nan_returns_true(self) -> None:
        assert _is_nan(float("nan")) is True

    def test_math_nan_returns_true(self) -> None:
        assert _is_nan(math.nan) is True

    def test_none_returns_true(self) -> None:
        # pd.isna(None) is True
        assert _is_nan(None) is True

    def test_regular_float_returns_false(self) -> None:
        assert _is_nan(3.14) is False

    def test_zero_float_returns_false(self) -> None:
        assert _is_nan(0.0) is False

    def test_integer_returns_false(self) -> None:
        assert _is_nan(42) is False

    def test_string_returns_false(self) -> None:
        assert _is_nan("hello") is False

    def test_pandas_na_returns_true(self) -> None:
        assert _is_nan(pd.NA) is True

    def test_pandas_nat_returns_true(self) -> None:
        assert _is_nan(pd.NaT) is True

    def test_pandas_nan_returns_true(self) -> None:
        assert _is_nan(float("nan")) is True
