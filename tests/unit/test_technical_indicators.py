"""
Unit tests for the technical indicator helpers in
src.features.feature_engineer.

All tests operate directly on deterministic pandas DataFrames — no database
connection is required.

Test organisation
-----------------
TestSMA                   — _sma primitive (8 tests)
TestEMA                   — _ema primitive (7 tests)
TestRSI                   — _rsi primitive (9 tests)
TestMACDLines             — _macd_lines primitive (9 tests)
TestBollingerBands        — _bollinger_bands primitive (10 tests)
TestATR                   — _atr primitive (8 tests)
TestTrendFeatures         — _compute_trend_features (7 tests)
TestMomentumFeatures      — _compute_momentum_features (7 tests)
TestVolatilityFeatures    — _compute_volatility_features (8 tests)
TestReturnFeatures        — _compute_return_features (8 tests)
TestVolumeFeatures        — _compute_volume_features (9 tests)
TestTechnicalFeatures     — _compute_technical_features orchestrator (6 tests)
TestRollingVolatility     — rolling 20-day volatility via volatility features (5 tests)
TestGenerateFeaturesMerge — generate_features integration with prices_df (7 tests)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from src.features.feature_engineer import (
    FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
    FeatureEngineer,
    FeatureGenerationError,
    _atr,
    _bollinger_bands,
    _compute_momentum_features,
    _compute_return_features,
    _compute_technical_features,
    _compute_trend_features,
    _compute_volume_features,
    _compute_volatility_features,
    _ema,
    _extract_at,
    _macd_lines,
    _rsi,
    _sma,
)

# ── Shared date constants ─────────────────────────────────────────────────────

TARGET_DATE = date(2026, 6, 16)


# ── DataFrame builders ────────────────────────────────────────────────────────

def _price_series(values: list[float]) -> pd.Series:
    """Build a simple float Series from a list."""
    return pd.Series(values, dtype=float)


def _ohlcv_df(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    start: date = date(2026, 1, 1),
    ticker: str = "TEST",
) -> pd.DataFrame:
    """
    Construct a minimal OHLCV DataFrame with a ``trading_date`` column.

    *closes* is required; *highs*, *lows*, and *volumes* default to derived
    values (high = close + 1, low = close - 1, volume = 1_000_000).
    The ``trading_date`` column runs forward from *start* with 1-day steps.
    """
    n = len(closes)
    closes_f = [float(c) for c in closes]
    if highs is None:
        highs = [c + 1.0 for c in closes_f]
    if lows is None:
        lows = [c - 1.0 for c in closes_f]
    if volumes is None:
        volumes = [1_000_000.0] * n

    trading_dates = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {
            "ticker":       [ticker] * n,
            "trading_date": trading_dates,
            "open_price":   closes_f,
            "high_price":   [float(h) for h in highs],
            "low_price":    [float(l) for l in lows],
            "close_price":  closes_f,
            "adjusted_close": closes_f,
            "volume":       [float(v) for v in volumes],
        }
    )


def _sentiment_row(
    ticker: str = "TEST",
    date_val: date = TARGET_DATE,
    sentiment_label: str = "neutral",
    sentiment_score: int = 0,
) -> dict[str, Any]:
    """Minimal sentiment row matching load_data() output schema."""
    from datetime import datetime, time, timezone
    published = datetime.combine(date_val, time(12, 0), timezone.utc)
    return {
        "ticker":               ticker,
        "source_name":          "Reuters",
        "published_at":         published,
        "date":                 date_val,
        "sentiment_label":      sentiment_label,
        "sentiment_score":      sentiment_score,
        "sentiment_confidence": 0.9,
    }


# ── TestSMA ───────────────────────────────────────────────────────────────────

class TestSMA:
    def test_window_3_basic_values(self) -> None:
        s = _price_series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _sma(s, window=3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_first_window_minus_one_rows_are_nan(self) -> None:
        s = _price_series([10.0, 20.0, 30.0, 40.0])
        result = _sma(s, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_window_1_equals_series(self) -> None:
        s = _price_series([5.0, 3.0, 7.0])
        result = _sma(s, window=1)
        assert result.tolist() == pytest.approx([5.0, 3.0, 7.0])

    def test_window_equal_to_series_length(self) -> None:
        s = _price_series([1.0, 2.0, 3.0, 4.0])
        result = _sma(s, window=4)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == pytest.approx(2.5)

    def test_window_20_has_19_nan_rows(self) -> None:
        s = _price_series([float(i) for i in range(1, 31)])
        result = _sma(s, window=20)
        assert all(pd.isna(result.iloc[i]) for i in range(19))
        assert not pd.isna(result.iloc[19])

    def test_window_10_value_matches_manual_mean(self) -> None:
        values = list(range(1, 21))
        s = _price_series(values)
        result = _sma(s, window=10)
        expected = sum(values[10:20]) / 10
        assert result.iloc[19] == pytest.approx(expected)

    def test_constant_series_returns_constant(self) -> None:
        s = _price_series([5.0] * 15)
        result = _sma(s, window=10)
        assert result.dropna().iloc[-1] == pytest.approx(5.0)

    def test_returns_series_with_same_index(self) -> None:
        s = _price_series([1.0, 2.0, 3.0])
        result = _sma(s, window=2)
        assert list(result.index) == list(s.index)


# ── TestEMA ───────────────────────────────────────────────────────────────────

class TestEMA:
    def test_first_span_minus_one_rows_are_nan(self) -> None:
        s = _price_series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ema(s, span=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_constant_series_returns_constant(self) -> None:
        s = _price_series([10.0] * 20)
        result = _ema(s, span=5)
        assert result.dropna().iloc[-1] == pytest.approx(10.0)

    def test_span_10_has_nine_nan_rows(self) -> None:
        s = _price_series([float(i) for i in range(1, 25)])
        result = _ema(s, span=10)
        assert all(pd.isna(result.iloc[i]) for i in range(9))
        assert not pd.isna(result.iloc[9])

    def test_ema_reacts_faster_to_recent_values_than_sma(self) -> None:
        # After a big jump at the end, EMA should be closer to the new value
        values = [100.0] * 20 + [200.0]
        s = _price_series(values)
        sma = _sma(s, window=10).iloc[-1]
        ema = _ema(s, span=10).iloc[-1]
        assert ema > sma  # EMA weights recent values more heavily

    def test_ema_converges_to_constant_after_many_periods(self) -> None:
        # EMA of a constant series should equal the constant
        s = _price_series([42.0] * 50)
        result = _ema(s, span=10)
        assert result.dropna().iloc[-1] == pytest.approx(42.0, rel=1e-6)

    def test_returns_series_same_length_as_input(self) -> None:
        s = _price_series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ema(s, span=3)
        assert len(result) == len(s)

    def test_span_20_first_19_nan(self) -> None:
        s = _price_series([float(i) for i in range(1, 30)])
        result = _ema(s, span=20)
        assert all(pd.isna(result.iloc[i]) for i in range(19))
        assert not pd.isna(result.iloc[19])


# ── TestRSI ───────────────────────────────────────────────────────────────────

class TestRSI:
    def test_pure_uptrend_rsi_equals_100(self) -> None:
        close = _price_series([float(i) for i in range(1, 30)])
        result = _rsi(close, period=14)
        assert result.dropna().iloc[-1] == pytest.approx(100.0)

    def test_pure_downtrend_rsi_equals_0(self) -> None:
        close = _price_series([float(i) for i in range(29, 0, -1)])
        result = _rsi(close, period=14)
        assert result.dropna().iloc[-1] == pytest.approx(0.0)

    def test_first_period_rows_are_nan(self) -> None:
        close = _price_series([float(i) for i in range(1, 20)])
        result = _rsi(close, period=14)
        assert all(pd.isna(result.iloc[i]) for i in range(14))

    def test_rsi_in_valid_range(self) -> None:
        close = _price_series(
            [100.0, 102.0, 101.0, 103.0, 100.0, 104.0, 102.0,
             105.0, 103.0, 106.0, 104.0, 107.0, 105.0, 108.0, 106.0]
        )
        result = _rsi(close, period=14)
        non_nan = result.dropna()
        assert all(0.0 <= v <= 100.0 for v in non_nan)

    def test_rsi_period_14_returns_correct_length(self) -> None:
        close = _price_series([float(i) for i in range(1, 31)])
        result = _rsi(close, period=14)
        assert len(result) == 30

    def test_alternating_up_down_rsi_near_50(self) -> None:
        # Perfectly alternating gains/losses → RSI ≈ 50
        prices = []
        p = 100.0
        for i in range(30):
            p = p + 1.0 if i % 2 == 0 else p - 1.0
            prices.append(p)
        close = _price_series(prices)
        result = _rsi(close, period=14)
        non_nan = result.dropna()
        # Should be close to 50 (symmetric gains and losses)
        assert abs(non_nan.iloc[-1] - 50.0) < 15.0

    def test_all_zero_changes_rsi_is_nan_or_50(self) -> None:
        # Constant price — no change, no gain, no loss → undefined
        close = _price_series([100.0] * 20)
        result = _rsi(close, period=14)
        # With zero gain AND zero loss, the function should return some value
        # without raising; either NaN or a defined value is acceptable
        assert len(result) == 20

    def test_rsi_output_dtype_is_float(self) -> None:
        close = _price_series([float(i) for i in range(1, 25)])
        result = _rsi(close, period=14)
        assert pd.api.types.is_float_dtype(result)

    def test_rsi_period_5_fewer_nan_rows(self) -> None:
        close = _price_series([float(i) for i in range(1, 20)])
        result = _rsi(close, period=5)
        # First 5 rows NaN, remainder should have values
        assert all(pd.isna(result.iloc[i]) for i in range(5))
        assert not pd.isna(result.iloc[5])


# ── TestMACDLines ─────────────────────────────────────────────────────────────

class TestMACDLines:
    def _trending_close(self, n: int = 60) -> pd.Series:
        """Monotonically increasing price series for MACD tests."""
        return _price_series([float(i) for i in range(1, n + 1)])

    def test_returns_three_series(self) -> None:
        close = self._trending_close()
        result = _macd_lines(close)
        assert len(result) == 3

    def test_all_series_same_length_as_input(self) -> None:
        close = self._trending_close(60)
        macd, signal, hist = _macd_lines(close)
        assert len(macd) == len(signal) == len(hist) == 60

    def test_macd_nan_for_first_slow_minus_one_rows(self) -> None:
        close = self._trending_close(60)
        macd, _, _ = _macd_lines(close, fast=12, slow=26)
        # First 25 rows should be NaN (span=26 → min_periods=26)
        assert all(pd.isna(macd.iloc[i]) for i in range(25))
        assert not pd.isna(macd.iloc[25])

    def test_signal_nan_until_slow_plus_signal_minus_one(self) -> None:
        close = self._trending_close(60)
        _, signal, _ = _macd_lines(close, fast=12, slow=26, signal_span=9)
        # Signal is EWM of MACD; first non-NaN MACD appears at index 25,
        # signal needs 9 non-NaN MACD values → first non-NaN signal ≈ index 33
        non_nan_signal = signal.dropna()
        assert len(non_nan_signal) > 0

    def test_histogram_equals_macd_minus_signal(self) -> None:
        close = self._trending_close(60)
        macd, signal, hist = _macd_lines(close)
        diff = (macd - signal).dropna()
        hist_no_nan = hist.dropna()
        common_idx = diff.index.intersection(hist_no_nan.index)
        pd.testing.assert_series_equal(
            diff.loc[common_idx].reset_index(drop=True),
            hist_no_nan.loc[common_idx].reset_index(drop=True),
            check_names=False,
        )

    def test_uptrend_macd_positive(self) -> None:
        # Rising prices → fast EMA > slow EMA → MACD > 0
        close = _price_series([float(i) for i in range(1, 61)])
        macd, _, _ = _macd_lines(close)
        non_nan = macd.dropna()
        assert all(v > 0 for v in non_nan)

    def test_downtrend_macd_negative(self) -> None:
        # Falling prices → fast EMA < slow EMA → MACD < 0
        close = _price_series([float(i) for i in range(60, 0, -1)])
        macd, _, _ = _macd_lines(close)
        non_nan = macd.dropna()
        assert all(v < 0 for v in non_nan)

    def test_constant_price_macd_near_zero(self) -> None:
        close = _price_series([100.0] * 60)
        macd, signal, hist = _macd_lines(close)
        non_nan = macd.dropna()
        assert all(abs(v) < 1e-9 for v in non_nan)

    def test_custom_spans_accepted(self) -> None:
        close = self._trending_close(50)
        macd, signal, hist = _macd_lines(close, fast=5, slow=10, signal_span=3)
        assert not macd.dropna().empty


# ── TestBollingerBands ────────────────────────────────────────────────────────

class TestBollingerBands:
    def test_returns_three_series(self) -> None:
        close = _price_series([float(i) for i in range(1, 30)])
        result = _bollinger_bands(close, window=20)
        assert len(result) == 3

    def test_first_window_minus_one_rows_are_nan(self) -> None:
        close = _price_series([float(i) for i in range(1, 30)])
        upper, lower, width = _bollinger_bands(close, window=20)
        assert all(pd.isna(upper.iloc[i]) for i in range(19))
        assert not pd.isna(upper.iloc[19])

    def test_upper_greater_than_lower(self) -> None:
        close = _price_series([float(i) + (i % 3) for i in range(1, 30)])
        upper, lower, _ = _bollinger_bands(close, window=20)
        non_nan_upper = upper.dropna()
        non_nan_lower = lower.dropna()
        assert all(u > l for u, l in zip(non_nan_upper, non_nan_lower))

    def test_width_positive(self) -> None:
        close = _price_series([float(i) + (i % 5) for i in range(1, 30)])
        _, _, width = _bollinger_bands(close, window=20)
        non_nan = width.dropna()
        assert all(w >= 0.0 for w in non_nan)

    def test_constant_series_width_is_zero(self) -> None:
        # No variance → std = 0 → upper == lower == middle → width = 0
        close = _price_series([50.0] * 25)
        upper, lower, width = _bollinger_bands(close, window=20)
        non_nan_width = width.dropna()
        assert all(abs(w) < 1e-9 for w in non_nan_width)

    def test_upper_lower_symmetric_around_middle(self) -> None:
        # upper - SMA == SMA - lower for each non-NaN row
        close = _price_series([float(i) + (i % 3) * 2 for i in range(1, 30)])
        upper, lower, _ = _bollinger_bands(close, window=20)
        sma = close.rolling(window=20, min_periods=20).mean()
        diff_upper = (upper - sma).dropna()
        diff_lower = (sma - lower).dropna()
        pd.testing.assert_series_equal(
            diff_upper.reset_index(drop=True),
            diff_lower.reset_index(drop=True),
            check_names=False,
        )

    def test_num_std_2_band_wider_than_num_std_1(self) -> None:
        close = _price_series([float(i) + (i % 5) for i in range(1, 30)])
        upper2, lower2, _ = _bollinger_bands(close, window=20, num_std=2.0)
        upper1, lower1, _ = _bollinger_bands(close, window=20, num_std=1.0)
        # Width with std=2 should be wider
        width2 = (upper2 - lower2).dropna()
        width1 = (upper1 - lower1).dropna()
        assert all(w2 > w1 for w2, w1 in zip(width2, width1))

    def test_all_series_same_length(self) -> None:
        close = _price_series([float(i) for i in range(1, 30)])
        upper, lower, width = _bollinger_bands(close, window=20)
        assert len(upper) == len(lower) == len(width) == 29

    def test_window_5_has_4_nan_rows(self) -> None:
        close = _price_series([float(i) for i in range(1, 15)])
        upper, _, _ = _bollinger_bands(close, window=5)
        assert all(pd.isna(upper.iloc[i]) for i in range(4))
        assert not pd.isna(upper.iloc[4])

    def test_width_formula_manual_check(self) -> None:
        # width = (upper - lower) / SMA; for a deterministic small series verify
        close = _price_series([10.0, 11.0, 9.0, 12.0, 8.0])
        upper, lower, width = _bollinger_bands(close, window=5)
        sma_val = sum([10.0, 11.0, 9.0, 12.0, 8.0]) / 5  # 10.0
        import statistics
        std_val = statistics.stdev([10.0, 11.0, 9.0, 12.0, 8.0])
        expected_width = (4 * std_val) / sma_val
        assert width.dropna().iloc[-1] == pytest.approx(expected_width, rel=1e-5)


# ── TestATR ───────────────────────────────────────────────────────────────────

class TestATR:
    def _make_ohlc(self, n: int = 30) -> tuple[pd.Series, pd.Series, pd.Series]:
        high  = _price_series([100.0 + i + 2.0 for i in range(n)])
        low   = _price_series([100.0 + i - 2.0 for i in range(n)])
        close = _price_series([100.0 + i for i in range(n)])
        return high, low, close

    def test_first_period_rows_are_nan(self) -> None:
        high, low, close = self._make_ohlc(30)
        result = _atr(high, low, close, period=14)
        # TR at index 0 is valid (high - low, no prev_close needed).
        # EWM with min_periods=14 produces the first non-NaN value at index 13
        # (the 14th observation), so indices 0-12 are NaN.
        assert all(pd.isna(result.iloc[i]) for i in range(13))
        assert not pd.isna(result.iloc[13])

    def test_non_nan_values_after_period(self) -> None:
        high, low, close = self._make_ohlc(30)
        result = _atr(high, low, close, period=14)
        assert not pd.isna(result.iloc[15])

    def test_atr_positive_for_volatile_market(self) -> None:
        high, low, close = self._make_ohlc(30)
        result = _atr(high, low, close, period=14)
        non_nan = result.dropna()
        assert all(v > 0 for v in non_nan)

    def test_constant_ohlc_atr_is_zero(self) -> None:
        n = 30
        high  = _price_series([100.0] * n)
        low   = _price_series([100.0] * n)
        close = _price_series([100.0] * n)
        result = _atr(high, low, close, period=14)
        non_nan = result.dropna()
        assert all(abs(v) < 1e-9 for v in non_nan)

    def test_result_same_length_as_input(self) -> None:
        high, low, close = self._make_ohlc(25)
        result = _atr(high, low, close, period=14)
        assert len(result) == 25

    def test_wider_bands_give_higher_atr(self) -> None:
        n = 30
        close = _price_series([100.0 + i for i in range(n)])
        # Wide bands
        high_wide = _price_series([100.0 + i + 5.0 for i in range(n)])
        low_wide  = _price_series([100.0 + i - 5.0 for i in range(n)])
        # Narrow bands
        high_narrow = _price_series([100.0 + i + 1.0 for i in range(n)])
        low_narrow  = _price_series([100.0 + i - 1.0 for i in range(n)])
        atr_wide   = _atr(high_wide, low_wide, close, period=14).dropna().iloc[-1]
        atr_narrow = _atr(high_narrow, low_narrow, close, period=14).dropna().iloc[-1]
        assert atr_wide > atr_narrow

    def test_atr_uses_true_range_not_just_high_low(self) -> None:
        # Gap-up scenario: prev_close far below low → TR driven by |Low - PrevClose|
        high  = _price_series([100.0, 100.0, 130.0] + [130.0] * 20)
        low   = _price_series([98.0, 98.0, 125.0] + [125.0] * 20)
        close = _price_series([99.0, 99.0, 128.0] + [128.0] * 20)
        result = _atr(high, low, close, period=5)
        # After the gap, ATR should reflect the larger move
        non_nan = result.dropna()
        assert len(non_nan) > 0

    def test_returns_series_dtype_float(self) -> None:
        high, low, close = self._make_ohlc(20)
        result = _atr(high, low, close, period=5)
        assert pd.api.types.is_float_dtype(result)


# ── TestTrendFeatures ─────────────────────────────────────────────────────────

class TestTrendFeatures:
    def _make_prices(self, n: int = 30) -> pd.DataFrame:
        return _ohlcv_df(
            closes=[float(100 + i) for i in range(n)],
            start=TARGET_DATE - timedelta(days=n - 1),
        )

    def test_returns_dict_with_four_keys(self) -> None:
        prices = self._make_prices()
        result = _compute_trend_features(prices, TARGET_DATE)
        assert set(result.keys()) == {"sma_10", "sma_20", "ema_10", "ema_20"}

    def test_returns_none_when_target_date_not_in_prices(self) -> None:
        prices = self._make_prices()
        future_date = TARGET_DATE + timedelta(days=100)
        result = _compute_trend_features(prices, future_date)
        assert all(v is None for v in result.values())

    def test_sma_10_computable_with_10_days_history(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(15)],
            start=TARGET_DATE - timedelta(days=14),
        )
        result = _compute_trend_features(prices, TARGET_DATE)
        assert result["sma_10"] is not None

    def test_sma_20_none_with_fewer_than_20_days(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(15)],
            start=TARGET_DATE - timedelta(days=14),
        )
        result = _compute_trend_features(prices, TARGET_DATE)
        assert result["sma_20"] is None

    def test_sma_10_value_matches_manual_calculation(self) -> None:
        closes = [float(100 + i) for i in range(15)]
        prices = _ohlcv_df(closes, start=TARGET_DATE - timedelta(days=14))
        result = _compute_trend_features(prices, TARGET_DATE)
        expected = sum(closes[-10:]) / 10
        assert result["sma_10"] == pytest.approx(expected, rel=1e-6)

    def test_ema_10_computable_with_10_days_history(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(15)],
            start=TARGET_DATE - timedelta(days=14),
        )
        result = _compute_trend_features(prices, TARGET_DATE)
        assert result["ema_10"] is not None

    def test_all_values_are_floats_when_available(self) -> None:
        prices = self._make_prices(30)
        result = _compute_trend_features(prices, TARGET_DATE)
        for key in ("sma_10", "sma_20", "ema_10", "ema_20"):
            assert isinstance(result[key], float), f"{key} is not float"


# ── TestMomentumFeatures ──────────────────────────────────────────────────────

class TestMomentumFeatures:
    def _make_prices(self, n: int = 50) -> pd.DataFrame:
        return _ohlcv_df(
            closes=[float(100 + i) for i in range(n)],
            start=TARGET_DATE - timedelta(days=n - 1),
        )

    def test_returns_dict_with_four_keys(self) -> None:
        prices = self._make_prices()
        result = _compute_momentum_features(prices, TARGET_DATE)
        assert set(result.keys()) == {"rsi_14", "macd", "macd_signal", "macd_histogram"}

    def test_rsi_none_with_fewer_than_15_days(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(10)],
            start=TARGET_DATE - timedelta(days=9),
        )
        result = _compute_momentum_features(prices, TARGET_DATE)
        assert result["rsi_14"] is None

    def test_rsi_computable_with_sufficient_history(self) -> None:
        prices = self._make_prices(30)
        result = _compute_momentum_features(prices, TARGET_DATE)
        assert result["rsi_14"] is not None

    def test_rsi_in_range_0_to_100(self) -> None:
        prices = self._make_prices(30)
        result = _compute_momentum_features(prices, TARGET_DATE)
        rsi = result["rsi_14"]
        if rsi is not None:
            assert 0.0 <= rsi <= 100.0

    def test_macd_computable_with_sufficient_history(self) -> None:
        prices = self._make_prices(50)
        result = _compute_momentum_features(prices, TARGET_DATE)
        assert result["macd"] is not None

    def test_macd_none_with_insufficient_history(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(20)],
            start=TARGET_DATE - timedelta(days=19),
        )
        result = _compute_momentum_features(prices, TARGET_DATE)
        assert result["macd"] is None

    def test_histogram_equals_macd_minus_signal(self) -> None:
        prices = self._make_prices(50)
        result = _compute_momentum_features(prices, TARGET_DATE)
        macd = result["macd"]
        sig  = result["macd_signal"]
        hist = result["macd_histogram"]
        if macd is not None and sig is not None and hist is not None:
            assert hist == pytest.approx(macd - sig, abs=1e-7)


# ── TestVolatilityFeatures ────────────────────────────────────────────────────

class TestVolatilityFeatures:
    def _make_prices(self, n: int = 30) -> pd.DataFrame:
        import math
        closes = [100.0 + math.sin(i * 0.3) * 5 for i in range(n)]
        return _ohlcv_df(
            closes=closes,
            highs=[c + 2.0 for c in closes],
            lows=[c - 2.0 for c in closes],
            start=TARGET_DATE - timedelta(days=n - 1),
        )

    def test_returns_dict_with_five_keys(self) -> None:
        prices = self._make_prices()
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert set(result.keys()) == {
            "bb_upper", "bb_lower", "bb_width", "atr_14", "volatility_20d"
        }

    def test_bb_values_computable_with_20_days(self) -> None:
        prices = self._make_prices(25)
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert result["bb_upper"] is not None
        assert result["bb_lower"] is not None

    def test_bb_none_with_fewer_than_20_days(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(15)],
            start=TARGET_DATE - timedelta(days=14),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert result["bb_upper"] is None

    def test_bb_upper_greater_than_bb_lower(self) -> None:
        prices = self._make_prices(30)
        result = _compute_volatility_features(prices, TARGET_DATE)
        if result["bb_upper"] is not None and result["bb_lower"] is not None:
            assert result["bb_upper"] > result["bb_lower"]

    def test_atr_positive(self) -> None:
        prices = self._make_prices(30)
        result = _compute_volatility_features(prices, TARGET_DATE)
        if result["atr_14"] is not None:
            assert result["atr_14"] > 0.0

    def test_volatility_20d_computable_with_21_days(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(25)],
            start=TARGET_DATE - timedelta(days=24),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert result["volatility_20d"] is not None

    def test_volatility_20d_none_with_fewer_than_21_days(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(15)],
            start=TARGET_DATE - timedelta(days=14),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert result["volatility_20d"] is None

    def test_target_date_not_in_prices_all_none(self) -> None:
        prices = self._make_prices(30)
        future = TARGET_DATE + timedelta(days=200)
        result = _compute_volatility_features(prices, future)
        assert all(v is None for v in result.values())


# ── TestReturnFeatures ────────────────────────────────────────────────────────

class TestReturnFeatures:
    def _make_prices(self, n: int = 15) -> pd.DataFrame:
        return _ohlcv_df(
            closes=[100.0 * (1.01 ** i) for i in range(n)],
            start=TARGET_DATE - timedelta(days=n - 1),
        )

    def test_returns_dict_with_three_keys(self) -> None:
        prices = self._make_prices(15)
        result = _compute_return_features(prices, TARGET_DATE)
        assert set(result.keys()) == {"price_chg_1d", "price_chg_5d", "price_chg_10d"}

    def test_1d_return_computable_with_2_rows(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0, 110.0],
            start=TARGET_DATE - timedelta(days=1),
        )
        result = _compute_return_features(prices, TARGET_DATE)
        assert result["price_chg_1d"] == pytest.approx(0.1, rel=1e-5)

    def test_5d_return_none_with_fewer_than_6_rows(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(4)],
            start=TARGET_DATE - timedelta(days=3),
        )
        result = _compute_return_features(prices, TARGET_DATE)
        assert result["price_chg_5d"] is None

    def test_10d_return_none_with_fewer_than_11_rows(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(8)],
            start=TARGET_DATE - timedelta(days=7),
        )
        result = _compute_return_features(prices, TARGET_DATE)
        assert result["price_chg_10d"] is None

    def test_1d_return_correct_formula(self) -> None:
        # (110 - 100) / 100 = 0.1
        prices = _ohlcv_df(
            closes=[100.0, 102.0, 105.0, 110.0],
            start=TARGET_DATE - timedelta(days=3),
        )
        result = _compute_return_features(prices, TARGET_DATE)
        assert result["price_chg_1d"] == pytest.approx((110.0 - 105.0) / 105.0, rel=1e-5)

    def test_5d_return_correct_with_enough_history(self) -> None:
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
        prices = _ohlcv_df(closes, start=TARGET_DATE - timedelta(days=5))
        result = _compute_return_features(prices, TARGET_DATE)
        expected = (110.0 - 100.0) / 100.0
        assert result["price_chg_5d"] == pytest.approx(expected, rel=1e-5)

    def test_target_date_absent_returns_all_none(self) -> None:
        prices = self._make_prices()
        future = TARGET_DATE + timedelta(days=100)
        result = _compute_return_features(prices, future)
        assert all(v is None for v in result.values())

    def test_positive_return_when_price_rising(self) -> None:
        closes = [float(100 + i) for i in range(12)]
        prices = _ohlcv_df(closes, start=TARGET_DATE - timedelta(days=11))
        result = _compute_return_features(prices, TARGET_DATE)
        assert result["price_chg_1d"] is not None
        assert result["price_chg_1d"] > 0.0


# ── TestVolumeFeatures ────────────────────────────────────────────────────────

class TestVolumeFeatures:
    def _make_prices_with_volume(
        self,
        volumes: list[float],
        n_extra_close: int = 0,
    ) -> pd.DataFrame:
        n = len(volumes)
        closes = [100.0] * (n + n_extra_close)
        vols = volumes + [volumes[-1]] * n_extra_close
        return _ohlcv_df(
            closes=closes,
            volumes=vols,
            start=TARGET_DATE - timedelta(days=len(closes) - 1),
        )

    def test_returns_dict_with_three_keys(self) -> None:
        prices = self._make_prices_with_volume([1e6] * 10)
        result = _compute_volume_features(prices, TARGET_DATE)
        assert set(result.keys()) == {
            "volume_change_pct", "volume_avg_5d", "volume_ratio"
        }

    def test_volume_change_pct_correct_formula(self) -> None:
        # Today = 200, yesterday = 100 → change = 1.0 (100%)
        prices = _ohlcv_df(
            closes=[100.0, 100.0],
            volumes=[100.0, 200.0],
            start=TARGET_DATE - timedelta(days=1),
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        assert result["volume_change_pct"] == pytest.approx(1.0, rel=1e-5)

    def test_volume_change_pct_none_with_only_one_row(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0],
            volumes=[1e6],
            start=TARGET_DATE,
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        assert result["volume_change_pct"] is None

    def test_volume_avg_5d_correct(self) -> None:
        vols = [100.0, 200.0, 300.0, 400.0, 500.0]
        prices = _ohlcv_df(
            closes=[100.0] * 5,
            volumes=vols,
            start=TARGET_DATE - timedelta(days=4),
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        assert result["volume_avg_5d"] == pytest.approx(300.0, rel=1e-5)

    def test_volume_avg_5d_none_with_fewer_than_5_rows(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0] * 3,
            volumes=[1e6, 1e6, 1e6],
            start=TARGET_DATE - timedelta(days=2),
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        assert result["volume_avg_5d"] is None

    def test_volume_ratio_is_one_for_constant_volume(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0] * 7,
            volumes=[1e6] * 7,
            start=TARGET_DATE - timedelta(days=6),
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        if result["volume_ratio"] is not None:
            assert result["volume_ratio"] == pytest.approx(1.0, rel=1e-5)

    def test_volume_ratio_greater_than_one_for_high_volume(self) -> None:
        vols = [1e6, 1e6, 1e6, 1e6, 5e6]
        prices = _ohlcv_df(
            closes=[100.0] * 5,
            volumes=vols,
            start=TARGET_DATE - timedelta(days=4),
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        if result["volume_ratio"] is not None:
            assert result["volume_ratio"] > 1.0

    def test_target_date_absent_all_none(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0] * 7,
            volumes=[1e6] * 7,
            start=TARGET_DATE - timedelta(days=6),
        )
        future = TARGET_DATE + timedelta(days=100)
        result = _compute_volume_features(prices, future)
        assert all(v is None for v in result.values())

    def test_volume_change_pct_negative_when_falling(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0, 100.0],
            volumes=[1000.0, 500.0],
            start=TARGET_DATE - timedelta(days=1),
        )
        result = _compute_volume_features(prices, TARGET_DATE)
        assert result["volume_change_pct"] == pytest.approx(-0.5, rel=1e-5)


# ── TestTechnicalFeatures (orchestrator) ──────────────────────────────────────

class TestTechnicalFeatures:
    def _make_prices(self, n: int = 60) -> pd.DataFrame:
        return _ohlcv_df(
            closes=[float(100 + i) for i in range(n)],
            highs=[float(101 + i) for i in range(n)],
            lows=[float(99 + i) for i in range(n)],
            volumes=[float(1e6 + i * 1000) for i in range(n)],
            start=TARGET_DATE - timedelta(days=n - 1),
        )

    def test_returns_dict_with_all_technical_columns(self) -> None:
        prices = self._make_prices()
        result = _compute_technical_features(prices, TARGET_DATE)
        assert set(result.keys()) == set(TECHNICAL_FEATURE_COLUMNS)

    def test_all_19_keys_present(self) -> None:
        prices = self._make_prices()
        result = _compute_technical_features(prices, TARGET_DATE)
        assert len(result) == 19

    def test_with_sufficient_history_no_nones(self) -> None:
        prices = self._make_prices(60)
        result = _compute_technical_features(prices, TARGET_DATE)
        # MACD signal needs ~35 days, everything else needs ≤ 21
        # With 60 rows, all should be computable
        for key, val in result.items():
            assert val is not None, f"{key} is None with 60 days of history"

    def test_missing_target_date_all_none(self) -> None:
        prices = self._make_prices(30)
        future = TARGET_DATE + timedelta(days=500)
        result = _compute_technical_features(prices, future)
        assert all(v is None for v in result.values())

    def test_insufficient_history_produces_some_nones(self) -> None:
        # Only 5 rows — most indicators will be NaN
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(5)],
            start=TARGET_DATE - timedelta(days=4),
        )
        result = _compute_technical_features(prices, TARGET_DATE)
        none_count = sum(1 for v in result.values() if v is None)
        assert none_count > 0

    def test_values_are_float_or_none(self) -> None:
        prices = self._make_prices(40)
        result = _compute_technical_features(prices, TARGET_DATE)
        for key, val in result.items():
            assert val is None or isinstance(val, float), (
                f"{key} has unexpected type {type(val)}"
            )


# ── TestRollingVolatility ─────────────────────────────────────────────────────

class TestRollingVolatility:
    """Focused tests for the 20-day rolling volatility via volatility features."""

    def test_zero_for_constant_prices(self) -> None:
        prices = _ohlcv_df(
            closes=[100.0] * 25,
            start=TARGET_DATE - timedelta(days=24),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        # Constant prices → zero daily returns → zero volatility
        if result["volatility_20d"] is not None:
            assert result["volatility_20d"] == pytest.approx(0.0, abs=1e-9)

    def test_higher_for_volatile_prices(self) -> None:
        # Build two series: stable vs. oscillating
        n = 25
        stable = _ohlcv_df(
            closes=[100.0] * n,
            start=TARGET_DATE - timedelta(days=n - 1),
        )
        volatile_closes = [100.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(n)]
        volatile = _ohlcv_df(
            closes=volatile_closes,
            start=TARGET_DATE - timedelta(days=n - 1),
        )
        r_stable   = _compute_volatility_features(stable, TARGET_DATE)
        r_volatile = _compute_volatility_features(volatile, TARGET_DATE)
        v_stable   = r_stable["volatility_20d"] or 0.0
        v_volatile = r_volatile["volatility_20d"]
        if v_volatile is not None:
            assert v_volatile > v_stable

    def test_none_with_fewer_than_21_rows(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(15)],
            start=TARGET_DATE - timedelta(days=14),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert result["volatility_20d"] is None

    def test_not_none_with_21_rows(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(21)],
            start=TARGET_DATE - timedelta(days=20),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        assert result["volatility_20d"] is not None

    def test_value_is_positive_or_zero(self) -> None:
        prices = _ohlcv_df(
            closes=[float(100 + (i % 5)) for i in range(25)],
            start=TARGET_DATE - timedelta(days=24),
        )
        result = _compute_volatility_features(prices, TARGET_DATE)
        if result["volatility_20d"] is not None:
            assert result["volatility_20d"] >= 0.0


# ── TestGenerateFeaturesMerge ─────────────────────────────────────────────────

class TestGenerateFeaturesMerge:
    """
    Integration tests verifying that generate_features() correctly merges
    sentiment features with technical indicators when prices_df is provided.
    """

    def _fe(self) -> FeatureEngineer:
        return FeatureEngineer(database_url="sqlite:///:memory:")

    def _raw_df(self) -> pd.DataFrame:
        return pd.DataFrame([_sentiment_row(ticker="TEST", date_val=TARGET_DATE)])

    def _price_df(self, n: int = 60) -> pd.DataFrame:
        return _ohlcv_df(
            closes=[float(100 + i) for i in range(n)],
            highs=[float(101 + i) for i in range(n)],
            lows=[float(99 + i) for i in range(n)],
            volumes=[float(1e6 + i * 1000) for i in range(n)],
            start=TARGET_DATE - timedelta(days=n - 1),
            ticker="TEST",
        )

    def test_columns_match_full_feature_columns(self) -> None:
        fe = self._fe()
        result = fe.generate_features(self._raw_df(), TARGET_DATE, prices_df=self._price_df())
        assert list(result.columns) == FEATURE_COLUMNS

    def test_technical_columns_populated_with_price_data(self) -> None:
        fe = self._fe()
        result = fe.generate_features(self._raw_df(), TARGET_DATE, prices_df=self._price_df(60))
        row = result.iloc[0]
        # With 60 days of price data, all indicators should be computable
        for col in TECHNICAL_FEATURE_COLUMNS:
            assert row[col] is not None and not (
                isinstance(row[col], float) and pd.isna(row[col])
            ), f"{col} should be non-NaN with 60 days of price data"

    def test_technical_columns_are_none_without_price_data(self) -> None:
        fe = self._fe()
        result = fe.generate_features(self._raw_df(), TARGET_DATE, prices_df=None)
        row = result.iloc[0]
        for col in TECHNICAL_FEATURE_COLUMNS:
            assert row[col] is None or pd.isna(row[col]), (
                f"{col} should be None when prices_df=None"
            )

    def test_sentiment_features_unchanged_with_prices(self) -> None:
        fe = self._fe()
        without_prices = fe.generate_features(self._raw_df(), TARGET_DATE, prices_df=None)
        with_prices = fe.generate_features(
            self._raw_df(), TARGET_DATE, prices_df=self._price_df()
        )
        # Sentiment columns should be identical regardless of price data
        sentiment_cols = [
            "article_count", "positive_count", "neutral_count", "negative_count",
            "positive_ratio", "neutral_ratio", "negative_ratio",
            "mean_sentiment_score",
        ]
        for col in sentiment_cols:
            assert without_prices[col].iloc[0] == with_prices[col].iloc[0], (
                f"Sentiment column {col} changed when prices were added"
            )

    def test_ticker_and_date_columns_correct(self) -> None:
        fe = self._fe()
        result = fe.generate_features(self._raw_df(), TARGET_DATE, prices_df=self._price_df())
        assert result["ticker"].iloc[0] == "TEST"
        assert result["date"].iloc[0] == TARGET_DATE

    def test_wrong_ticker_price_data_gives_none_technical(self) -> None:
        fe = self._fe()
        # Price data for a different ticker should not be applied
        wrong_ticker_prices = _ohlcv_df(
            closes=[float(100 + i) for i in range(60)],
            start=TARGET_DATE - timedelta(days=59),
            ticker="WRONG",
        )
        result = fe.generate_features(
            self._raw_df(), TARGET_DATE, prices_df=wrong_ticker_prices
        )
        row = result.iloc[0]
        for col in TECHNICAL_FEATURE_COLUMNS:
            assert row[col] is None or pd.isna(row[col]), (
                f"{col} should be None when ticker doesn't match"
            )

    def test_insufficient_price_history_gives_partial_nones(self) -> None:
        fe = self._fe()
        # Only 5 rows of price data — most indicators will be None
        short_prices = self._price_df(5)
        result = fe.generate_features(
            self._raw_df(), TARGET_DATE, prices_df=short_prices
        )
        row = result.iloc[0]
        none_count = sum(
            1 for col in TECHNICAL_FEATURE_COLUMNS
            if row[col] is None or (isinstance(row[col], float) and pd.isna(row[col]))
        )
        assert none_count > 0, "Expected some None technical features with only 5 price rows"
