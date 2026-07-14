"""
Higher-level analytics built purely by *aggregating* and *re-plotting* data
that the existing backend already produces. No pipeline math is duplicated:

* Technical indicators reuse the exact private helper functions from
  :mod:`src.features.feature_engineer` (``_sma``, ``_ema``, ``_rsi``,
  ``_macd_lines``, ``_bollinger_bands``, ``_atr``) so chart values are
  identical to what Phase 4 stores as ML features.
* Sentiment analytics aggregate FinBERT results already stored by Phase 2/3.
* Model explainability wraps SHAP around the already-trained XGBoost
  artifact loaded via :mod:`app.services.prediction_service`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.services import database_service, prediction_service
from src.features.feature_engineer import (
    _atr,
    _bollinger_bands,
    _ema,
    _macd_lines,
    _rsi,
    _sma,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ML_DATA_DIR = _PROJECT_ROOT / "data" / "ml"


# ── Technical analysis (Page 5) ─────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def compute_technical_indicators_series(ticker: str) -> pd.DataFrame:
    """
    Full OHLCV history for *ticker* with technical-indicator columns computed
    over the whole series (for charting), using the same primitives Phase 4
    uses to build the single-date ML feature snapshot.
    """
    df = database_service.get_prices_df(ticker)
    if df.empty:
        return df

    df = df.sort_values("trading_date").reset_index(drop=True)
    close = df["close_price"].astype(float)
    high = df["high_price"].astype(float)
    low = df["low_price"].astype(float)
    volume = df["volume"].astype(float)

    df["sma_10"] = _sma(close, 10)
    df["sma_20"] = _sma(close, 20)
    df["ema_10"] = _ema(close, 10)
    df["ema_20"] = _ema(close, 20)
    df["rsi_14"] = _rsi(close, 14)

    macd_line, signal_line, histogram = _macd_lines(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram

    bb_upper, bb_lower, bb_width = _bollinger_bands(close, window=20)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width

    df["atr_14"] = _atr(high, low, close, 14)
    df["volume_avg_5d"] = volume.rolling(window=5, min_periods=1).mean()

    return df


# ── Sentiment analytics (Page 6) ────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_sentiment_daily_trend(
    tickers: tuple[str, ...] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Daily positive/neutral/negative counts, mean score, and rolling averages."""
    raw = database_service.get_sentiment_articles_df(tickers=tickers, start=start, end=end)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "day", "positive", "neutral", "negative", "total",
                "mean_sentiment_score", "mean_confidence", "rolling_7d_mean",
            ]
        )

    raw["published_at"] = pd.to_datetime(raw["published_at"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["published_at"])
    raw["day"] = raw["published_at"].dt.date

    grouped = raw.groupby("day")
    out = grouped.agg(
        positive=("sentiment_label", lambda s: int((s == "positive").sum())),
        neutral=("sentiment_label", lambda s: int((s == "neutral").sum())),
        negative=("sentiment_label", lambda s: int((s == "negative").sum())),
        total=("sentiment_label", "size"),
        mean_sentiment_score=("sentiment_score", "mean"),
        mean_confidence=("sentiment_confidence", "mean"),
    ).reset_index().sort_values("day")

    out["rolling_7d_mean"] = out["mean_sentiment_score"].rolling(window=7, min_periods=1).mean()
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_ticker_sentiment_summary(tickers: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Per-ticker sentiment share and average confidence."""
    raw = database_service.get_sentiment_articles_df(tickers=tickers)
    if raw.empty:
        return pd.DataFrame(
            columns=["ticker", "positive", "neutral", "negative", "total", "mean_confidence"]
        )

    grouped = raw.groupby("ticker")
    out = grouped.agg(
        positive=("sentiment_label", lambda s: int((s == "positive").sum())),
        neutral=("sentiment_label", lambda s: int((s == "neutral").sum())),
        negative=("sentiment_label", lambda s: int((s == "negative").sum())),
        total=("sentiment_label", "size"),
        mean_confidence=("sentiment_confidence", "mean"),
    ).reset_index().sort_values("total", ascending=False)
    return out


# ── Model explainability / SHAP (Page 7) ────────────────────────────────────────

def _find_latest_ml_dataset() -> Path | None:
    if not _ML_DATA_DIR.exists():
        return None
    candidates = [p for p in _ML_DATA_DIR.glob("*.csv") if "_predictions" not in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@st.cache_data(ttl=300, show_spinner=False)
def load_background_dataset(
    feature_columns: tuple[str, ...], sample_size: int = 300
) -> pd.DataFrame | None:
    """Sample real historical feature rows for global SHAP context, if available."""
    path = _find_latest_ml_dataset()
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    missing = set(feature_columns) - set(df.columns)
    if missing:
        return None
    X = df[list(feature_columns)].astype(float)
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
    return X.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_shap_explainer() -> Any:
    """Cached ``shap.TreeExplainer`` wrapping the trained XGBoost model."""
    import shap  # noqa: PLC0415 — lazy import, heavy dependency

    artifact = prediction_service.get_model_artifact()
    return shap.TreeExplainer(artifact["model"])


def feature_dict_to_frame(
    feature_dict: dict[str, float], feature_columns: list[str]
) -> pd.DataFrame:
    """Align an arbitrary feature dict (e.g. ``PredictionResult.feature_values``) to model order."""
    row = {c: feature_dict.get(c, np.nan) for c in feature_columns}
    return pd.DataFrame([row], columns=feature_columns).astype(float)


def explain_instance(feature_dict: dict[str, float]) -> Any:
    """Return a ``shap.Explanation`` for a single feature vector (all classes)."""
    artifact = prediction_service.get_model_artifact()
    feature_columns: list[str] = list(artifact["feature_columns"])
    frame = feature_dict_to_frame(feature_dict, feature_columns)
    explainer = get_shap_explainer()
    return explainer(frame)


@st.cache_data(ttl=300, show_spinner=False)
def compute_global_shap(sample_size: int = 200) -> dict[str, Any] | None:
    """
    Global SHAP summary over a sample of real historical feature rows.

    Returns ``None`` when no local ML dataset is available (e.g. fresh clone
    without ``data/``) — the page falls back to the model's built-in
    feature importances in that case.
    """
    artifact = prediction_service.get_model_artifact()
    feature_columns: list[str] = list(artifact["feature_columns"])
    classes: list[str] = list(artifact["label_encoder"].classes_)

    background = load_background_dataset(tuple(feature_columns), sample_size=sample_size)
    if background is None or background.empty:
        return None

    explainer = get_shap_explainer()
    explanation = explainer(background)

    values = np.asarray(explanation.values)  # (n_samples, n_features, n_classes)
    mean_abs_per_class = np.abs(values).mean(axis=0)  # (n_features, n_classes)
    mean_abs_overall = mean_abs_per_class.mean(axis=1)  # (n_features,)

    return {
        "feature_columns": feature_columns,
        "classes": classes,
        "sample": background,
        "values": values,
        "mean_abs_per_class": mean_abs_per_class,
        "mean_abs_overall": mean_abs_overall,
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_builtin_feature_importance() -> pd.DataFrame:
    """XGBoost's native gain-based feature importance (no SHAP required)."""
    artifact = prediction_service.get_model_artifact()
    model = artifact["model"]
    feature_columns: list[str] = list(artifact["feature_columns"])
    importances = np.asarray(model.feature_importances_)
    df = pd.DataFrame({"feature": feature_columns, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_plain_english_explanation(
    feature_dict: dict[str, float],
    shap_row: np.ndarray,
    feature_columns: list[str],
    predicted_class: str,
    top_n: int = 5,
) -> list[str]:
    """
    Turn a single-class SHAP contribution vector into plain-English bullets,
    e.g. "High RSI (72.4) pushed the model toward SELL."
    """
    contributions = list(zip(feature_columns, shap_row))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    bullets: list[str] = []
    for name, shap_val in contributions[:top_n]:
        value = feature_dict.get(name)
        direction = "toward" if shap_val > 0 else "away from"
        magnitude = "strongly" if abs(shap_val) > np.abs(shap_row).mean() * 2 else "mildly"
        value_str = f"{value:.4f}" if isinstance(value, (int, float)) and value is not None else "N/A"
        bullets.append(
            f"**{name}** = `{value_str}` pushed the model {magnitude} {direction} "
            f"**{predicted_class}** (SHAP contribution: `{shap_val:+.4f}`)."
        )
    return bullets
