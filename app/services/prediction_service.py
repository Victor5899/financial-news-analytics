"""
Prediction / model service — wraps Phase 7 & Phase 8 backend objects.

Every function here is a thin, cached pass-through to existing modules:

* :class:`src.realtime.realtime_pipeline.RealtimePipeline` — live inference
* :class:`src.realtime.finnhub_client.FinnhubClient` — live news
* :class:`src.model.predictor.ModelPredictor` — batch / vector inference
* :mod:`src.model.model_io` — raw artifact loading (model + label encoder)

No prediction, feature-generation, or sentiment logic is reimplemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.model.model_io import load_model as load_model_artifact
from src.model.predictor import ModelPredictor
from src.processing.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.realtime.finnhub_client import FinnhubClient, FinnhubError
from src.realtime.realtime_pipeline import (
    NoNewsError,
    PredictionResult,
    RealtimePipeline,
    RealtimePipelineError,
)
from src.utils.config import settings

__all__ = [
    "FinnhubError",
    "NoNewsError",
    "PredictionResult",
    "RealtimePipelineError",
    "SUPPORTED_TICKERS",
    "get_finnhub_client",
    "get_model_artifact",
    "get_model_best_params",
    "get_model_metrics",
    "get_model_predictor",
    "get_pipeline",
    "run_live_prediction",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "models" / "xgboost_direction_model.joblib"
_METRICS_PATH = _PROJECT_ROOT / "artifacts" / "metrics" / "xgboost_metrics.json"
_BEST_PARAMS_PATH = _PROJECT_ROOT / "artifacts" / "metrics" / "xgboost_best_params.json"

#: Tickers exposed in the UI (mirrors the project's default tracked universe).
SUPPORTED_TICKERS: list[str] = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN"]


# ── Pipeline / model (cached resources — expensive to load) ────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline(lookback_days: int = 7) -> RealtimePipeline:
    """Cached :class:`RealtimePipeline` — loads FinBERT + XGBoost once per session."""
    return RealtimePipeline(model_path=_MODEL_PATH, lookback_days=lookback_days)


@st.cache_resource(show_spinner=False)
def get_finnhub_client() -> FinnhubClient:
    """Cached Finnhub client, reused by the Live News page."""
    return FinnhubClient()


@st.cache_resource(show_spinner=False)
def get_model_predictor() -> ModelPredictor:
    """Cached :class:`ModelPredictor` for batch / explainability use."""
    return ModelPredictor(model_path=_MODEL_PATH).load_model()


@st.cache_resource(show_spinner=False)
def get_model_artifact() -> dict[str, Any]:
    """Raw joblib artifact bundle: ``{model, label_encoder, feature_columns, metadata}``."""
    return load_model_artifact(_MODEL_PATH)


# ── Metrics (small JSON files — cheap, but cached for consistency) ─────────────

@st.cache_data(show_spinner=False)
def get_model_metrics() -> dict[str, Any] | None:
    if not _METRICS_PATH.exists():
        return None
    with open(_METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_model_best_params() -> dict[str, Any] | None:
    if not _BEST_PARAMS_PATH.exists():
        return None
    with open(_BEST_PARAMS_PATH) as f:
        return json.load(f)


def model_file_status() -> dict[str, Any]:
    """Return artifact presence/metadata for status badges (Settings / System pages)."""
    exists = _MODEL_PATH.exists()
    info: dict[str, Any] = {"exists": exists, "path": str(_MODEL_PATH)}
    if exists:
        try:
            artifact = get_model_artifact()
            info["metadata"] = artifact.get("metadata", {})
            info["n_features"] = len(artifact.get("feature_columns", []))
            info["classes"] = list(artifact.get("label_encoder").classes_)
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
    return info


# ── Live prediction ─────────────────────────────────────────────────────────────

def run_live_prediction(ticker: str, *, save: bool = True) -> PredictionResult:
    """
    Run the full live prediction pipeline for *ticker*.

    Delegates entirely to :meth:`RealtimePipeline.predict`. Raises
    :class:`NoNewsError`, :class:`RealtimePipelineError`, or
    :class:`FinnhubError` — callers (pages) should catch these to show
    friendly UI messages.
    """
    pipeline = get_pipeline()
    return pipeline.predict(ticker.upper(), save=save)


def fetch_latest_news(ticker: str, *, days_back: int | None = None) -> list[dict[str, Any]]:
    """Fetch the latest Finnhub articles for *ticker* (used by the Live News page)."""
    client = get_finnhub_client()
    return client.fetch_latest_news(ticker.upper(), days_back=days_back)


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> FinBERTSentimentAnalyzer:
    """Cached FinBERT analyzer — same class the real-time pipeline uses internally."""
    analyzer = FinBERTSentimentAnalyzer(
        model_name=settings.finbert_model,
        batch_size=settings.finbert_batch_size,
        device=settings.finbert_device,
    )
    analyzer.load()
    return analyzer


def score_articles_sentiment(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach FinBERT sentiment (label/score/confidence) to each article dict in place."""
    if not articles:
        return articles
    analyzer = get_sentiment_analyzer()
    texts = [
        f"{a.get('title') or ''}. {a.get('description') or ''}".strip(". ") or None
        for a in articles
    ]
    results = analyzer.analyse_texts(texts)
    for article, sentiment in zip(articles, results):
        article["sentiment_label"] = sentiment["sentiment_label"]
        article["sentiment_score"] = sentiment["sentiment_score"]
        article["sentiment_confidence"] = sentiment["sentiment_confidence"]
    return articles


def api_key_status() -> bool:
    """Whether a (non-placeholder) Finnhub API key is configured."""
    key = (settings.finnhub_api_key or "").strip()
    return bool(key) and key != "your_finnhub_api_key_here"
