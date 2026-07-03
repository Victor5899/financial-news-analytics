"""
Unit tests for src.realtime.realtime_pipeline — RealtimePipeline.

Finnhub, FinBERT, feature engineering, and model inference are all mocked
so the suite runs without internet, GPU, or a trained artifact on disk.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from xgboost import XGBClassifier

from src.features.feature_engineer import FEATURE_COLUMNS
from src.model.model_io import save_model
from src.model.trainer import DIRECTION_CLASSES
from src.realtime.realtime_pipeline import (
    NoNewsError,
    RealtimePipeline,
    RealtimePipelineError,
    _get_model_predictor,
    reset_model_cache,
)
from src.storage.models import Base

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    reset_model_cache()
    yield
    reset_model_cache()


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    mock.database_url = "sqlite:///:memory:"
    mock.finbert_model = "ProsusAI/finbert"
    mock.finbert_batch_size = 32
    mock.finbert_device = "cpu"
    monkeypatch.setattr("src.realtime.realtime_pipeline.settings", mock)
    return mock


@pytest.fixture
def artifact_path(tmp_path: Path) -> Path:
    """Minimal XGBoost artifact using a subset of FEATURE_COLUMNS."""
    feature_cols = [
        "article_count",
        "mean_sentiment_score",
        "positive_ratio",
        "rsi_14",
        "price_chg_1d",
    ]
    rng = np.random.default_rng(42)
    n = 60
    X = rng.uniform(0, 1, (n, len(feature_cols))).astype(float)

    le = LabelEncoder()
    le.fit(DIRECTION_CLASSES)
    y = rng.choice(len(DIRECTION_CLASSES), size=n)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=5,
        random_state=42,
    )
    model.fit(X, y)

    path = tmp_path / "model.joblib"
    save_model(
        {
            "model":           model,
            "label_encoder":   le,
            "feature_columns": feature_cols,
            "metadata":        {"n_features": len(feature_cols), "classes": list(le.classes_)},
        },
        path,
    )
    return path


@pytest.fixture
def sample_article() -> dict:
    return {
        "ticker":       "AAPL",
        "source_id":    "100001",
        "source_name":  "Reuters",
        "author":       None,
        "title":        "Apple announces new AI features",
        "description":  "Apple unveiled new AI capabilities for iPhone.",
        "url":          "https://example.com/apple-ai",
        "published_at": datetime(2024, 1, 10, 9, 0, tzinfo=UTC),
        "content":      None,
        "fetched_at":   datetime(2024, 1, 10, 12, 0, tzinfo=UTC),
    }


@pytest.fixture
def mock_finnhub(sample_article: dict) -> MagicMock:
    client = MagicMock()
    client.fetch_latest_news.return_value = [sample_article]
    return client


@pytest.fixture
def mock_sentiment() -> MagicMock:
    analyzer = MagicMock()
    analyzer.analyse_texts.return_value = [{
        "sentiment_label":      "positive",
        "sentiment_score":      1,
        "sentiment_confidence": 0.82,
        "analysed_at":          "2024-01-10T12:00:00+00:00",
    }]
    return analyzer


def _make_features_df(feature_cols: list[str]) -> pd.DataFrame:
    row: dict = {"ticker": "AAPL", "date": date(2024, 1, 10)}
    for col in FEATURE_COLUMNS:
        if col in ("ticker", "date"):
            continue
        row[col] = 0.5 if col in feature_cols else 1.0
    return pd.DataFrame([row])


@pytest.fixture
def mock_feature_engineer(artifact_path: Path) -> MagicMock:
    from src.model.model_io import load_model  # noqa: PLC0415

    feature_cols = load_model(artifact_path)["feature_columns"]
    eng = MagicMock()
    eng.load_data.return_value = pd.DataFrame()
    eng.load_price_data.return_value = pd.DataFrame()
    eng.generate_features.return_value = _make_features_df(feature_cols)
    return eng


@pytest.fixture
def pipeline(
    mock_settings: MagicMock,
    mock_finnhub: MagicMock,
    mock_sentiment: MagicMock,
    mock_feature_engineer: MagicMock,
    artifact_path: Path,
) -> RealtimePipeline:
    return RealtimePipeline(
        database_url="sqlite:///:memory:",
        model_path=artifact_path,
        finnhub_client=mock_finnhub,
        sentiment_analyzer=mock_sentiment,
        feature_engineer=mock_feature_engineer,
    )


# ── TestModelLoading ──────────────────────────────────────────────────────────

class TestModelLoading:
    def test_get_model_predictor_loads_once(
        self, artifact_path: Path
    ) -> None:
        p1 = _get_model_predictor(artifact_path)
        p2 = _get_model_predictor(artifact_path)
        assert p1 is p2

    def test_get_model_predictor_raises_on_missing(
        self, tmp_path: Path
    ) -> None:
        from src.model.model_io import ModelNotFoundError  # noqa: PLC0415

        reset_model_cache()
        with pytest.raises(ModelNotFoundError):
            _get_model_predictor(tmp_path / "missing.joblib")


# ── TestRealtimePipeline ──────────────────────────────────────────────────────

class TestRealtimePipeline:
    def test_predict_returns_structured_result(
        self, pipeline: RealtimePipeline
    ) -> None:
        result = pipeline.predict("AAPL", save=False)

        assert result.ticker == "AAPL"
        assert result.headline == "Apple announces new AI features"
        assert result.prediction in DIRECTION_CLASSES
        assert result.confidence > 0
        assert set(result.probabilities.keys()) == set(DIRECTION_CLASSES)
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
        assert result.sentiment_label == "positive"
        assert result.sentiment_score == 1
        assert result.sentiment_confidence == pytest.approx(0.82)
        assert isinstance(result.feature_values, dict)

    def test_predict_calls_pipeline_stages_in_order(
        self,
        pipeline: RealtimePipeline,
        mock_finnhub: MagicMock,
        mock_sentiment: MagicMock,
        mock_feature_engineer: MagicMock,
    ) -> None:
        pipeline.predict("AAPL", save=False)

        mock_finnhub.fetch_latest_news.assert_called_once_with("AAPL")
        mock_sentiment.analyse_texts.assert_called_once()
        mock_feature_engineer.load_data.assert_called_once()
        mock_feature_engineer.generate_features.assert_called_once()

    def test_predict_raises_no_news_error(
        self,
        pipeline: RealtimePipeline,
        mock_finnhub: MagicMock,
    ) -> None:
        mock_finnhub.fetch_latest_news.return_value = []

        with pytest.raises(NoNewsError, match="AAPL"):
            pipeline.predict("AAPL", save=False)

    def test_predict_saves_to_database(
        self,
        pipeline: RealtimePipeline,
        mock_settings: MagicMock,
    ) -> None:
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(engine)

        with patch.object(pipeline, "_db") as mock_db:
            from src.storage.database import DatabaseManager  # noqa: PLC0415

            real_db = DatabaseManager("sqlite:///:memory:")
            real_db.create_tables()
            mock_db.get_session = real_db.get_session

            result = pipeline.predict("AAPL", save=True)

        assert result.saved_id is not None
        assert result.saved_id > 0

    def test_predict_without_database_url_raises(
        self, mock_settings: MagicMock, artifact_path: Path
    ) -> None:
        mock_settings.database_url = None
        with pytest.raises(RealtimePipelineError, match="DATABASE_URL"):
            RealtimePipeline(model_path=artifact_path)

    def test_probabilities_match_confidence_of_predicted_class(
        self, pipeline: RealtimePipeline
    ) -> None:
        result = pipeline.predict("AAPL", save=False)
        assert result.confidence == pytest.approx(
            result.probabilities[result.prediction], abs=1e-6
        )
