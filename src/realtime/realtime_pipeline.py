"""
Phase 8: Real-time prediction pipeline.

Orchestrates live Finnhub news ingestion, FinBERT sentiment analysis,
feature engineering, XGBoost inference, and PostgreSQL persistence.

Usage
-----
    from src.realtime.realtime_pipeline import RealtimePipeline

    pipeline = RealtimePipeline()
    result = pipeline.predict("AAPL")
    print(result.prediction, result.confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.feature_engineer import (
    FeatureEngineer,
    FeatureGenerationError,
)
from src.model.predictor import ModelPredictor
from src.processing.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.realtime.finnhub_client import FinnhubClient, FinnhubError
from src.realtime.prediction_repository import PredictionRepository
from src.storage.database import DatabaseManager
from src.utils.config import settings
from src.utils.logger import get_logger

UTC = timezone.utc
logger = get_logger(__name__)

# Match FeatureEngineer price lookback for technical indicators.
_PRICE_LOOKBACK_DAYS = 90

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts"
    / "models"
    / "xgboost_direction_model.joblib"
)

# Module-level model cache — loaded once and reused across predictions.
_model_predictor: ModelPredictor | None = None


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Structured output from a live prediction run."""

    ticker: str
    headline: str
    published_at: datetime | None
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    sentiment_score: int
    sentiment_label: str
    sentiment_confidence: float = 0.0
    feature_values: dict[str, float] = field(default_factory=dict)
    saved_id: int | None = None


# ── Exceptions ────────────────────────────────────────────────────────────────

class RealtimePipelineError(Exception):
    """Base exception for real-time pipeline errors."""


class NoNewsError(RealtimePipelineError):
    """No articles were returned from Finnhub for the requested ticker."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model_predictor(model_path: Path | None = None) -> ModelPredictor:
    """Return a cached :class:`ModelPredictor`, loading it on first call."""
    global _model_predictor  # noqa: PLW0603
    path = model_path or _DEFAULT_MODEL_PATH
    if _model_predictor is None or _model_predictor._model_path != path:
        logger.info("Loading model …")
        _model_predictor = ModelPredictor(model_path=path)
        _model_predictor.load_model()
    return _model_predictor


def _build_sentiment_text(article: dict[str, Any]) -> str | None:
    """Combine title and description for FinBERT input."""
    title = str(article.get("title") or "").strip()
    desc = str(article.get("description") or "").strip()
    if title and desc:
        return f"{title}. {desc}"
    return title or desc or None


def _article_to_feature_row(
    article: dict[str, Any],
    sentiment: dict[str, Any],
) -> dict[str, Any]:
    """Convert a live article + sentiment into a FeatureEngineer-compatible row."""
    published_at = article.get("published_at")
    if isinstance(published_at, str):
        published_at = pd.to_datetime(published_at, utc=True).to_pydatetime()

    pub_date = published_at.date() if published_at else datetime.now(UTC).date()

    return {
        "ticker":               article["ticker"],
        "source_name":          article.get("source_name"),
        "published_at":         published_at,
        "date":                 pub_date,
        "sentiment_label":      sentiment["sentiment_label"],
        "sentiment_score":      sentiment["sentiment_score"],
        "sentiment_confidence": sentiment["sentiment_confidence"],
    }


def _merge_historical_and_live(
    historical_df: pd.DataFrame,
    live_row: dict[str, Any],
) -> pd.DataFrame:
    """
    Merge historical DB articles with the live article, deduplicating by
    (ticker, published_at, title) when possible.
    """
    live_df = pd.DataFrame([live_row])
    if historical_df.empty:
        return live_df

    combined = pd.concat([historical_df, live_df], ignore_index=True)

    # Drop exact duplicates on ticker + published_at + title.
    subset = ["ticker", "published_at", "title"] if "title" in combined.columns else None
    if subset and all(c in combined.columns for c in subset):
        combined = combined.drop_duplicates(subset=subset, keep="last")

    return combined


def _extract_feature_values(
    features_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Build a feature-name → value dict from the first feature row."""
    if features_df.empty:
        return {}
    row = features_df.iloc[0]
    values: dict[str, float] = {}
    for col in feature_columns:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            values[col] = float(val)
    return values


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RealtimePipeline:
    """
    End-to-end live inference pipeline.

    Parameters
    ----------
    database_url : str | None
        SQLAlchemy connection URL.  Defaults to ``settings.database_url``.
    model_path : Path | None
        Path to the trained XGBoost artifact.  Defaults to
        ``artifacts/models/xgboost_direction_model.joblib``.
    finnhub_client : FinnhubClient | None
        Optional client override (useful for testing).
    sentiment_analyzer : FinBERTSentimentAnalyzer | None
        Optional sentiment analyzer override (useful for testing).
    feature_engineer : FeatureEngineer | None
        Optional feature engineer override (useful for testing).
    lookback_days : int
        Historical article window for rolling features.  Default: ``7``.
    """

    def __init__(
        self,
        database_url: str | None = None,
        model_path: Path | None = None,
        *,
        finnhub_client: FinnhubClient | None = None,
        sentiment_analyzer: FinBERTSentimentAnalyzer | None = None,
        feature_engineer: FeatureEngineer | None = None,
        lookback_days: int = 7,
    ) -> None:
        resolved_db = database_url or settings.database_url
        if not resolved_db:
            raise RealtimePipelineError(
                "No database URL configured. "
                "Set DATABASE_URL in your .env file for prediction storage."
            )

        self._database_url = resolved_db
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._finnhub = finnhub_client or FinnhubClient()
        self._sentiment = sentiment_analyzer or FinBERTSentimentAnalyzer(
            model_name=settings.finbert_model,
            batch_size=settings.finbert_batch_size,
            device=settings.finbert_device,
        )
        self._feature_engineer = feature_engineer or FeatureEngineer(resolved_db)
        self._lookback_days = lookback_days
        self._db = DatabaseManager(resolved_db)

    def predict(
        self,
        ticker: str,
        *,
        save: bool = True,
    ) -> PredictionResult:
        """
        Run the full live prediction pipeline for *ticker*.

        Steps
        -----
        1. Fetch latest news from Finnhub
        2. Run FinBERT sentiment on the newest article
        3. Build feature vector (historical DB data + live article)
        4. Load trained XGBoost model (cached after first call)
        5. Predict BUY / HOLD / SELL with probabilities
        6. Save prediction to PostgreSQL (when ``save=True``)
        7. Return structured :class:`PredictionResult`

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. ``"AAPL"``).
        save : bool
            When ``True`` (default), persist the prediction to PostgreSQL.

        Returns
        -------
        PredictionResult

        Raises
        ------
        NoNewsError
            When Finnhub returns no articles.
        RealtimePipelineError
            On feature generation or database failures.
        FinnhubError
            On Finnhub API errors.
        """
        ticker = ticker.upper()

        # 1. Fetch latest news
        logger.info("Fetching news …")
        try:
            articles = self._finnhub.fetch_latest_news(ticker)
        except FinnhubError:
            raise

        if not articles:
            raise NoNewsError(f"No news articles found for {ticker}")

        article = articles[0]
        headline = article.get("title") or ""
        published_at = article.get("published_at")
        target_date: date = (
            published_at.date()
            if isinstance(published_at, datetime)
            else datetime.now(UTC).date()
        )

        # 2. Run FinBERT sentiment
        logger.info("Running sentiment …")
        text = _build_sentiment_text(article)
        sentiment_results = self._sentiment.analyse_texts([text])
        sentiment = sentiment_results[0]

        # 3. Build feature vector
        logger.info("Generating features …")
        live_row = _article_to_feature_row(article, sentiment)

        try:
            historical_df = self._feature_engineer.load_data(
                tickers=[ticker],
                target_date=target_date,
                lookback_days=self._lookback_days,
            )
        except Exception as exc:  # noqa: BLE001
            raise RealtimePipelineError(
                f"Failed to load historical data: {exc}"
            ) from exc

        raw_df = _merge_historical_and_live(historical_df, live_row)

        # Ensure published_at is timezone-aware for time-window features.
        raw_df["published_at"] = pd.to_datetime(
            raw_df["published_at"], utc=True, errors="coerce"
        )
        if "date" not in raw_df.columns:
            raw_df["date"] = raw_df["published_at"].dt.date

        prices_df: pd.DataFrame | None = None
        try:
            price_start = target_date - timedelta(days=_PRICE_LOOKBACK_DAYS)
            loaded = self._feature_engineer.load_price_data(
                tickers=[ticker],
                start_date=price_start,
                end_date=target_date,
            )
            if not loaded.empty:
                prices_df = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not load price data ({exc}) — "
                "technical indicators will be None"
            )

        try:
            features_df = self._feature_engineer.generate_features(
                raw_df, target_date, prices_df=prices_df
            )
        except FeatureGenerationError as exc:
            raise RealtimePipelineError(
                f"Feature generation failed for {ticker} on {target_date}: {exc}"
            ) from exc

        # 4–5. Load model and predict
        logger.info("Loading model …")
        predictor = _get_model_predictor(self._model_path)

        logger.info("Predicting …")
        feature_columns = predictor.feature_columns
        feature_dict: dict[str, float] = {}
        for col in feature_columns:
            if col in features_df.columns and pd.notna(features_df.iloc[0][col]):
                feature_dict[col] = float(features_df.iloc[0][col])
            else:
                feature_dict[col] = float("nan")
        pred = predictor.predict_from_vector(feature_dict)
        predicted = pred["predicted_direction"]
        probabilities = pred["probabilities"]
        confidence = float(probabilities[predicted])

        result = PredictionResult(
            ticker=ticker,
            headline=headline,
            published_at=published_at if isinstance(published_at, datetime) else None,
            prediction=predicted,
            confidence=confidence,
            probabilities=probabilities,
            sentiment_score=int(sentiment["sentiment_score"]),
            sentiment_label=str(sentiment["sentiment_label"]),
            sentiment_confidence=float(sentiment["sentiment_confidence"]),
            feature_values=_extract_feature_values(features_df, feature_columns),
        )

        # 6. Save to PostgreSQL
        if save:
            logger.info("Saving prediction …")
            saved_id = self._save_prediction(result)
            result.saved_id = saved_id

        logger.info("Prediction complete.")
        return result

    def _save_prediction(self, result: PredictionResult) -> int:
        """Persist *result* and return the new row id."""
        record = {
            "ticker":           result.ticker,
            "prediction":       result.prediction,
            "confidence":       result.confidence,
            "buy_probability":  result.probabilities.get("BUY", 0.0),
            "hold_probability": result.probabilities.get("HOLD", 0.0),
            "sell_probability": result.probabilities.get("SELL", 0.0),
            "headline":         result.headline,
            "published_at":     result.published_at,
        }
        try:
            with self._db.get_session() as session:
                repo = PredictionRepository(session)
                saved = repo.save_prediction(record)
                return saved.id
        except Exception as exc:  # noqa: BLE001
            raise RealtimePipelineError(
                f"Failed to save prediction to database: {exc}"
            ) from exc


def reset_model_cache() -> None:
    """Clear the module-level model cache (for tests)."""
    global _model_predictor  # noqa: PLW0603
    _model_predictor = None
