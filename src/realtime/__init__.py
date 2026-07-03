"""
Phase 8: Real-time prediction engine.

Provides live Finnhub news ingestion, feature engineering, XGBoost inference,
and PostgreSQL persistence for dashboard-ready prediction results.
"""

from src.realtime.finnhub_client import FinnhubClient, FinnhubError
from src.realtime.prediction_repository import PredictionRepository
from src.realtime.realtime_pipeline import PredictionResult, RealtimePipeline

__all__ = [
    "FinnhubClient",
    "FinnhubError",
    "PredictionRepository",
    "PredictionResult",
    "RealtimePipeline",
]
