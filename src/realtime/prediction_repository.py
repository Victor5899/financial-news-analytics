"""
Phase 8: PostgreSQL persistence for live predictions.

``PredictionRepository`` owns all SQL reads and writes for the ``predictions``
table.  It follows the same session-injection pattern as ``ArticleRepository``.

Usage
-----
    from src.realtime.prediction_repository import PredictionRepository

    with db.get_session() as session:
        repo = PredictionRepository(session)
        record = repo.save_prediction({...})
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.storage.models import Prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PREDICTION_FIELDS: tuple[str, ...] = (
    "ticker",
    "prediction",
    "confidence",
    "buy_probability",
    "hold_probability",
    "sell_probability",
    "headline",
    "published_at",
)


@dataclass(frozen=True)
class SavedPrediction:
    """Return value from :meth:`PredictionRepository.save_prediction`."""

    id: int
    ticker: str
    prediction: str
    confidence: float


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_datetime(value: Any) -> datetime | None:
    value = _nan_to_none(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _coerce_prediction_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw prediction dict to an insert-ready dict."""
    return {
        "ticker":           str(_nan_to_none(raw.get("ticker")) or "").upper(),
        "prediction":       str(_nan_to_none(raw.get("prediction")) or ""),
        "confidence":       float(_nan_to_none(raw.get("confidence")) or 0.0),
        "buy_probability":  float(_nan_to_none(raw.get("buy_probability")) or 0.0),
        "hold_probability": float(_nan_to_none(raw.get("hold_probability")) or 0.0),
        "sell_probability": float(_nan_to_none(raw.get("sell_probability")) or 0.0),
        "headline":         str(_nan_to_none(raw.get("headline")) or ""),
        "published_at":     _to_datetime(raw.get("published_at")),
    }


class PredictionRepository:
    """
    Data-access object for the ``predictions`` table.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.  The caller owns transaction management
        via ``DatabaseManager.get_session()``.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_prediction(self, record: dict[str, Any]) -> SavedPrediction:
        """
        Insert a new prediction row and return its saved identity.

        Parameters
        ----------
        record : dict
            Must contain all fields in :data:`_PREDICTION_FIELDS`.

        Returns
        -------
        SavedPrediction
            The inserted row's ``id``, ``ticker``, ``prediction``, and
            ``confidence``.

        Raises
        ------
        ValueError
            If required fields are missing or empty.
        """
        clean = _coerce_prediction_record(record)

        if not clean["ticker"]:
            raise ValueError("Prediction record must include a non-empty ticker")
        if not clean["prediction"]:
            raise ValueError("Prediction record must include a prediction label")
        if not clean["headline"]:
            raise ValueError("Prediction record must include a headline")

        logger.debug(
            f"Saving prediction for {clean['ticker']}: "
            f"{clean['prediction']} ({clean['confidence']:.1%})"
        )

        row = Prediction(**{k: clean[k] for k in _PREDICTION_FIELDS})
        self._session.add(row)
        self._session.flush()

        logger.info(
            f"Prediction saved (id={row.id}): "
            f"{clean['ticker']} → {clean['prediction']}"
        )

        return SavedPrediction(
            id=row.id,
            ticker=clean["ticker"],
            prediction=clean["prediction"],
            confidence=clean["confidence"],
        )

    def get_latest_by_ticker(self, ticker: str) -> Prediction | None:
        """Return the most recent prediction for *ticker*, or ``None``."""
        return (
            self._session.query(Prediction)
            .filter(Prediction.ticker == ticker.upper())
            .order_by(Prediction.created_at.desc())
            .first()
        )

    def count_predictions(self, ticker: str | None = None) -> int:
        """Return total prediction count, optionally filtered by ticker."""
        q = self._session.query(Prediction)
        if ticker:
            q = q.filter(Prediction.ticker == ticker.upper())
        return q.count()
