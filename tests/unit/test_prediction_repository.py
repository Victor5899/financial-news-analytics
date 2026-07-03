"""
Unit tests for src.realtime.prediction_repository — PredictionRepository.

Uses an in-memory SQLite database — no PostgreSQL required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.realtime.prediction_repository import (
    PredictionRepository,
    SavedPrediction,
    _coerce_prediction_record,
)
from src.storage.models import Base, Prediction

UTC = timezone.utc


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine):
    session = Session(sqlite_engine)
    yield session
    session.close()


def _sample_record(**overrides: object) -> dict:
    base = {
        "ticker":           "AAPL",
        "prediction":       "BUY",
        "confidence":       0.735,
        "buy_probability":  0.735,
        "hold_probability": 0.182,
        "sell_probability": 0.083,
        "headline":         "Apple announces new AI features",
        "published_at":     datetime(2024, 1, 10, 9, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return base


# ── TestCoercePredictionRecord ────────────────────────────────────────────────

class TestCoercePredictionRecord:
    def test_normalises_ticker_to_uppercase(self) -> None:
        result = _coerce_prediction_record(_sample_record(ticker="aapl"))
        assert result["ticker"] == "AAPL"

    def test_parses_iso_published_at(self) -> None:
        result = _coerce_prediction_record(
            _sample_record(published_at="2024-01-10T09:00:00+00:00")
        )
        assert result["published_at"] is not None
        assert result["published_at"].year == 2024


# ── TestPredictionRepository ──────────────────────────────────────────────────

class TestPredictionRepository:
    def test_save_prediction_returns_saved_prediction(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        saved = repo.save_prediction(_sample_record())

        assert isinstance(saved, SavedPrediction)
        assert saved.id > 0
        assert saved.ticker == "AAPL"
        assert saved.prediction == "BUY"
        assert saved.confidence == pytest.approx(0.735)

    def test_save_prediction_persists_row(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        saved = repo.save_prediction(_sample_record())
        db_session.commit()

        row = db_session.get(Prediction, saved.id)
        assert row is not None
        assert row.headline == "Apple announces new AI features"
        assert row.buy_probability == pytest.approx(0.735)
        assert row.hold_probability == pytest.approx(0.182)
        assert row.sell_probability == pytest.approx(0.083)

    def test_save_prediction_requires_ticker(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        with pytest.raises(ValueError, match="ticker"):
            repo.save_prediction(_sample_record(ticker=""))

    def test_save_prediction_requires_headline(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        with pytest.raises(ValueError, match="headline"):
            repo.save_prediction(_sample_record(headline=""))

    def test_get_latest_by_ticker(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        repo.save_prediction(_sample_record(prediction="BUY"))
        repo.save_prediction(_sample_record(prediction="SELL"))
        db_session.commit()

        latest = repo.get_latest_by_ticker("AAPL")
        assert latest is not None
        assert latest.prediction == "SELL"

    def test_count_predictions(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        repo.save_prediction(_sample_record(ticker="AAPL"))
        repo.save_prediction(_sample_record(ticker="TSLA"))
        db_session.commit()

        assert repo.count_predictions() == 2
        assert repo.count_predictions("AAPL") == 1

    def test_get_latest_by_ticker_returns_none_when_empty(
        self, db_session: Session
    ) -> None:
        repo = PredictionRepository(db_session)
        assert repo.get_latest_by_ticker("AAPL") is None
