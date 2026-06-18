"""
Shared pytest fixtures for the financial-news-analytics test suite.

The most important fixture here is ``mock_settings``, which patches the
Pydantic settings singleton before any module-level code in config.py
tries to read from .env. This lets tests run in CI without real API keys.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Patch settings with safe test values for every test automatically."""
    fake = MagicMock()
    fake.finnhub_api_key = "test_finnhub_key_32chars_xxxxxxxx"
    fake.tickers = ["AAPL", "TSLA"]
    fake.news_lookback_days = 7
    fake.log_level = "DEBUG"
    fake.finbert_model = "ProsusAI/finbert"
    fake.finbert_batch_size = 32
    fake.finbert_device = "auto"
    fake.database_url = None

    with ExitStack() as stack:
        stack.enter_context(patch("src.ingestion.news_client.settings", fake))
        stack.enter_context(patch("src.features.feature_engineer.settings", fake))
        yield fake
