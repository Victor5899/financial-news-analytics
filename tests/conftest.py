"""
Shared pytest fixtures for the financial-news-analytics test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_settings() -> MagicMock:  # type: ignore[return]
    """
    Patch ``settings`` in the feature_engineer module for tests that need to
    control database URL resolution without touching the real environment.
    """
    with patch("src.features.feature_engineer.settings") as mock:
        yield mock
