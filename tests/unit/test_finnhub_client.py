"""
Unit tests for src.realtime.finnhub_client — FinnhubClient.

All HTTP calls are mocked — no real API key or internet required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.realtime.finnhub_client import (
    ARTICLE_KEYS,
    FinnhubAuthError,
    FinnhubClient,
    FinnhubConfigError,
    FinnhubNetworkError,
    FinnhubRateLimitError,
    FinnhubRequestError,
    _parse_article,
    _resolve_api_key,
)

UTC = timezone.utc
_TEST_KEY = "test_finnhub_key_32chars_xxxxxxxx"
_TS = 1704877200  # 2024-01-10 09:00:00 UTC


def _make_raw_article(
    headline: str = "Apple announces new AI features",
    url: str = "https://example.com/apple-ai",
    unix_ts: int = _TS,
    article_id: int = 100001,
) -> dict[str, Any]:
    return {
        "category": "company news",
        "datetime": unix_ts,
        "headline": headline,
        "id":       article_id,
        "image":    "https://example.com/img.jpg",
        "related":  "AAPL",
        "source":   "Reuters",
        "summary":  "Apple unveiled new AI capabilities...",
        "url":      url,
    }


def _make_mock_response(
    body: Any,
    status_code: int = 200,
) -> MagicMock:
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.ok = 200 <= status_code < 300
    mock.json.return_value = body
    mock.text = json.dumps(body)
    mock.headers = {}
    return mock


# ── TestResolveApiKey ─────────────────────────────────────────────────────────

class TestResolveApiKey:
    def test_explicit_key_is_returned(self) -> None:
        assert _resolve_api_key("my_key") == "my_key"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cfg = MagicMock()
        mock_cfg.finnhub_api_key = ""
        monkeypatch.setattr("src.realtime.finnhub_client.settings", mock_cfg)
        with pytest.raises(FinnhubConfigError, match="FINNHUB_API_KEY"):
            _resolve_api_key(None)

    def test_placeholder_key_raises(self) -> None:
        with pytest.raises(FinnhubConfigError, match="FINNHUB_API_KEY"):
            _resolve_api_key("your_finnhub_api_key_here")


# ── TestParseArticle ──────────────────────────────────────────────────────────

class TestParseArticle:
    def test_maps_finnhub_fields(self) -> None:
        fetched = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        result = _parse_article(_make_raw_article(), "AAPL", fetched)

        assert result["ticker"] == "AAPL"
        assert result["source_id"] == "100001"
        assert result["source_name"] == "Reuters"
        assert result["title"] == "Apple announces new AI features"
        assert result["description"] == "Apple unveiled new AI capabilities..."
        assert result["url"] == "https://example.com/apple-ai"
        assert result["author"] is None
        assert result["content"] is None
        assert result["fetched_at"] == fetched
        assert result["published_at"] == datetime.fromtimestamp(_TS, tz=UTC)

    def test_all_article_keys_present(self) -> None:
        fetched = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        result = _parse_article(_make_raw_article(), "AAPL", fetched)
        for key in ARTICLE_KEYS:
            assert key in result


# ── TestFinnhubClient ─────────────────────────────────────────────────────────

class TestFinnhubClient:
    @pytest.fixture
    def client(self) -> FinnhubClient:
        return FinnhubClient(api_key=_TEST_KEY)

    def test_fetch_latest_news_returns_normalized_dicts(
        self, client: FinnhubClient
    ) -> None:
        articles = [_make_raw_article(), _make_raw_article(
            headline="Second article",
            url="https://example.com/second",
            unix_ts=_TS - 3600,
            article_id=100002,
        )]
        mock_resp = _make_mock_response(articles)

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.fetch_latest_news("AAPL")

        assert len(result) == 2
        assert all(key in result[0] for key in ARTICLE_KEYS)
        assert result[0]["title"] == "Apple announces new AI features"
        assert result[0]["ticker"] == "AAPL"

    def test_fetch_latest_news_sorted_newest_first(
        self, client: FinnhubClient
    ) -> None:
        older = _make_raw_article(
            headline="Older", url="https://example.com/old", unix_ts=_TS - 7200
        )
        newer = _make_raw_article(
            headline="Newer", url="https://example.com/new", unix_ts=_TS
        )
        mock_resp = _make_mock_response([older, newer])

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.fetch_latest_news("AAPL")

        assert result[0]["title"] == "Newer"
        assert result[1]["title"] == "Older"

    def test_fetch_latest_news_deduplicates_by_url(
        self, client: FinnhubClient
    ) -> None:
        dup = _make_raw_article()
        mock_resp = _make_mock_response([dup, dup])

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.fetch_latest_news("AAPL")

        assert len(result) == 1

    def test_fetch_latest_news_empty_response(
        self, client: FinnhubClient
    ) -> None:
        mock_resp = _make_mock_response([])

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.fetch_latest_news("AAPL")

        assert result == []

    def test_fetch_latest_news_auth_error(
        self, client: FinnhubClient
    ) -> None:
        mock_resp = _make_mock_response({"error": "Invalid token"}, status_code=401)

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(FinnhubAuthError):
                client.fetch_latest_news("AAPL")

    def test_fetch_latest_news_rate_limit_error(
        self, client: FinnhubClient
    ) -> None:
        mock_resp = _make_mock_response({"error": "limit"}, status_code=429)
        mock_resp.headers = {"Retry-After": "60"}

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(FinnhubRateLimitError):
                client.fetch_latest_news("AAPL")

    def test_fetch_latest_news_request_error(
        self, client: FinnhubClient
    ) -> None:
        mock_resp = _make_mock_response({"error": "bad request"}, status_code=400)

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(FinnhubRequestError):
                client.fetch_latest_news("AAPL")

    def test_fetch_latest_news_network_error(
        self, client: FinnhubClient
    ) -> None:
        with patch.object(
            client._session,
            "get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(FinnhubNetworkError):
                client.fetch_latest_news("AAPL")

    def test_ticker_is_uppercased(self, client: FinnhubClient) -> None:
        mock_resp = _make_mock_response([_make_raw_article()])

        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            client.fetch_latest_news("aapl")
            params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
            assert params["symbol"] == "AAPL"
