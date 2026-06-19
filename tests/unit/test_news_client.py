"""
Unit tests for src/ingestion/news_client.py (Finnhub backend).

All HTTP calls are mocked — these tests run without a real Finnhub key
or internet connection. Each test class covers one logical concern:

  TestParseArticle        — _parse_article helper (Finnhub field mapping)
  TestCheckResponseErrors — HTTP error detection (Finnhub status codes)
  TestFetchArticles       — full fetch_articles() path (single request, no pagination)
  TestFetchAllTickers     — multi-ticker orchestration
  TestSummariseResults    — summary DataFrame builder

Key Finnhub differences from the previous NewsAPI implementation:
  - Response body is a JSON array, not a {status, articles} envelope
  - "source" is a plain string, not {id, name}
  - "headline" maps to title; "summary" maps to description
  - "datetime" is a Unix timestamp (int seconds), not an ISO-8601 string
  - "id" (int) is the article identifier → stored in source_id
  - No author or content fields
  - No pagination — one request covers the full date window
  - 403 is a valid error code (premium-plan gate)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.ingestion.news_client import (
    ARTICLE_COLUMNS,
    FinnhubAuthError,
    FinnhubNetworkError,
    FinnhubRateLimitError,
    FinnhubRequestError,
    _check_response_errors,
    _parse_article,
    fetch_all_tickers,
    fetch_articles,
    summarise_results,
)

UTC = timezone.utc

# ── Test constants ────────────────────────────────────────────────────────────

# Sentinel API key used throughout this test module.  It is intentionally
# different from any real key so that if the patch below is accidentally
# bypassed the assertion will still fail clearly rather than hitting Finnhub.
_TEST_FINNHUB_KEY = "test_finnhub_key_32chars_xxxxxxxx"


# ── Module-level settings patch ───────────────────────────────────────────────
#
# Root-cause: src.utils.config.settings is a Pydantic singleton created at
# *import time* from the project's .env file.  news_client.fetch_articles()
# reads settings.finnhub_api_key at *call time*, so any real key present in
# .env leaks straight into the request params.
#
# Fix: replace the `settings` name inside the news_client module's namespace
# for every test in this file.  monkeypatch ensures the original is restored
# after each test, so no other test module is affected.

@pytest.fixture(autouse=True)
def _patch_news_client_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cfg = MagicMock()
    mock_cfg.finnhub_api_key  = _TEST_FINNHUB_KEY
    mock_cfg.news_lookback_days = 7  # used by fetch_articles for default from_date
    monkeypatch.setattr("src.ingestion.news_client.settings", mock_cfg)


# ── Shared test data ──────────────────────────────────────────────────────────

# Unix timestamp for 2024-01-10 09:00:00 UTC
_TS_JAN10 = 1704877200
_TS_JAN11 = 1704963600
_TS_JAN12 = 1705050000


def _make_raw_article(
    headline: str = "Apple Q4 Earnings Beat Expectations",
    url: str = "https://example.com/apple-earnings",
    unix_ts: int = _TS_JAN10,
    source: str = "Reuters",
    article_id: int = 100001,
    summary: str = "Apple reported strong Q4 earnings...",
) -> dict[str, Any]:
    """Return a Finnhub-shaped article dict."""
    return {
        "category": "company news",
        "datetime": unix_ts,
        "headline": headline,
        "id":       article_id,
        "image":    "https://example.com/img.jpg",
        "related":  "AAPL",
        "source":   source,
        "summary":  summary,
        "url":      url,
    }


def _make_mock_response(
    body: Any,                           # list on success; dict on error
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock requests.Response that returns ``body`` from .json()."""
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.ok = (200 <= status_code < 300)
    mock.json.return_value = body
    mock.text = json.dumps(body)
    mock.headers = headers or {}
    return mock


# ── TestParseArticle ─────────────────────────────────────────────────────────

class TestParseArticle:
    def test_maps_headline_to_title(self):
        raw = _make_raw_article(headline="Nvidia smashes earnings")
        record = _parse_article(raw, "NVDA", datetime.now(UTC))
        assert record["title"] == "Nvidia smashes earnings"

    def test_maps_summary_to_description(self):
        raw = _make_raw_article(summary="NVDA beat analyst estimates by 30%")
        record = _parse_article(raw, "NVDA", datetime.now(UTC))
        assert record["description"] == "NVDA beat analyst estimates by 30%"

    def test_maps_source_string_to_source_name(self):
        raw = _make_raw_article(source="Bloomberg")
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["source_name"] == "Bloomberg"
        # source_id is the stringified article id, not the source name
        assert record["source_id"] == str(raw["id"])

    def test_maps_integer_id_to_source_id_as_string(self):
        raw = _make_raw_article(article_id=999888)
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["source_id"] == "999888"

    def test_author_is_always_none(self):
        record = _parse_article(_make_raw_article(), "AAPL", datetime.now(UTC))
        assert record["author"] is None

    def test_content_is_always_none(self):
        record = _parse_article(_make_raw_article(), "AAPL", datetime.now(UTC))
        assert record["content"] is None

    def test_unix_timestamp_stored_as_int_before_coercion(self):
        raw = _make_raw_article(unix_ts=_TS_JAN10)
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        # At this stage it is still the raw integer — _coerce_dtypes converts it
        assert record["published_at"] == _TS_JAN10

    def test_fetched_at_stored_as_iso_string(self):
        fetched_at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        record = _parse_article(_make_raw_article(), "AAPL", fetched_at)
        assert record["fetched_at"] == fetched_at.isoformat()

    def test_ticker_is_normalised_to_uppercase_by_caller(self):
        record = _parse_article(_make_raw_article(), "AAPL", datetime.now(UTC))
        assert record["ticker"] == "AAPL"

    def test_missing_id_yields_none_source_id(self):
        raw = _make_raw_article()
        del raw["id"]
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["source_id"] is None

    def test_missing_summary_yields_none_description(self):
        raw = _make_raw_article()
        del raw["summary"]
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["description"] is None


# ── TestCheckResponseErrors ──────────────────────────────────────────────────

class TestCheckResponseErrors:
    def test_401_raises_auth_error(self):
        mock = _make_mock_response({"error": "Invalid API key."}, status_code=401)
        mock.ok = False
        with pytest.raises(FinnhubAuthError, match="Authentication failed"):
            _check_response_errors(mock, "AAPL")

    def test_403_raises_auth_error_for_premium_endpoint(self):
        mock = _make_mock_response(
            {"error": "You don't have access to this resource."},
            status_code=403,
        )
        mock.ok = False
        with pytest.raises(FinnhubAuthError, match="premium"):
            _check_response_errors(mock, "AAPL")

    def test_429_raises_rate_limit_error_with_retry_after(self):
        mock = _make_mock_response({}, status_code=429)
        mock.ok = False
        mock.headers = {"Retry-After": "60"}
        with pytest.raises(FinnhubRateLimitError, match="Retry-After: 60s"):
            _check_response_errors(mock, "AAPL")

    def test_429_without_retry_after_header_still_raises(self):
        mock = _make_mock_response({}, status_code=429)
        mock.ok = False
        mock.headers = {}
        with pytest.raises(FinnhubRateLimitError):
            _check_response_errors(mock, "AAPL")

    def test_500_raises_request_error(self):
        mock = _make_mock_response({"error": "internal error"}, status_code=500)
        mock.ok = False
        with pytest.raises(FinnhubRequestError, match="500"):
            _check_response_errors(mock, "AAPL")

    def test_non_json_error_body_still_raises_request_error(self):
        mock = MagicMock(spec=requests.Response)
        mock.status_code = 503
        mock.ok = False
        mock.json.side_effect = ValueError("not json")
        mock.text = "Service Unavailable"
        mock.headers = {}
        with pytest.raises(FinnhubRequestError):
            _check_response_errors(mock, "AAPL")

    def test_200_with_list_body_passes_silently(self):
        mock = _make_mock_response([_make_raw_article()])
        mock.ok = True
        _check_response_errors(mock, "AAPL")   # no exception raised

    def test_200_with_empty_list_passes_silently(self):
        mock = _make_mock_response([])
        mock.ok = True
        _check_response_errors(mock, "AAPL")   # no exception; empty = no articles


# ── TestFetchArticles ────────────────────────────────────────────────────────

class TestFetchArticles:
    def _mock_session(self, body: Any, status_code: int = 200) -> MagicMock:
        session = MagicMock()
        session.get.return_value = _make_mock_response(body, status_code)
        return session

    # ── Schema & basic correctness ─────────────────────────────────────────

    def test_returns_dataframe_with_correct_schema(self):
        session = self._mock_session([_make_raw_article()])
        df = fetch_articles("AAPL", session=session)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ARTICLE_COLUMNS

    def test_all_rows_have_the_requested_ticker(self):
        articles = [_make_raw_article(url=f"https://x.com/{i}") for i in range(3)]
        session = self._mock_session(articles)
        df = fetch_articles("AAPL", session=session)
        assert (df["ticker"] == "AAPL").all()

    def test_ticker_input_normalised_to_uppercase(self):
        session = self._mock_session([_make_raw_article()])
        df = fetch_articles("aapl", session=session)
        assert (df["ticker"] == "AAPL").all()

    # ── Timestamp coercion ────────────────────────────────────────────────

    def test_published_at_is_utc_aware_datetime(self):
        session = self._mock_session([_make_raw_article(unix_ts=_TS_JAN10)])
        df = fetch_articles("AAPL", session=session)
        assert pd.api.types.is_datetime64_any_dtype(df["published_at"])
        assert df["published_at"].dt.tz is not None

    def test_published_at_correctly_converts_unix_timestamp(self):
        session = self._mock_session([_make_raw_article(unix_ts=_TS_JAN10)])
        df = fetch_articles("AAPL", session=session)
        expected = pd.Timestamp(_TS_JAN10, unit="s", tz="UTC")
        assert df["published_at"].iloc[0] == expected

    # ── Dedup & sort ──────────────────────────────────────────────────────

    def test_deduplicates_by_url(self):
        art = _make_raw_article(url="https://duplicate.com")
        session = self._mock_session([art, art, art])
        df = fetch_articles("AAPL", session=session)
        assert len(df) == 1

    def test_sorted_descending_by_published_at(self):
        articles = [
            _make_raw_article(url="https://x.com/1", unix_ts=_TS_JAN10),
            _make_raw_article(url="https://x.com/2", unix_ts=_TS_JAN12),
            _make_raw_article(url="https://x.com/3", unix_ts=_TS_JAN11),
        ]
        session = self._mock_session(articles)
        df = fetch_articles("AAPL", session=session)
        dates = df["published_at"].tolist()
        assert dates == sorted(dates, reverse=True)

    # ── Empty / no-results ────────────────────────────────────────────────

    def test_returns_empty_dataframe_for_empty_api_response(self):
        session = self._mock_session([])
        df = fetch_articles("AAPL", session=session)
        assert df.empty
        assert list(df.columns) == ARTICLE_COLUMNS

    # ── Single request — no pagination ────────────────────────────────────

    def test_makes_exactly_one_api_call_per_ticker(self):
        session = self._mock_session([_make_raw_article()])
        fetch_articles("AAPL", session=session)
        assert session.get.call_count == 1

    def test_max_pages_param_does_not_trigger_extra_requests(self):
        session = self._mock_session([_make_raw_article()])
        fetch_articles("AAPL", session=session, max_pages=10)
        assert session.get.call_count == 1

    # ── Request parameters ────────────────────────────────────────────────

    def test_uses_symbol_param_for_ticker(self):
        session = self._mock_session([_make_raw_article()])
        fetch_articles("TSLA", session=session)
        params = session.get.call_args[1]["params"]
        assert params["symbol"] == "TSLA"

    def test_uses_token_param_for_auth(self):
        session = self._mock_session([_make_raw_article()])
        fetch_articles("AAPL", session=session)
        params = session.get.call_args[1]["params"]
        assert params["token"] == _TEST_FINNHUB_KEY

    def test_date_params_are_formatted_as_yyyy_mm_dd(self):
        session = self._mock_session([_make_raw_article()])
        fetch_articles("AAPL", session=session)
        params = session.get.call_args[1]["params"]
        # Verify the format, not the exact value (which depends on "now")
        assert len(params["from"]) == 10 and params["from"][4] == "-"
        assert len(params["to"]) == 10 and params["to"][4] == "-"

    def test_calls_company_news_endpoint(self):
        session = self._mock_session([_make_raw_article()])
        fetch_articles("AAPL", session=session)
        url_called = session.get.call_args[0][0]
        assert url_called.endswith("/company-news")

    # ── Error propagation ─────────────────────────────────────────────────

    def test_propagates_auth_error_on_401(self):
        session = self._mock_session({"error": "Invalid key"}, status_code=401)
        session.get.return_value.ok = False
        with pytest.raises(FinnhubAuthError):
            fetch_articles("AAPL", session=session)

    def test_propagates_auth_error_on_403(self):
        session = self._mock_session({"error": "Premium only"}, status_code=403)
        session.get.return_value.ok = False
        with pytest.raises(FinnhubAuthError, match="premium"):
            fetch_articles("AAPL", session=session)

    def test_propagates_rate_limit_error_on_429(self):
        mock = _make_mock_response({}, status_code=429)
        mock.ok = False
        mock.headers = {}
        session = MagicMock()
        session.get.return_value = mock
        with pytest.raises(FinnhubRateLimitError):
            fetch_articles("AAPL", session=session)

    def test_raises_network_error_on_connection_failure(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(FinnhubNetworkError, match="Connection failed"):
            fetch_articles("AAPL", session=session)

    def test_raises_network_error_on_timeout(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(FinnhubNetworkError, match="timed out"):
            fetch_articles("AAPL", session=session)

    def test_raises_network_error_on_generic_request_exception(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.RequestException("boom")
        with pytest.raises(FinnhubNetworkError):
            fetch_articles("AAPL", session=session)


# ── TestFetchAllTickers ──────────────────────────────────────────────────────

class TestFetchAllTickers:
    @patch("src.ingestion.news_client.fetch_articles")
    def test_returns_dataframe_per_ticker(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(columns=ARTICLE_COLUMNS)
        results = fetch_all_tickers(tickers=["AAPL", "TSLA"])
        assert set(results.keys()) == {"AAPL", "TSLA"}
        assert mock_fetch.call_count == 2

    @patch("src.ingestion.news_client.fetch_articles")
    def test_auth_error_aborts_all_remaining_tickers(self, mock_fetch):
        mock_fetch.side_effect = FinnhubAuthError("bad key")
        with pytest.raises(FinnhubAuthError):
            fetch_all_tickers(tickers=["AAPL", "TSLA", "NVDA"])
        assert mock_fetch.call_count == 1   # TSLA and NVDA never called

    @patch("src.ingestion.news_client.fetch_articles")
    def test_rate_limit_stops_remaining_tickers(self, mock_fetch):
        mock_fetch.side_effect = [
            pd.DataFrame(columns=ARTICLE_COLUMNS),
            FinnhubRateLimitError("quota"),
        ]
        results = fetch_all_tickers(tickers=["AAPL", "TSLA", "NVDA"])
        assert mock_fetch.call_count == 2   # NVDA never called
        assert results["TSLA"].empty

    @patch("src.ingestion.news_client.fetch_articles")
    def test_network_error_skips_ticker_and_continues(self, mock_fetch):
        good_df = pd.DataFrame({"ticker": ["TSLA"]})
        mock_fetch.side_effect = [
            FinnhubNetworkError("timeout"),
            good_df,
        ]
        results = fetch_all_tickers(tickers=["AAPL", "TSLA"])
        assert results["AAPL"].empty
        assert not results["TSLA"].empty

    @patch("src.ingestion.news_client.fetch_articles")
    def test_request_error_skips_ticker_and_continues(self, mock_fetch):
        mock_fetch.side_effect = [
            FinnhubRequestError("bad request"),
            pd.DataFrame({"ticker": ["TSLA"]}),
        ]
        results = fetch_all_tickers(tickers=["AAPL", "TSLA"])
        assert results["AAPL"].empty

    @patch("src.ingestion.news_client.fetch_articles")
    def test_max_pages_param_is_accepted_without_error(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(columns=ARTICLE_COLUMNS)
        # max_pages should be silently accepted (Finnhub ignores it)
        fetch_all_tickers(tickers=["AAPL"], max_pages=5)
        assert mock_fetch.call_count == 1


# ── TestSummariseResults ─────────────────────────────────────────────────────

class TestSummariseResults:
    def test_ok_ticker_row_counts_articles_and_sources(self):
        df = pd.DataFrame({
            "ticker":       ["AAPL", "AAPL"],
            "published_at": pd.to_datetime([_TS_JAN10, _TS_JAN11], unit="s", utc=True),
            "source_name":  ["Bloomberg", "Reuters"],
        })
        summary = summarise_results({"AAPL": df})
        row = summary[summary["ticker"] == "AAPL"].iloc[0]
        assert row["article_count"] == 2
        assert row["unique_sources"] == 2
        assert row["status"] == "ok"

    def test_empty_ticker_shows_zeros_and_empty_status(self):
        summary = summarise_results({"TSLA": pd.DataFrame(columns=ARTICLE_COLUMNS)})
        row = summary[summary["ticker"] == "TSLA"].iloc[0]
        assert row["article_count"] == 0
        assert row["unique_sources"] == 0
        assert row["status"] == "empty"

    def test_mixed_results_contain_both_statuses(self):
        good_df = pd.DataFrame({
            "ticker":       ["NVDA"],
            "published_at": pd.to_datetime([_TS_JAN10], unit="s", utc=True),
            "source_name":  ["CNBC"],
        })
        summary = summarise_results({
            "NVDA": good_df,
            "MSFT": pd.DataFrame(columns=ARTICLE_COLUMNS),
        })
        assert len(summary) == 2
        assert set(summary["status"]) == {"ok", "empty"}

    def test_earliest_and_latest_reflect_date_range(self):
        df = pd.DataFrame({
            "ticker":       ["AAPL", "AAPL", "AAPL"],
            "published_at": pd.to_datetime(
                [_TS_JAN10, _TS_JAN11, _TS_JAN12], unit="s", utc=True
            ),
            "source_name":  ["src", "src", "src"],
        })
        summary = summarise_results({"AAPL": df})
        row = summary.iloc[0]
        assert row["earliest"] < row["latest"]
