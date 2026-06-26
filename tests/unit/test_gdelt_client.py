"""
Unit tests for src/ingestion/gdelt_client.py.

All HTTP calls are mocked — these tests run without a live GDELT API
connection or any API key.

Test classes
------------
  TestBuildQuery          — ticker → company-name query string mapping
  TestParseSeendate       — GDELT seendate format parsing
  TestUrlToSourceId       — deterministic hash-based source_id generation
  TestParseArticle        — GDELT field → pipeline schema mapping
  TestFetchChunk          — single sub-request: HTTP mocking, error paths
  TestRetry429Behavior    — HTTP 429 exponential-backoff retry logic
  TestFetchArticles       — full fetch path: chunking, dedup, sort, errors
  TestFetchAllTickers     — multi-ticker orchestration and error handling
  TestSummariseResults    — summary DataFrame builder
  TestGDELTExceptions     — exception hierarchy validation
  TestDateChunks          — date window splitting logic
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.ingestion.gdelt_client import (
    _MAX_429_RETRIES,
    _RETRY_429_BASE_DELAY,
    GDELT_ARTICLE_COLUMNS,
    GDELT_MAX_RECORDS,
    TICKER_QUERY_MAP,
    GDELTAPIError,
    GDELTError,
    GDELTValidationError,
    _build_query,
    _coerce_dtypes,
    _date_chunks,
    _fetch_chunk,
    _parse_article,
    _parse_seendate,
    _request_with_429_retry,
    _url_to_source_id,
    fetch_all_tickers,
    fetch_articles,
    summarise_results,
)

UTC = timezone.utc

# ── Shared test data ──────────────────────────────────────────────────────────

_SEENDATE_JAN15 = "20250115T120000Z"
_SEENDATE_JAN16 = "20250116T080000Z"
_SEENDATE_JAN17 = "20250117T150000Z"

_DT_JAN15 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
_DT_JAN16 = datetime(2025, 1, 16, 8, 0, 0, tzinfo=UTC)
_DT_JAN17 = datetime(2025, 1, 17, 15, 0, 0, tzinfo=UTC)


def _make_gdelt_article(
    url: str = "https://reuters.com/technology/apple-earnings-2025",
    title: str = "Apple Inc Reports Strong Quarterly Results",
    seendate: str = _SEENDATE_JAN15,
    domain: str = "reuters.com",
) -> dict[str, Any]:
    """Return a GDELT-shaped article dict."""
    return {
        "url":           url,
        "url_mobile":    "",
        "title":         title,
        "seendate":      seendate,
        "socialimage":   "https://example.com/image.jpg",
        "domain":        domain,
        "language":      "English",
        "sourcecountry": "United States",
    }


def _make_gdelt_response(
    articles: list[dict[str, Any]] | None,
    status_code: int = 200,
) -> MagicMock:
    """Build a mock requests.Response returning a GDELT articles envelope."""
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.ok = 200 <= status_code < 300
    body = {"articles": articles}
    mock.json.return_value = body
    mock.text = json.dumps(body)
    return mock


def _make_mock_session(
    articles: list[dict[str, Any]] | None = None,
    status_code: int = 200,
) -> MagicMock:
    """Return a mock requests.Session whose .get() yields a GDELT response."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_gdelt_response(articles or [], status_code)
    return session


# ── TestBuildQuery ────────────────────────────────────────────────────────────

class TestBuildQuery:
    def test_aapl_maps_to_apple_inc(self) -> None:
        assert _build_query("AAPL") == '"Apple Inc"'

    def test_tsla_maps_to_tesla(self) -> None:
        assert _build_query("TSLA") == '"Tesla"'

    def test_nvda_maps_to_nvidia(self) -> None:
        assert _build_query("NVDA") == '"NVIDIA"'

    def test_msft_maps_to_microsoft(self) -> None:
        assert _build_query("MSFT") == '"Microsoft"'

    def test_amzn_maps_to_amazon(self) -> None:
        assert _build_query("AMZN") == '"Amazon"'

    def test_unknown_ticker_falls_back_to_symbol(self) -> None:
        assert _build_query("GOOGL") == "GOOGL"

    def test_lowercase_ticker_normalised(self) -> None:
        assert _build_query("aapl") == '"Apple Inc"'

    def test_all_mapped_tickers_use_quoted_phrases(self) -> None:
        for ticker, query in TICKER_QUERY_MAP.items():
            assert query.startswith('"'), f"{ticker} query should start with quote"
            assert query.endswith('"'), f"{ticker} query should end with quote"


# ── TestParseSeendate ─────────────────────────────────────────────────────────

class TestParseSeendate:
    def test_valid_seendate_returns_utc_datetime(self) -> None:
        result = _parse_seendate(_SEENDATE_JAN15)
        assert result == _DT_JAN15

    def test_result_is_utc_aware(self) -> None:
        result = _parse_seendate(_SEENDATE_JAN15)
        assert result is not None
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_none_input_returns_none(self) -> None:
        assert _parse_seendate(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_seendate("") is None

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_seendate("2025-01-15") is None
        assert _parse_seendate("not-a-date") is None
        assert _parse_seendate("20250115") is None

    def test_already_utc_datetime_returned_unchanged(self) -> None:
        dt = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        assert _parse_seendate(dt) == dt

    def test_naive_datetime_gets_utc_attached(self) -> None:
        naive = datetime(2025, 1, 15, 12, 0)
        result = _parse_seendate(naive)
        assert result is not None
        assert result.tzinfo == UTC

    def test_non_string_non_datetime_returns_none(self) -> None:
        assert _parse_seendate(12345) is None
        assert _parse_seendate([]) is None

    def test_another_valid_seendate(self) -> None:
        result = _parse_seendate(_SEENDATE_JAN16)
        assert result == _DT_JAN16


# ── TestUrlToSourceId ─────────────────────────────────────────────────────────

class TestUrlToSourceId:
    def test_returns_16_char_hex_string(self) -> None:
        sid = _url_to_source_id("https://example.com/article")
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)

    def test_same_url_yields_same_id(self) -> None:
        url = "https://reuters.com/apple-q1-2025"
        assert _url_to_source_id(url) == _url_to_source_id(url)

    def test_different_urls_yield_different_ids(self) -> None:
        id1 = _url_to_source_id("https://example.com/a")
        id2 = _url_to_source_id("https://example.com/b")
        assert id1 != id2

    def test_known_hash_is_deterministic(self) -> None:
        import hashlib
        url = "https://reuters.com/test"
        expected = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        assert _url_to_source_id(url) == expected


# ── TestParseArticle ──────────────────────────────────────────────────────────

class TestParseArticle:
    def test_maps_url_correctly(self) -> None:
        raw = _make_gdelt_article(url="https://reuters.com/aapl")
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["url"] == "https://reuters.com/aapl"

    def test_maps_title_correctly(self) -> None:
        raw = _make_gdelt_article(title="NVIDIA smashes earnings")
        record = _parse_article(raw, "NVDA", datetime.now(UTC))
        assert record["title"] == "NVIDIA smashes earnings"

    def test_maps_domain_to_source_name(self) -> None:
        raw = _make_gdelt_article(domain="bloomberg.com")
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["source_name"] == "bloomberg.com"

    def test_source_id_is_16_char_hex(self) -> None:
        raw = _make_gdelt_article(url="https://reuters.com/test")
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert len(record["source_id"]) == 16
        assert all(c in "0123456789abcdef" for c in record["source_id"])

    def test_source_id_derived_from_url(self) -> None:
        url = "https://reuters.com/unique-article"
        raw = _make_gdelt_article(url=url)
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["source_id"] == _url_to_source_id(url)

    def test_ticker_is_preserved(self) -> None:
        record = _parse_article(_make_gdelt_article(), "TSLA", datetime.now(UTC))
        assert record["ticker"] == "TSLA"

    def test_author_is_always_none(self) -> None:
        record = _parse_article(_make_gdelt_article(), "AAPL", datetime.now(UTC))
        assert record["author"] is None

    def test_description_is_always_none(self) -> None:
        record = _parse_article(_make_gdelt_article(), "AAPL", datetime.now(UTC))
        assert record["description"] is None

    def test_content_is_always_none(self) -> None:
        record = _parse_article(_make_gdelt_article(), "AAPL", datetime.now(UTC))
        assert record["content"] is None

    def test_published_at_stored_as_raw_seendate_string(self) -> None:
        raw = _make_gdelt_article(seendate=_SEENDATE_JAN15)
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["published_at"] == _SEENDATE_JAN15

    def test_fetched_at_stored_as_iso_string(self) -> None:
        fetched = datetime(2025, 1, 20, 10, 0, tzinfo=UTC)
        record = _parse_article(_make_gdelt_article(), "AAPL", fetched)
        assert record["fetched_at"] == fetched.isoformat()

    def test_empty_url_yields_empty_source_id(self) -> None:
        raw = _make_gdelt_article(url="")
        record = _parse_article(raw, "AAPL", datetime.now(UTC))
        assert record["source_id"] == ""


# ── TestFetchChunk ────────────────────────────────────────────────────────────

class TestFetchChunk:
    def _chunk_args(self) -> tuple[str, datetime, datetime, int]:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC)
        return '"Apple Inc"', start, end, 250

    def test_returns_articles_list_on_success(self) -> None:
        articles = [_make_gdelt_article()]
        session = _make_mock_session(articles)
        query, start, end, max_rec = self._chunk_args()
        result = _fetch_chunk(query, start, end, max_rec, session)
        assert result == articles

    def test_returns_empty_list_when_articles_key_is_null(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _make_gdelt_response(None)
        query, start, end, max_rec = self._chunk_args()
        result = _fetch_chunk(query, start, end, max_rec, session)
        assert result == []

    def test_returns_empty_list_when_no_articles(self) -> None:
        session = _make_mock_session([])
        query, start, end, max_rec = self._chunk_args()
        result = _fetch_chunk(query, start, end, max_rec, session)
        assert result == []

    def test_raises_api_error_on_non_200(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _make_gdelt_response([], status_code=500)
        query, start, end, max_rec = self._chunk_args()
        with pytest.raises(GDELTAPIError, match="500"):
            _fetch_chunk(query, start, end, max_rec, session)

    def test_raises_api_error_on_connection_failure(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.ConnectionError("refused")
        query, start, end, max_rec = self._chunk_args()
        with pytest.raises(GDELTAPIError, match="Connection"):
            _fetch_chunk(query, start, end, max_rec, session)

    def test_raises_api_error_on_timeout(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        query, start, end, max_rec = self._chunk_args()
        with pytest.raises(GDELTAPIError, match="timed out"):
            _fetch_chunk(query, start, end, max_rec, session)

    def test_raises_api_error_on_request_exception(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.RequestException("boom")
        query, start, end, max_rec = self._chunk_args()
        with pytest.raises(GDELTAPIError):
            _fetch_chunk(query, start, end, max_rec, session)

    def test_raises_validation_error_on_non_json_body(self) -> None:
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "not json"
        session = MagicMock(spec=requests.Session)
        session.get.return_value = mock_resp
        query, start, end, max_rec = self._chunk_args()
        with pytest.raises(GDELTValidationError):
            _fetch_chunk(query, start, end, max_rec, session)

    def test_raises_validation_error_when_response_is_list_not_dict(self) -> None:
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = []    # array instead of object
        mock_resp.text = "[]"
        session = MagicMock(spec=requests.Session)
        session.get.return_value = mock_resp
        query, start, end, max_rec = self._chunk_args()
        with pytest.raises(GDELTValidationError):
            _fetch_chunk(query, start, end, max_rec, session)

    def test_max_records_capped_at_gdelt_limit(self) -> None:
        session = _make_mock_session([])
        query, start, end, _ = self._chunk_args()
        _fetch_chunk(query, start, end, 9999, session)
        call_kwargs = session.get.call_args[1]
        assert call_kwargs["params"]["maxrecords"] == GDELT_MAX_RECORDS

    def test_request_uses_correct_date_format(self) -> None:
        session = _make_mock_session([])
        start = datetime(2025, 3, 5, 0, 0, 0, tzinfo=UTC)
        end = datetime(2025, 3, 19, 23, 59, 59, tzinfo=UTC)
        _fetch_chunk('"Apple Inc"', start, end, 250, session)
        params = session.get.call_args[1]["params"]
        assert params["startdatetime"] == "20250305000000"
        assert params["enddatetime"] == "20250319235959"

    def test_request_uses_artlist_mode(self) -> None:
        session = _make_mock_session([])
        query, start, end, max_rec = self._chunk_args()
        _fetch_chunk(query, start, end, max_rec, session)
        params = session.get.call_args[1]["params"]
        assert params["mode"] == "ArtList"
        assert params["format"] == "json"


# ── TestRetry429Behavior ──────────────────────────────────────────────────────

class TestRetry429Behavior:
    """Verify HTTP 429 exponential-backoff retry logic in _request_with_429_retry.

    All tests patch ``time.sleep`` so no real delays occur during the suite.
    The standard GDELT URL and a minimal params dict are used to drive the
    helper directly; _fetch_chunk is exercised separately via TestFetchChunk.
    """

    _URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    _PARAMS: dict[str, str] = {"query": '"Apple Inc"', "mode": "ArtList"}

    def _make_response(self, status_code: int) -> MagicMock:
        mock = MagicMock(spec=requests.Response)
        mock.status_code = status_code
        mock.ok = 200 <= status_code < 300
        mock.json.return_value = {"articles": []}
        mock.text = ""
        mock.headers = {}
        return mock

    def _make_429(self, retry_after: str | None = None) -> MagicMock:
        resp = self._make_response(429)
        resp.headers = {"Retry-After": retry_after} if retry_after else {}
        return resp

    # ── Success after retries ──────────────────────────────────────────────

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_success_on_first_attempt_no_sleep(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_response(200)
        resp = _request_with_429_retry(session, self._URL, self._PARAMS)
        assert resp.status_code == 200
        mock_sleep.assert_not_called()

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_success_after_one_429(self, mock_sleep: MagicMock) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [self._make_429(), self._make_response(200)]
        resp = _request_with_429_retry(session, self._URL, self._PARAMS)
        assert resp.status_code == 200
        assert session.get.call_count == 2

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_success_after_three_429s(self, mock_sleep: MagicMock) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            self._make_429(),
            self._make_429(),
            self._make_429(),
            self._make_response(200),
        ]
        resp = _request_with_429_retry(session, self._URL, self._PARAMS)
        assert resp.status_code == 200
        assert session.get.call_count == 4

    # ── Sleep delay schedule ───────────────────────────────────────────────

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_first_retry_sleeps_base_delay(self, mock_sleep: MagicMock) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [self._make_429(), self._make_response(200)]
        _request_with_429_retry(session, self._URL, self._PARAMS)
        mock_sleep.assert_called_once_with(_RETRY_429_BASE_DELAY)

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_second_retry_sleeps_double_base(self, mock_sleep: MagicMock) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            self._make_429(),
            self._make_429(),
            self._make_response(200),
        ]
        _request_with_429_retry(session, self._URL, self._PARAMS)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [5.0, 10.0]

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_third_retry_sleeps_four_times_base(self, mock_sleep: MagicMock) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            self._make_429(),
            self._make_429(),
            self._make_429(),
            self._make_response(200),
        ]
        _request_with_429_retry(session, self._URL, self._PARAMS)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [5.0, 10.0, 20.0]

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_full_backoff_sequence_is_5_10_20_40_80(
        self, mock_sleep: MagicMock
    ) -> None:
        """Five retries produce the full 5 → 10 → 20 → 40 → 80 delay sequence."""
        session = MagicMock(spec=requests.Session)
        # 1 initial + 4 failing retries + 1 final success = 6 calls
        session.get.side_effect = [
            self._make_429(),  # initial
            self._make_429(),  # retry 1
            self._make_429(),  # retry 2
            self._make_429(),  # retry 3
            self._make_429(),  # retry 4
            self._make_response(200),  # retry 5 succeeds
        ]
        _request_with_429_retry(session, self._URL, self._PARAMS)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [5.0, 10.0, 20.0, 40.0, 80.0]

    # ── Retry exhaustion ───────────────────────────────────────────────────

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_raises_after_max_retries_exhausted(
        self, mock_sleep: MagicMock
    ) -> None:
        """All _MAX_429_RETRIES + 1 attempts return 429 → GDELTAPIError raised."""
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_429()
        with pytest.raises(GDELTAPIError, match="rate-limited"):
            _request_with_429_retry(session, self._URL, self._PARAMS)

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_total_attempts_equals_max_retries_plus_one(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_429()
        with pytest.raises(GDELTAPIError):
            _request_with_429_retry(session, self._URL, self._PARAMS)
        assert session.get.call_count == _MAX_429_RETRIES + 1

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_sleep_called_max_retries_times_on_exhaustion(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_429()
        with pytest.raises(GDELTAPIError):
            _request_with_429_retry(session, self._URL, self._PARAMS)
        assert mock_sleep.call_count == _MAX_429_RETRIES

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_error_message_mentions_retry_count(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_429()
        with pytest.raises(GDELTAPIError, match=str(_MAX_429_RETRIES)):
            _request_with_429_retry(session, self._URL, self._PARAMS)

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_retry_after_header_included_in_log_but_not_used_for_delay(
        self, mock_sleep: MagicMock
    ) -> None:
        """Our own backoff schedule is used; Retry-After header is logged only."""
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            self._make_429(retry_after="60"),  # server says wait 60s
            self._make_response(200),
        ]
        _request_with_429_retry(session, self._URL, self._PARAMS)
        # We sleep our own delay (5s), not the server's 60s
        mock_sleep.assert_called_once_with(_RETRY_429_BASE_DELAY)

    # ── Non-429 errors are not retried ─────────────────────────────────────

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_500_returned_immediately_without_retry(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_response(500)
        resp = _request_with_429_retry(session, self._URL, self._PARAMS)
        assert resp.status_code == 500
        assert session.get.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_404_returned_immediately_without_retry(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = self._make_response(404)
        resp = _request_with_429_retry(session, self._URL, self._PARAMS)
        assert resp.status_code == 404
        mock_sleep.assert_not_called()

    # ── Network errors are not retried ─────────────────────────────────────

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_connection_error_raises_without_retry(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(GDELTAPIError, match="Connection"):
            _request_with_429_retry(session, self._URL, self._PARAMS)
        mock_sleep.assert_not_called()

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_timeout_raises_without_retry(
        self, mock_sleep: MagicMock
    ) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(GDELTAPIError, match="timed out"):
            _request_with_429_retry(session, self._URL, self._PARAMS)
        mock_sleep.assert_not_called()

    # ── Integration with _fetch_chunk ──────────────────────────────────────

    @patch("src.ingestion.gdelt_client.time.sleep")
    def test_fetch_chunk_retries_429_via_helper(
        self, mock_sleep: MagicMock
    ) -> None:
        """_fetch_chunk delegates to _request_with_429_retry, so 429 is retried."""
        import json as _json

        ok_body = {"articles": [{"url": "https://x.com/a", "title": "T",
                                  "seendate": "20250115T120000Z", "domain": "x.com"}]}
        ok_resp = MagicMock(spec=requests.Response)
        ok_resp.status_code = 200
        ok_resp.ok = True
        ok_resp.json.return_value = ok_body
        ok_resp.text = _json.dumps(ok_body)
        ok_resp.headers = {}

        rate_limited = MagicMock(spec=requests.Response)
        rate_limited.status_code = 429
        rate_limited.ok = False
        rate_limited.text = ""
        rate_limited.headers = {}

        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [rate_limited, ok_resp]

        from datetime import datetime, timezone
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end   = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = _fetch_chunk('"Apple Inc"', start, end, 250, session)
        assert len(result) == 1
        assert session.get.call_count == 2
        mock_sleep.assert_called_once_with(_RETRY_429_BASE_DELAY)


# ── TestFetchArticles ─────────────────────────────────────────────────────────

class TestFetchArticles:
    def test_returns_dataframe_with_correct_schema(self) -> None:
        session = _make_mock_session([_make_gdelt_article()])
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == GDELT_ARTICLE_COLUMNS

    def test_all_rows_have_requested_ticker(self) -> None:
        arts = [
            _make_gdelt_article(url=f"https://example.com/{i}")
            for i in range(3)
        ]
        session = _make_mock_session(arts)
        df = fetch_articles("TSLA", from_date=_DT_JAN15, to_date=_DT_JAN17, session=session)
        assert (df["ticker"] == "TSLA").all()

    def test_ticker_normalised_to_uppercase(self) -> None:
        session = _make_mock_session([_make_gdelt_article()])
        df = fetch_articles("aapl", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        assert (df["ticker"] == "AAPL").all()

    def test_published_at_is_utc_aware_datetime(self) -> None:
        session = _make_mock_session([_make_gdelt_article(seendate=_SEENDATE_JAN15)])
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        assert pd.api.types.is_datetime64_any_dtype(df["published_at"])
        assert df["published_at"].dt.tz is not None

    def test_published_at_correctly_parsed_from_seendate(self) -> None:
        session = _make_mock_session([_make_gdelt_article(seendate=_SEENDATE_JAN15)])
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        expected = pd.Timestamp(_DT_JAN15)
        assert df["published_at"].iloc[0] == expected

    def test_deduplicates_by_url(self) -> None:
        dup = _make_gdelt_article(url="https://duplicate.com/story")
        session = _make_mock_session([dup, dup, dup])
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        assert len(df) == 1

    def test_sorted_descending_by_published_at(self) -> None:
        arts = [
            _make_gdelt_article(url="https://x.com/1", seendate=_SEENDATE_JAN15),
            _make_gdelt_article(url="https://x.com/2", seendate=_SEENDATE_JAN17),
            _make_gdelt_article(url="https://x.com/3", seendate=_SEENDATE_JAN16),
        ]
        session = _make_mock_session(arts)
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN17, session=session)
        dates = df["published_at"].tolist()
        assert dates == sorted(dates, reverse=True)

    def test_returns_empty_dataframe_on_no_results(self) -> None:
        session = _make_mock_session([])
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        assert df.empty
        assert list(df.columns) == GDELT_ARTICLE_COLUMNS

    def test_chunk_failures_are_skipped_not_fatal(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.ConnectionError("down")
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        assert df.empty
        assert list(df.columns) == GDELT_ARTICLE_COLUMNS

    def test_default_from_date_used_when_not_provided(self) -> None:
        session = _make_mock_session([_make_gdelt_article()])
        df = fetch_articles("AAPL", session=session)
        assert isinstance(df, pd.DataFrame)

    def test_source_id_is_derived_from_url(self) -> None:
        url = "https://reuters.com/apple-story"
        session = _make_mock_session([_make_gdelt_article(url=url)])
        df = fetch_articles("AAPL", from_date=_DT_JAN15, to_date=_DT_JAN16, session=session)
        expected = _url_to_source_id(url)
        assert df["source_id"].iloc[0] == expected


# ── TestFetchAllTickers ───────────────────────────────────────────────────────

class TestFetchAllTickers:
    @patch("src.ingestion.gdelt_client.fetch_articles")
    def test_returns_dataframe_per_ticker(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS)
        results = fetch_all_tickers(
            tickers=["AAPL", "TSLA"],
            from_date=_DT_JAN15,
            to_date=_DT_JAN16,
        )
        assert set(results.keys()) == {"AAPL", "TSLA"}
        assert mock_fetch.call_count == 2

    @patch("src.ingestion.gdelt_client.fetch_articles")
    def test_api_error_stores_empty_and_continues(self, mock_fetch: MagicMock) -> None:
        good_df = pd.DataFrame({"ticker": ["TSLA"]})
        mock_fetch.side_effect = [
            GDELTAPIError("connection refused"),
            good_df,
        ]
        results = fetch_all_tickers(
            tickers=["AAPL", "TSLA"],
            from_date=_DT_JAN15,
            to_date=_DT_JAN16,
        )
        assert results["AAPL"].empty
        assert not results["TSLA"].empty

    @patch("src.ingestion.gdelt_client.fetch_articles")
    def test_validation_error_stores_empty_and_continues(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = [
            GDELTValidationError("bad response"),
            pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS),
        ]
        results = fetch_all_tickers(
            tickers=["AAPL", "NVDA"],
            from_date=_DT_JAN15,
            to_date=_DT_JAN16,
        )
        assert results["AAPL"].empty

    @patch("src.ingestion.gdelt_client.fetch_articles")
    def test_all_tickers_called(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS)
        tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
        fetch_all_tickers(
            tickers=tickers,
            from_date=_DT_JAN15,
            to_date=_DT_JAN16,
        )
        assert mock_fetch.call_count == 5

    @patch("src.ingestion.gdelt_client.fetch_articles")
    def test_from_date_passed_to_each_ticker(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS)
        fetch_all_tickers(
            tickers=["AAPL"],
            from_date=_DT_JAN15,
            to_date=_DT_JAN17,
        )
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["from_date"] == _DT_JAN15
        assert call_kwargs["to_date"] == _DT_JAN17


# ── TestSummariseResults ──────────────────────────────────────────────────────

class TestSummariseResults:
    def test_ok_row_counts_articles_and_sources(self) -> None:
        df = pd.DataFrame({
            "ticker":       ["AAPL", "AAPL"],
            "published_at": pd.to_datetime([_DT_JAN15, _DT_JAN16], utc=True),
            "source_name":  ["reuters.com", "bloomberg.com"],
        })
        summary = summarise_results({"AAPL": df})
        row = summary[summary["ticker"] == "AAPL"].iloc[0]
        assert row["article_count"] == 2
        assert row["unique_sources"] == 2
        assert row["status"] == "ok"

    def test_empty_ticker_shows_zeros_and_empty_status(self) -> None:
        summary = summarise_results({"TSLA": pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS)})
        row = summary[summary["ticker"] == "TSLA"].iloc[0]
        assert row["article_count"] == 0
        assert row["unique_sources"] == 0
        assert row["status"] == "empty"

    def test_mixed_results_have_both_statuses(self) -> None:
        good_df = pd.DataFrame({
            "ticker":       ["NVDA"],
            "published_at": pd.to_datetime([_DT_JAN15], utc=True),
            "source_name":  ["cnbc.com"],
        })
        summary = summarise_results({
            "NVDA": good_df,
            "MSFT": pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS),
        })
        assert len(summary) == 2
        assert set(summary["status"]) == {"ok", "empty"}

    def test_earliest_less_than_latest(self) -> None:
        df = pd.DataFrame({
            "ticker":       ["AAPL", "AAPL", "AAPL"],
            "published_at": pd.to_datetime([_DT_JAN15, _DT_JAN16, _DT_JAN17], utc=True),
            "source_name":  ["s1", "s2", "s3"],
        })
        summary = summarise_results({"AAPL": df})
        row = summary.iloc[0]
        assert row["earliest"] < row["latest"]


# ── TestGDELTExceptions ───────────────────────────────────────────────────────

class TestGDELTExceptions:
    def test_gdelt_api_error_is_subclass_of_gdelt_error(self) -> None:
        with pytest.raises(GDELTError):
            raise GDELTAPIError("test")

    def test_gdelt_validation_error_is_subclass_of_gdelt_error(self) -> None:
        with pytest.raises(GDELTError):
            raise GDELTValidationError("test")

    def test_gdelt_api_error_is_exception(self) -> None:
        assert issubclass(GDELTAPIError, Exception)

    def test_gdelt_validation_error_is_exception(self) -> None:
        assert issubclass(GDELTValidationError, Exception)

    def test_exception_hierarchy_order(self) -> None:
        assert issubclass(GDELTAPIError, GDELTError)
        assert issubclass(GDELTValidationError, GDELTError)
        assert issubclass(GDELTError, Exception)


# ── TestDateChunks ────────────────────────────────────────────────────────────

class TestDateChunks:
    def test_single_day_window_yields_one_chunk(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end   = datetime(2025, 1, 1, 23, 59, 59, tzinfo=UTC)
        chunks = list(_date_chunks(start, end, chunk_days=15))
        assert len(chunks) == 1
        assert chunks[0] == (start, end)

    def test_chunks_cover_full_date_range(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end   = datetime(2025, 1, 31, tzinfo=UTC)
        chunks = list(_date_chunks(start, end, chunk_days=15))
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

    def test_chunks_do_not_overlap(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end   = datetime(2025, 3, 31, tzinfo=UTC)
        chunks = list(_date_chunks(start, end, chunk_days=15))
        for i in range(len(chunks) - 1):
            assert chunks[i][1] < chunks[i + 1][0]

    def test_30_day_window_with_15_day_chunks_yields_two_chunks(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end   = datetime(2025, 1, 30, 23, 59, 59, tzinfo=UTC)
        chunks = list(_date_chunks(start, end, chunk_days=15))
        assert len(chunks) == 2

    def test_chunk_end_never_exceeds_overall_end(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end   = datetime(2025, 2, 28, tzinfo=UTC)
        for _, chunk_end in _date_chunks(start, end, chunk_days=15):
            assert chunk_end <= end


# ── TestCoerceDtypes ──────────────────────────────────────────────────────────

class TestCoerceDtypes:
    def test_seendate_string_converted_to_utc_datetime(self) -> None:
        df = pd.DataFrame({
            "published_at": [_SEENDATE_JAN15],
            "fetched_at":   [datetime(2025, 1, 20, tzinfo=UTC).isoformat()],
            "title":        ["Test"],
            "source_name":  ["reuters.com"],
        })
        result = _coerce_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(result["published_at"])
        assert result["published_at"].dt.tz is not None

    def test_none_seendate_becomes_nat(self) -> None:
        df = pd.DataFrame({
            "published_at": [None],
            "fetched_at":   [datetime(2025, 1, 20, tzinfo=UTC).isoformat()],
            "title":        ["Test"],
            "source_name":  ["reuters.com"],
        })
        result = _coerce_dtypes(df)
        assert pd.isna(result["published_at"].iloc[0])

    def test_whitespace_stripped_from_title(self) -> None:
        df = pd.DataFrame({
            "published_at": [_SEENDATE_JAN15],
            "fetched_at":   [datetime(2025, 1, 20, tzinfo=UTC).isoformat()],
            "title":        ["  Apple News  "],
            "source_name":  ["reuters.com"],
        })
        result = _coerce_dtypes(df)
        assert result["title"].iloc[0] == "Apple News"

    def test_empty_string_source_name_becomes_none(self) -> None:
        df = pd.DataFrame({
            "published_at": [_SEENDATE_JAN15],
            "fetched_at":   [datetime(2025, 1, 20, tzinfo=UTC).isoformat()],
            "title":        ["Test"],
            "source_name":  [""],
        })
        result = _coerce_dtypes(df)
        assert result["source_name"].iloc[0] is None
