"""
Phase 8: Finnhub client for live company news.

Fetches the latest company news articles from Finnhub and returns them as
normalized article dictionaries ready for sentiment analysis and feature
engineering.

The API key is read from :attr:`~src.utils.config.Settings.finnhub_api_key`
(loaded via ``.env`` through :mod:`src.utils.config`).

Usage
-----
    from src.realtime.finnhub_client import FinnhubClient

    client = FinnhubClient()
    articles = client.fetch_latest_news("AAPL")
    latest = articles[0]  # sorted newest-first
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import settings
from src.utils.logger import get_logger

UTC = timezone.utc
logger = get_logger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Normalized article dict keys (aligned with Phase 1 ingestion schema).
ARTICLE_KEYS: tuple[str, ...] = (
    "ticker",
    "source_id",
    "source_name",
    "author",
    "title",
    "description",
    "url",
    "published_at",
    "content",
    "fetched_at",
)


# ── Exceptions ────────────────────────────────────────────────────────────────

class FinnhubError(Exception):
    """Base exception for Finnhub realtime client errors."""


class FinnhubConfigError(FinnhubError):
    """Missing or invalid API key configuration."""


class FinnhubAuthError(FinnhubError):
    """Invalid or missing API key (HTTP 401 / 403)."""


class FinnhubRateLimitError(FinnhubError):
    """Per-minute request quota exceeded (HTTP 429)."""


class FinnhubRequestError(FinnhubError):
    """Malformed request or unexpected API-level error."""


class FinnhubNetworkError(FinnhubError):
    """Network-level failure: DNS, connection refused, timeout, etc."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_api_key(explicit_key: str | None = None) -> str:
    """
    Resolve the Finnhub API key from an explicit value or application settings.

    Raises
    ------
    FinnhubConfigError
        When the key is absent or still set to the placeholder value.
    """
    key = (explicit_key or settings.finnhub_api_key).strip()
    if not key or key == "your_finnhub_api_key_here":
        raise FinnhubConfigError(
            "FINNHUB_API_KEY is not set. "
            "Copy .env.example to .env and add your key from https://finnhub.io/register"
        )
    return key


def _build_session() -> requests.Session:
    """Return a Session with retry logic for transient server errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parse_article(
    raw: dict[str, Any],
    ticker: str,
    fetched_at: datetime,
) -> dict[str, Any]:
    """Flatten a raw Finnhub article dict into the normalized schema."""
    unix_ts = raw.get("datetime")
    published_at: datetime | None = None
    if unix_ts is not None:
        published_at = datetime.fromtimestamp(int(unix_ts), tz=UTC)

    return {
        "ticker":       ticker,
        "source_id":    str(raw["id"]) if raw.get("id") is not None else None,
        "source_name":  (raw.get("source") or "").strip() or None,
        "author":       None,
        "title":        (raw.get("headline") or "").strip(),
        "description":  (raw.get("summary") or "").strip() or None,
        "url":          (raw.get("url") or "").strip(),
        "published_at": published_at,
        "content":      None,
        "fetched_at":   fetched_at,
    }


def _check_response_errors(response: requests.Response, ticker: str) -> None:
    """Inspect HTTP status codes and raise typed exceptions."""
    if response.status_code == 401:
        raise FinnhubAuthError(
            "Authentication failed. "
            "Check that FINNHUB_API_KEY in your .env is correct."
        )

    if response.status_code == 403:
        raise FinnhubAuthError(
            f"Access denied for '{ticker}'. "
            "This endpoint may require a Finnhub premium plan."
        )

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        raise FinnhubRateLimitError(
            f"Finnhub rate limit exceeded while fetching '{ticker}'. "
            f"Retry-After: {retry_after}s."
        )

    if not response.ok:
        try:
            body = response.json()
            msg = body.get("error", response.text[:200])
        except ValueError:
            msg = response.text[:200]
        raise FinnhubRequestError(
            f"Finnhub returned HTTP {response.status_code} for '{ticker}': {msg}"
        )


# ── Client ────────────────────────────────────────────────────────────────────

class FinnhubClient:
    """
    Fetch live company news from Finnhub for real-time inference.

    Parameters
    ----------
    api_key : str | None
        Explicit API key override.  When ``None``, reads
        ``settings.finnhub_api_key`` from :mod:`src.utils.config`.
    days_back : int
        Number of calendar days to look back when fetching news.  Default: ``7``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        days_back: int = 7,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self._days_back = days_back
        self._session = _build_session()

    def fetch_latest_news(
        self,
        ticker: str,
        *,
        days_back: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch company news for *ticker* and return normalized article dicts.

        Articles are deduplicated by URL, sorted newest-first, and returned
        as a list of dictionaries with keys matching :data:`ARTICLE_KEYS`.

        Parameters
        ----------
        ticker : str
            Uppercase stock ticker symbol (e.g. ``"AAPL"``).
        days_back : int | None
            Override the instance-level lookback window.

        Returns
        -------
        list[dict]
            Normalized article dictionaries.  Empty list when no articles found.

        Raises
        ------
        FinnhubAuthError, FinnhubRateLimitError, FinnhubRequestError,
        FinnhubNetworkError
        """
        ticker = ticker.upper()
        lookback = days_back if days_back is not None else self._days_back
        now = datetime.now(UTC)
        from_date = now - timedelta(days=lookback)
        fetched_at = now

        logger.info(f"[{ticker}] Fetching latest news from Finnhub …")

        params: dict[str, Any] = {
            "symbol": ticker,
            "from":   from_date.strftime("%Y-%m-%d"),
            "to":     now.strftime("%Y-%m-%d"),
            "token":  self._api_key,
        }

        try:
            response = self._session.get(
                f"{FINNHUB_BASE_URL}/company-news",
                params=params,
                timeout=15,
            )
        except requests.exceptions.ConnectionError as exc:
            raise FinnhubNetworkError(
                f"[{ticker}] Connection failed: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise FinnhubNetworkError(
                f"[{ticker}] Request timed out after 15s: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise FinnhubNetworkError(
                f"[{ticker}] Unexpected request error: {exc}"
            ) from exc

        _check_response_errors(response, ticker)

        raw_articles: list[dict[str, Any]] = response.json()
        if not raw_articles:
            logger.warning(f"[{ticker}] No articles found in the last {lookback} day(s)")
            return []

        parsed = [_parse_article(a, ticker, fetched_at) for a in raw_articles]

        # Deduplicate by URL, keep newest first.
        seen_urls: set[str] = set()
        unique: list[dict[str, Any]] = []
        for article in sorted(
            parsed,
            key=lambda a: a["published_at"] or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        ):
            url = article.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(article)

        logger.info(f"[{ticker}] Fetched {len(unique)} unique article(s)")
        return unique
