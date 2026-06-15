"""
Finnhub News client for financial news ingestion.

Provides two public functions:

- ``fetch_articles(ticker, ...)``     → pd.DataFrame  (single ticker)
- ``fetch_all_tickers(tickers, ...)`` → dict[str, pd.DataFrame]

Both functions return DataFrames with a consistent schema and handle all
API error conditions — auth failures, rate limits, network timeouts, and
empty result sets — without leaking raw HTTP details to callers.

Finnhub free-tier constraints
-------------------------------
- 60 API calls / minute
- Company news available up to ~1 year back
- No ``author`` or full ``content`` field (Finnhub omits both)
- ``source_id`` is populated with the integer article ID from Finnhub

The module-level ``_rate_limiter`` enforces 0.9 req/s (burst of 5) to stay
safely within the 60 calls/minute quota even when fetching many tickers.

Finnhub API reference
----------------------
GET https://finnhub.io/api/v1/company-news
  ?symbol=AAPL&from=2024-01-01&to=2024-01-07&token=<key>

Response: JSON array of article objects (not a wrapped envelope):
  [
    {
      "category": "company news",
      "datetime": 1704844800,      # Unix timestamp (seconds)
      "headline": "Apple...",
      "id":       123456,          # Finnhub article ID (integer)
      "image":    "https://...",
      "related":  "AAPL",
      "source":   "Reuters",       # plain string, not {id, name}
      "summary":  "...",
      "url":      "https://..."
    },
    ...
  ]
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.rate_limiter import TokenBucketRateLimiter

UTC = timezone.utc

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# DataFrame column order — used when returning empty DataFrames so callers
# can always rely on a consistent schema regardless of API results.
ARTICLE_COLUMNS: list[str] = [
    "ticker",
    "source_id",      # Finnhub integer article ID (stored as string)
    "source_name",    # Finnhub "source" field (plain string)
    "author",         # Always None — Finnhub does not provide author
    "title",          # Finnhub "headline" field
    "description",    # Finnhub "summary" field
    "url",
    "published_at",   # Converted from Finnhub Unix timestamp (UTC-aware)
    "content",        # Always None — Finnhub does not provide full content
    "fetched_at",
]

# Finnhub free tier: 60 calls/minute = 1 call/second.
# 0.9 req/s with burst-5 keeps us ~10% under the cap.
_rate_limiter = TokenBucketRateLimiter(rate=0.9, capacity=5)


# ── Exceptions ───────────────────────────────────────────────────────────────

class FinnhubError(Exception):
    """Base exception for all Finnhub news client errors."""


class FinnhubAuthError(FinnhubError):
    """Invalid or missing API key (HTTP 401 / 403)."""


class FinnhubRateLimitError(FinnhubError):
    """Per-minute request quota exceeded (HTTP 429)."""


class FinnhubRequestError(FinnhubError):
    """Malformed request or unexpected API-level error."""


class FinnhubNetworkError(FinnhubError):
    """Network-level failure: DNS, connection refused, timeout, etc."""


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Return a Session with retry logic for transient server errors.

    Retries up to 3 times with exponential backoff on 5xx responses.
    4xx errors are NOT retried — they represent caller mistakes.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,           # waits 1s, 2s, 4s between retries
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,        # we inspect status codes ourselves
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parse_article(raw: dict[str, Any], ticker: str, fetched_at: datetime) -> dict[str, Any]:
    """Flatten a single raw Finnhub article dict into our schema.

    Field mapping
    -------------
    Finnhub field   → Schema column
    -------------      -------------
    id (int)        → source_id (str)
    source (str)    → source_name
    headline        → title
    summary         → description
    datetime (int)  → published_at  (Unix seconds → kept as int here,
                                     converted to datetime in _coerce_dtypes)
    url             → url
    (absent)        → author   = None
    (absent)        → content  = None
    """
    return {
        "ticker":      ticker,
        "source_id":   str(raw["id"]) if raw.get("id") is not None else None,
        "source_name": raw.get("source", ""),
        "author":      None,
        "title":       raw.get("headline", ""),
        "description": raw.get("summary"),
        "url":         raw.get("url", ""),
        "published_at": raw.get("datetime"),   # Unix int — coerced later
        "content":     None,
        "fetched_at":  fetched_at.isoformat(),
    }


def _check_response_errors(response: requests.Response, ticker: str) -> None:
    """Inspect HTTP status codes and Finnhub error payloads, raising typed exceptions.

    Finnhub success: HTTP 200 + JSON array body.
    Finnhub errors:  non-2xx status + optional ``{"error": "..."}`` JSON body.
    """
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
            f"Retry-After: {retry_after}s. "
            "Wait before retrying or reduce request frequency."
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


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Unix timestamps to UTC datetimes and strip string whitespace."""
    # Finnhub published_at is a Unix timestamp (integer seconds).
    df["published_at"] = pd.to_datetime(
        df["published_at"], unit="s", utc=True, errors="coerce"
    )
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")

    for col in ("title", "description", "source_name"):
        if col in df.columns:
            df[col] = df[col].str.strip().replace("", None)

    return df


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=ARTICLE_COLUMNS)


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_articles(
    ticker: str,
    *,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    max_pages: int | None = None,  # noqa: ARG001 — kept for API compatibility
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch company news articles for a single ticker from Finnhub.

    Finnhub returns all results for the requested date range in a single
    response (no pagination). The ``max_pages`` parameter is accepted but
    ignored; it exists to keep this function's signature compatible with
    the rest of the codebase.

    Parameters
    ----------
    ticker:
        Uppercase stock ticker symbol (e.g. ``"AAPL"``).
    from_date:
        Start of the date range (UTC-aware). Defaults to
        ``settings.news_lookback_days`` days before now.
    to_date:
        End of the date range (UTC-aware). Defaults to now.
    max_pages:
        Accepted but unused. Finnhub does not support pagination.
    session:
        Optional pre-built ``requests.Session``. Pass a shared session
        when fetching multiple tickers to reuse TCP connections.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``source_id``, ``source_name``, ``author``,
        ``title``, ``description``, ``url``, ``published_at``, ``content``,
        ``fetched_at``.
        Sorted descending by ``published_at``, duplicates removed by URL.
        Returns an empty DataFrame (correct schema) if no articles are found.

    Raises
    ------
    FinnhubAuthError
        Invalid or missing API key (401), or premium-plan restriction (403).
    FinnhubRateLimitError
        Per-minute quota exceeded (429).
    FinnhubRequestError
        Unexpected API response or unparseable body.
    FinnhubNetworkError
        DNS failure, connection refused, or request timeout.
    """
    ticker = ticker.upper()
    now = datetime.now(UTC)
    from_date = from_date or (now - timedelta(days=settings.news_lookback_days))
    to_date = to_date or now
    session = session or _build_session()
    fetched_at = now

    logger.info(
        f"[{ticker}] Fetching from Finnhub | "
        f"window: {from_date.date()} → {to_date.date()}"
    )

    _rate_limiter.acquire()

    params: dict[str, Any] = {
        "symbol": ticker,
        "from":   from_date.strftime("%Y-%m-%d"),
        "to":     to_date.strftime("%Y-%m-%d"),
        "token":  settings.finnhub_api_key,
    }

    logger.debug(f"[{ticker}] GET /company-news")

    try:
        response = session.get(
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

    articles: list[dict[str, Any]] = response.json()

    if not articles:
        logger.warning(f"[{ticker}] No articles found for the given date window")
        return _empty_dataframe()

    logger.info(f"[{ticker}] Received {len(articles)} articles from Finnhub")

    parsed = [_parse_article(a, ticker, fetched_at) for a in articles]
    df = pd.DataFrame(parsed, columns=ARTICLE_COLUMNS)
    df = _coerce_dtypes(df)
    df = df.drop_duplicates(subset="url", keep="first")
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    logger.info(f"[{ticker}] Done — {len(df)} unique articles")
    return df


def fetch_all_tickers(
    tickers: list[str] | None = None,
    *,
    days_back: int | None = None,
    max_pages: int | None = None,  # noqa: ARG001 — kept for API compatibility
) -> dict[str, pd.DataFrame]:
    """Fetch news for multiple tickers, returning a mapping of ticker → DataFrame.

    Tickers that encounter non-fatal errors (network issues, empty results)
    are logged and stored as empty DataFrames; the loop continues with the
    remaining tickers. Auth and rate-limit errors abort the entire run
    because all subsequent requests would fail too.

    Parameters
    ----------
    tickers:
        List of uppercase ticker symbols. Defaults to ``settings.tickers``.
    days_back:
        Override ``settings.news_lookback_days`` for all tickers.
    max_pages:
        Accepted but unused. Finnhub does not support pagination.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are the ticker symbols. Values are DataFrames (may be empty).
    """
    tickers = tickers or settings.tickers
    days_back = days_back or settings.news_lookback_days

    now = datetime.now(UTC)
    from_date = now - timedelta(days=days_back)
    session = _build_session()
    results: dict[str, pd.DataFrame] = {}

    logger.info(
        f"Fetching news for {len(tickers)} tickers: {tickers} | "
        f"lookback: {days_back}d"
    )

    for ticker in tickers:
        try:
            results[ticker] = fetch_articles(
                ticker,
                from_date=from_date,
                to_date=now,
                session=session,
            )

        except FinnhubAuthError:
            # Fatal — all requests use the same key, so no point continuing.
            logger.error(
                "Authentication error — aborting. "
                "Verify FINNHUB_API_KEY in your .env file."
            )
            raise

        except FinnhubRateLimitError as exc:
            # Fatal for this minute — stop to avoid hammering the endpoint.
            logger.error(f"[{ticker}] Rate limit hit: {exc}")
            logger.warning(
                "Stopping remaining tickers to avoid further quota waste. "
                "Finnhub allows 60 calls/minute — wait before retrying."
            )
            results[ticker] = _empty_dataframe()
            break

        except FinnhubNetworkError as exc:
            # Transient — log and continue with next ticker.
            logger.warning(f"[{ticker}] Network error (skipping): {exc}")
            results[ticker] = _empty_dataframe()

        except FinnhubRequestError as exc:
            # Unexpected API response — log and continue.
            logger.warning(f"[{ticker}] Request error (skipping): {exc}")
            results[ticker] = _empty_dataframe()

    successful = sum(1 for df in results.values() if not df.empty)
    total_articles = sum(len(df) for df in results.values())
    logger.info(
        f"Fetch complete — {successful}/{len(tickers)} tickers returned data | "
        f"total articles: {total_articles}"
    )
    return results


def summarise_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a one-row-per-ticker summary DataFrame for quick inspection.

    Returns a DataFrame with columns: ``ticker``, ``article_count``,
    ``earliest``, ``latest``, ``unique_sources``, ``status``.
    """
    rows: list[dict[str, Any]] = []
    for ticker, df in results.items():
        if df.empty:
            rows.append({
                "ticker":         ticker,
                "article_count":  0,
                "earliest":       None,
                "latest":         None,
                "unique_sources": 0,
                "status":         "empty",
            })
        else:
            rows.append({
                "ticker":         ticker,
                "article_count":  len(df),
                "earliest":       df["published_at"].min(),
                "latest":         df["published_at"].max(),
                "unique_sources": df["source_name"].nunique(),
                "status":         "ok",
            })
    return pd.DataFrame(rows)
