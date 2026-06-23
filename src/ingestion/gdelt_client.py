"""
GDELT DOC API v2 client for historical financial news ingestion.

Provides two public functions:

- ``fetch_articles(ticker, ...)``      → pd.DataFrame  (single ticker)
- ``fetch_all_tickers(tickers, ...)``  → dict[str, pd.DataFrame]

Both functions return DataFrames with the same column schema as the Finnhub
client (``ARTICLE_COLUMNS`` from ``news_client.py``) so that GDELT output
can flow through the existing FinBERT → PostgreSQL → Feature Engineering →
ML Dataset pipeline without modification.

GDELT DOC API v2
----------------
GET https://api.gdeltproject.org/api/v2/doc/doc
    ?query="Apple Inc"
    &mode=ArtList
    &format=json
    &sort=DateDesc
    &maxrecords=250
    &startdatetime=20250101000000
    &enddatetime=20250115235959

Response envelope::

    {
      "articles": [
        {
          "url":          "https://reuters.com/...",
          "title":        "Apple Inc Reports Strong Q1",
          "seendate":     "20250115T120000Z",
          "domain":       "reuters.com",
          "language":     "English",
          "sourcecountry":"United States"
        },
        ...
      ]
    }

GDELT constraints
-----------------
- Free; no API key required.
- Hard limit of 250 results per request.
- ``seendate`` records when GDELT first indexed the article (UTC).
- No article-level ID, author, summary, or full-content fields.

Field mapping
-------------
GDELT field         Schema column     Notes
-----------         -------------     -----
url                 url
domain              source_name       e.g. ``"reuters.com"``
title               title
seendate            published_at      ``"20250115T120000Z"`` → UTC datetime
sha256(url)[:16]    source_id         deterministic 16-char hex ID from URL
(absent)            description=None  GDELT provides no summaries
(absent)            author=None
(absent)            content=None

Date chunking
-------------
GDELT returns at most 250 results per call.  For date windows longer than
``_CHUNK_DAYS`` days the client automatically splits the range into
sub-windows and merges the results, then deduplicates by URL.  A small
delay (``_REQUEST_DELAY`` seconds) is inserted between sub-requests as a
courtesy to GDELT's free-tier infrastructure.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logger import get_logger

UTC = timezone.utc

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Ticker → quoted company-name query string.
# Quoted phrases prevent spurious partial matches (e.g. "Apple" → "Applebee's").
TICKER_QUERY_MAP: dict[str, str] = {
    "AAPL": '"Apple Inc"',
    "TSLA": '"Tesla"',
    "NVDA": '"NVIDIA"',
    "MSFT": '"Microsoft"',
    "AMZN": '"Amazon"',
}

# Hard per-call limit enforced by GDELT.
GDELT_MAX_RECORDS: int = 250

# Days covered per sub-request when splitting a long date window.
_CHUNK_DAYS: int = 7

# Politeness delay (seconds) inserted between consecutive GDELT requests.
_REQUEST_DELAY: float = 5.0

# HTTP 429 retry configuration.
# GDELT occasionally rate-limits free-tier clients.  We retry up to
# _MAX_429_RETRIES times with exponential backoff starting at
# _RETRY_429_BASE_DELAY seconds: 5s → 10s → 20s → 40s → 80s.
_MAX_429_RETRIES: int = 5
_RETRY_429_BASE_DELAY: float = 5.0

# DataFrame column order — intentionally identical to ARTICLE_COLUMNS in
# news_client.py so that Finnhub and GDELT DataFrames are interchangeable
# throughout the pipeline.
GDELT_ARTICLE_COLUMNS: list[str] = [
    "ticker",
    "source_id",      # SHA-256(url)[:16] — stable deterministic identifier
    "source_name",    # GDELT "domain" field  (e.g. "reuters.com")
    "author",         # Always None — GDELT does not provide author
    "title",          # GDELT "title" field
    "description",    # Always None — GDELT does not provide summaries
    "url",
    "published_at",   # Converted from GDELT "seendate" (YYYYMMDDTHHMMSSZ)
    "content",        # Always None — GDELT does not provide full content
    "fetched_at",
]


# ── Exceptions ───────────────────────────────────────────────────────────────

class GDELTError(Exception):
    """Base exception for all GDELT client errors."""


class GDELTAPIError(GDELTError):
    """Network-level failure or non-2xx HTTP response from the GDELT DOC API."""


class GDELTValidationError(GDELTError):
    """Malformed or structurally unexpected response from the GDELT DOC API."""


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Return a Session with retry logic for transient 5xx errors.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s).
    4xx errors are not retried — they represent caller mistakes.
    """
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


def _build_query(ticker: str) -> str:
    """Return the GDELT query string for *ticker*.

    Uses a quoted company-name phrase from ``TICKER_QUERY_MAP`` when
    available to avoid noisy partial-word matches.  Unknown tickers fall
    back to the symbol itself (e.g. ``"GOOGL"``).
    """
    return TICKER_QUERY_MAP.get(ticker.upper(), ticker.upper())


def _parse_seendate(seendate: Any) -> datetime | None:
    """Parse a GDELT ``seendate`` value to a UTC-aware ``datetime``.

    Accepts the canonical GDELT format ``"YYYYMMDDTHHMMSSZ"``
    (e.g. ``"20250115T120000Z"``).  Returns ``None`` on any parse failure
    so that downstream ``pd.to_datetime(..., errors='coerce')`` produces
    ``NaT`` rather than raising.
    """
    if seendate is None:
        return None
    if isinstance(seendate, datetime):
        if seendate.tzinfo is None:
            return seendate.replace(tzinfo=UTC)
        return seendate
    if not isinstance(seendate, str) or not seendate:
        return None
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        logger.debug(f"Could not parse GDELT seendate: {seendate!r}")
        return None


def _url_to_source_id(url: str) -> str:
    """Return a stable 16-hex-char source ID derived from *url*.

    GDELT provides no article-level integer ID.  Taking the first 16
    characters of SHA-256(url) yields a collision-resistant, deterministic
    identifier that remains consistent across repeated fetches of the same
    article and fits cleanly in the ``source_id VARCHAR(64)`` column.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _parse_article(
    raw: dict[str, Any],
    ticker: str,
    fetched_at: datetime,
) -> dict[str, Any]:
    """Flatten a single GDELT article dict into the shared pipeline schema.

    Field mapping
    -------------
    GDELT field      Schema column     Notes
    -----------      -------------     -----
    url              url
    domain           source_name
    title            title
    seendate         published_at      raw string; coerced by ``_coerce_dtypes``
    sha256(url)[:16] source_id
    (absent)         description=None
    (absent)         author=None
    (absent)         content=None
    """
    url = raw.get("url", "")
    return {
        "ticker":       ticker,
        "source_id":    _url_to_source_id(url) if url else "",
        "source_name":  raw.get("domain", ""),
        "author":       None,
        "title":        raw.get("title", ""),
        "description":  None,
        "url":          url,
        "published_at": raw.get("seendate"),   # raw string; converted in _coerce_dtypes
        "content":      None,
        "fetched_at":   fetched_at.isoformat(),
    }


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GDELT seendate strings to UTC datetimes and strip whitespace."""
    df["published_at"] = pd.to_datetime(
        [_parse_seendate(v) for v in df["published_at"]],
        utc=True,
        errors="coerce",
    )
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    for col in ("title", "source_name"):
        if col in df.columns:
            df[col] = df[col].str.strip().replace("", None)
    return df


def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the correct column schema."""
    return pd.DataFrame(columns=GDELT_ARTICLE_COLUMNS)


def _date_chunks(
    start: datetime,
    end: datetime,
    chunk_days: int = _CHUNK_DAYS,
) -> Iterator[tuple[datetime, datetime]]:
    """Yield non-overlapping ``(chunk_start, chunk_end)`` pairs covering ``[start, end]``.

    Each chunk spans exactly ``chunk_days`` days except the final chunk which
    may be shorter.  Chunks do not overlap — the end of chunk N is exactly
    one second before the start of chunk N+1.
    """
    current = start
    while current <= end:
        chunk_end = min(
            current + timedelta(days=chunk_days) - timedelta(seconds=1),
            end,
        )
        yield current, chunk_end
        current = current + timedelta(days=chunk_days)


def _request_with_429_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int = 30,
) -> requests.Response:
    """Execute a GET request, retrying on HTTP 429 with exponential backoff.

    GDELT occasionally rate-limits free-tier clients (HTTP 429).  This
    function retries up to ``_MAX_429_RETRIES`` times, waiting
    ``_RETRY_429_BASE_DELAY`` seconds before the first retry and doubling
    the wait on each subsequent attempt:

    Attempt   Action          Wait before next
    -------   ------          ----------------
    1 (init)  GET …           —
    2         retry 1/5       5 s
    3         retry 2/5       10 s
    4         retry 3/5       20 s
    5         retry 4/5       40 s
    6         retry 5/5       80 s
    —         raise           (all retries exhausted)

    Non-429 responses (including 5xx) are returned immediately without
    further retry — the caller is responsible for inspecting ``response.ok``.

    Raises
    ------
    GDELTAPIError
        Network-level failure (connection refused, timeout, etc.) or all
        429 retry attempts exhausted.
    """
    delay = _RETRY_429_BASE_DELAY

    for attempt in range(_MAX_429_RETRIES + 1):   # attempt 0 = initial request
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.exceptions.ConnectionError as exc:
            raise GDELTAPIError(f"Connection to GDELT failed: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise GDELTAPIError(
                f"GDELT request timed out after {timeout}s: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise GDELTAPIError(f"GDELT request error: {exc}") from exc

        if response.status_code != 429:
            return response

        # HTTP 429 received — decide whether to retry or give up.
        if attempt >= _MAX_429_RETRIES:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise GDELTAPIError(
                f"GDELT rate-limited (HTTP 429) after {_MAX_429_RETRIES} retries. "
                f"Retry-After header: {retry_after}. Try again later."
            )

        retry_after_hdr = response.headers.get("Retry-After")
        extra = f" (Retry-After: {retry_after_hdr}s)" if retry_after_hdr else ""
        logger.warning(
            f"GDELT HTTP 429 — rate limited{extra}. "
            f"Retry {attempt + 1}/{_MAX_429_RETRIES} in {delay:.0f}s..."
        )
        time.sleep(delay)
        delay *= 2

    # Unreachable — the loop always returns or raises before this point.
    raise GDELTAPIError("GDELT retry loop exited unexpectedly")  # pragma: no cover


def _fetch_chunk(
    query: str,
    chunk_start: datetime,
    chunk_end: datetime,
    max_records: int,
    session: requests.Session,
) -> list[dict[str, Any]]:
    """Execute a single GDELT DOC API call and return the raw articles list.

    HTTP 429 (rate limit) responses are retried automatically via
    ``_request_with_429_retry`` with exponential backoff (5 → 10 → 20 →
    40 → 80 seconds, up to ``_MAX_429_RETRIES`` attempts).

    Parameters
    ----------
    query:
        GDELT query string (e.g. ``'"Apple Inc"'``).
    chunk_start / chunk_end:
        UTC-aware datetime boundaries for this sub-window.
    max_records:
        Capped at ``GDELT_MAX_RECORDS`` (250) — GDELT's hard limit.
    session:
        Shared ``requests.Session`` with retry logic.

    Returns
    -------
    list[dict]
        Raw GDELT article dicts.  Empty list if GDELT found nothing.

    Raises
    ------
    GDELTAPIError
        Network failure, all 429 retries exhausted, or other non-2xx HTTP.
    GDELTValidationError
        Response body is not parseable JSON or lacks the ``articles`` key.
    """
    params: dict[str, Any] = {
        "query":         query,
        "mode":          "ArtList",
        "format":        "json",
        "sort":          "DateDesc",
        "maxrecords":    min(max_records, GDELT_MAX_RECORDS),
        "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   chunk_end.strftime("%Y%m%d%H%M%S"),
    }

    logger.debug(
        f"GDELT request | query={query!r} "
        f"start={chunk_start.date()} end={chunk_end.date()} "
        f"maxrecords={params['maxrecords']}"
    )

    response = _request_with_429_retry(session, GDELT_BASE_URL, params)

    if not response.ok:
        raise GDELTAPIError(
            f"GDELT returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise GDELTValidationError(
            f"GDELT response is not valid JSON: {response.text[:200]}"
        ) from exc

    if not isinstance(body, dict):
        raise GDELTValidationError(
            f"Expected a JSON object from GDELT; got {type(body).__name__}"
        )

    articles = body.get("articles") or []
    if not isinstance(articles, list):
        raise GDELTValidationError(
            f"Expected 'articles' to be a list; got {type(articles).__name__}"
        )

    return articles


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_articles(
    ticker: str,
    *,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    max_records: int = GDELT_MAX_RECORDS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch historical news articles for a single ticker from GDELT.

    The date window is automatically split into ``_CHUNK_DAYS``-day
    sub-windows (each limited to ``max_records`` results) to avoid
    truncation on news-heavy periods.  A ``_REQUEST_DELAY``-second pause
    is inserted between sub-requests as a courtesy to GDELT's free
    infrastructure.

    Parameters
    ----------
    ticker:
        Uppercase stock ticker symbol (e.g. ``"AAPL"``).
    from_date:
        Start of the date range (UTC-aware).  Defaults to 30 days before now.
    to_date:
        End of the date range (UTC-aware).  Defaults to now.
    max_records:
        Maximum articles per GDELT sub-request.  Hard-capped at 250.
    session:
        Optional pre-built ``requests.Session``.  Pass a shared session
        when fetching multiple tickers to reuse TCP connections.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``source_id``, ``source_name``, ``author``,
        ``title``, ``description``, ``url``, ``published_at``, ``content``,
        ``fetched_at``.
        Sorted descending by ``published_at``, deduplicated by URL.
        Returns an empty DataFrame (correct schema) if no articles found.

    Raises
    ------
    GDELTAPIError
        Network-level failure or non-200 HTTP response.
    GDELTValidationError
        Unexpected response structure.
    """
    ticker = ticker.upper()
    now = datetime.now(UTC)
    to_date = to_date or now
    from_date = from_date or (now - timedelta(days=30))
    session = session or _build_session()
    fetched_at = now
    query = _build_query(ticker)

    logger.info(
        f"[{ticker}] Fetching from GDELT | "
        f"query={query!r} | "
        f"window: {from_date.date()} → {to_date.date()}"
    )

    all_articles: list[dict[str, Any]] = []
    chunk_count = 0

    for chunk_start, chunk_end in _date_chunks(from_date, to_date):
        if chunk_count > 0:
            time.sleep(_REQUEST_DELAY)

        try:
            chunk_articles = _fetch_chunk(
                query=query,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                max_records=max_records,
                session=session,
            )
        except (GDELTAPIError, GDELTValidationError) as exc:
            logger.warning(
                f"[{ticker}] Chunk {chunk_start.date()}–{chunk_end.date()} "
                f"failed (skipping): {exc}"
            )
            chunk_count += 1
            continue

        logger.debug(
            f"[{ticker}] Chunk {chunk_start.date()}–{chunk_end.date()} "
            f"→ {len(chunk_articles)} articles"
        )
        all_articles.extend(chunk_articles)
        chunk_count += 1

    if not all_articles:
        logger.warning(f"[{ticker}] No articles found in the given date window")
        return _empty_dataframe()

    logger.info(
        f"[{ticker}] Raw total: {len(all_articles)} articles "
        f"across {chunk_count} chunk(s)"
    )

    parsed = [_parse_article(a, ticker, fetched_at) for a in all_articles]
    df = pd.DataFrame(parsed, columns=GDELT_ARTICLE_COLUMNS)
    df = _coerce_dtypes(df)
    df = df.drop_duplicates(subset="url", keep="first")
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    logger.info(f"[{ticker}] Done — {len(df)} unique articles after deduplication")
    return df


def fetch_all_tickers(
    tickers: list[str] | None = None,
    *,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    max_records: int = GDELT_MAX_RECORDS,
) -> dict[str, pd.DataFrame]:
    """Fetch news for multiple tickers, returning a mapping of ticker → DataFrame.

    Tickers that encounter non-fatal errors are logged and stored as empty
    DataFrames; the loop continues with remaining tickers.

    Parameters
    ----------
    tickers:
        List of uppercase ticker symbols.  Defaults to ``settings.tickers``
        when not provided.
    from_date:
        Start of the date range (UTC-aware).  Defaults to 30 days before now.
    to_date:
        End of the date range (UTC-aware).  Defaults to now.
    max_records:
        Maximum articles per GDELT sub-request per chunk.  Hard-capped at 250.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ticker symbols.  Values are DataFrames (may be empty).
    """
    if tickers is None:
        from src.utils.config import settings  # noqa: PLC0415
        tickers = list(settings.tickers)

    now = datetime.now(UTC)
    to_date = to_date or now
    from_date = from_date or (now - timedelta(days=30))
    session = _build_session()
    results: dict[str, pd.DataFrame] = {}

    logger.info(
        f"Fetching GDELT news for {len(tickers)} tickers: {tickers} | "
        f"window: {from_date.date()} → {to_date.date()}"
    )

    for ticker in tickers:
        try:
            results[ticker] = fetch_articles(
                ticker,
                from_date=from_date,
                to_date=to_date,
                max_records=max_records,
                session=session,
            )
        except GDELTAPIError as exc:
            logger.warning(f"[{ticker}] API error (skipping): {exc}")
            results[ticker] = _empty_dataframe()
        except GDELTValidationError as exc:
            logger.warning(f"[{ticker}] Validation error (skipping): {exc}")
            results[ticker] = _empty_dataframe()
        except GDELTError as exc:
            logger.warning(f"[{ticker}] GDELT error (skipping): {exc}")
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
