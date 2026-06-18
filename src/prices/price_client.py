"""
Phase 5: Yahoo Finance price client.

``YFinancePriceClient`` downloads historical daily OHLCV data from Yahoo
Finance via the ``yfinance`` library and returns normalised records ready
for storage in PostgreSQL.

Exception hierarchy
-------------------
PriceIngestionError
    PriceFetchError        — network / yfinance errors during download
    PriceValidationError   — bad inputs or empty / malformed results

Usage
-----
    from src.prices.price_client import YFinancePriceClient

    client = YFinancePriceClient()
    records = client.fetch_prices("AAPL", lookback_days=365)

    all_records = client.fetch_multiple_tickers(
        ["AAPL", "TSLA", "NVDA"],
        lookback_days=365,
    )
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class PriceIngestionError(Exception):
    """Base class for all price-ingestion errors."""


class PriceFetchError(PriceIngestionError):
    """Raised when yfinance fails to download data (network / API error)."""


class PriceValidationError(PriceIngestionError):
    """Raised when inputs are invalid or the fetched data is unusable."""


# ── Client ────────────────────────────────────────────────────────────────────

class YFinancePriceClient:
    """
    Yahoo Finance price client backed by ``yfinance``.

    Downloads daily OHLCV data for one or more ticker symbols and returns
    normalised, database-ready records.

    Parameters
    ----------
    request_timeout : int
        Seconds before an HTTP request times out. Default: 30.
    """

    _REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
    _ADJ_CLOSE_CANDIDATES: tuple[str, ...] = ("Adj Close", "Adj. Close")

    def __init__(self, request_timeout: int = 30) -> None:
        self._timeout = request_timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        ticker: str,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        lookback_days: int = 365,
    ) -> list[dict[str, Any]]:
        """
        Fetch daily OHLCV data for a single ticker.

        Parameters
        ----------
        ticker : str
            Stock symbol (e.g. ``"AAPL"``).  Case-insensitive; normalised
            to uppercase internally.
        start_date : date | None
            Inclusive start of the date range.  If ``None``, derived from
            ``lookback_days`` counting back from ``end_date``.
        end_date : date | None
            End of the date range (yfinance ``end`` is exclusive, but the
            last trading day on or before this date will be included).
            Defaults to today when ``None``.
        lookback_days : int
            Number of calendar days to look back from ``end_date`` when
            ``start_date`` is not supplied.  Default: 365.

        Returns
        -------
        list[dict[str, Any]]
            One dict per trading day with keys:
            ``ticker``, ``trading_date``, ``open_price``, ``high_price``,
            ``low_price``, ``close_price``, ``adjusted_close``, ``volume``.

        Raises
        ------
        PriceValidationError
            Empty ticker, invalid date range, or no rows returned.
        PriceFetchError
            ``yfinance`` raises an unexpected exception during download.
        """
        ticker = ticker.strip().upper()
        if not ticker:
            raise PriceValidationError("ticker must not be empty")

        resolved_end   = end_date or date.today()
        resolved_start = start_date if start_date is not None else (
            resolved_end - timedelta(days=lookback_days)
        )

        if resolved_start >= resolved_end:
            raise PriceValidationError(
                f"start_date ({resolved_start}) must be strictly before "
                f"end_date ({resolved_end})"
            )

        logger.info(
            f"[{ticker}] Fetching daily prices "
            f"{resolved_start} → {resolved_end}"
        )

        try:
            yf_ticker = yf.Ticker(ticker)
            df: pd.DataFrame = yf_ticker.history(
                start=resolved_start.isoformat(),
                end=resolved_end.isoformat(),
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise PriceFetchError(
                f"[{ticker}] yfinance download failed: {exc}"
            ) from exc

        return self._normalise(ticker, df)

    def fetch_multiple_tickers(
        self,
        tickers: list[str],
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        lookback_days: int = 365,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch daily OHLCV data for multiple tickers.

        Individual ticker errors are logged and skipped so the caller always
        receives a partial result rather than a hard failure.

        Parameters
        ----------
        tickers : list[str]
            Stock symbols to fetch.  Must not be empty.
        start_date, end_date, lookback_days :
            Forwarded verbatim to :meth:`fetch_prices` for each ticker.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping ``TICKER → list of OHLCV dicts``.
            Tickers that errored are absent from the mapping.

        Raises
        ------
        PriceValidationError
            If ``tickers`` is an empty list.
        """
        if not tickers:
            raise PriceValidationError("tickers list must not be empty")

        results: dict[str, list[dict[str, Any]]] = {}

        for raw in tickers:
            symbol = raw.strip().upper()
            if not symbol:
                logger.warning("Skipping empty ticker symbol in tickers list")
                continue
            try:
                rows = self.fetch_prices(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    lookback_days=lookback_days,
                )
                results[symbol] = rows
                logger.info(f"[{symbol}] Fetched {len(rows)} rows")
            except (PriceFetchError, PriceValidationError) as exc:
                logger.error(f"[{symbol}] Skipped — {exc}")

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalise(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Convert a raw yfinance DataFrame into a list of normalised record dicts.

        Raises
        ------
        PriceValidationError
            DataFrame is empty or missing required OHLCV columns.
        """
        if df is None or df.empty:
            raise PriceValidationError(
                f"[{ticker}] No data returned from Yahoo Finance. "
                "The ticker may be invalid, delisted, or the date range "
                "falls outside available history."
            )

        # yfinance may return MultiIndex columns when downloading multiple tickers
        # at once via yf.download(); flatten to single level for safety.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        missing = [c for c in self._REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise PriceValidationError(
                f"[{ticker}] yfinance response missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        adj_col: Optional[str] = next(
            (c for c in self._ADJ_CLOSE_CANDIDATES if c in df.columns),
            None,
        )

        records: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            trading_dt: date
            if isinstance(idx, pd.Timestamp):
                trading_dt = idx.date()
            else:
                try:
                    trading_dt = pd.Timestamp(idx).date()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        f"[{ticker}] Could not parse index value {idx!r} as date — skipping row"
                    )
                    continue

            adj_value: Optional[float]
            if adj_col is not None and not _is_nan(row[adj_col]):
                adj_value = float(row[adj_col])
            elif not _is_nan(row["Close"]):
                adj_value = float(row["Close"])
            else:
                adj_value = None

            records.append({
                "ticker":        ticker,
                "trading_date":  trading_dt,
                "open_price":    float(row["Open"])   if not _is_nan(row["Open"])   else None,
                "high_price":    float(row["High"])   if not _is_nan(row["High"])   else None,
                "low_price":     float(row["Low"])    if not _is_nan(row["Low"])    else None,
                "close_price":   float(row["Close"])  if not _is_nan(row["Close"])  else None,
                "adjusted_close": adj_value,
                "volume":        int(row["Volume"])   if not _is_nan(row["Volume"]) else None,
            })

        logger.debug(f"[{ticker}] Normalised {len(records)} rows")
        return records


# ── Module-level helpers ──────────────────────────────────────────────────────

def _is_nan(value: Any) -> bool:
    """Return ``True`` if *value* is float NaN, ``None``, or pandas NA."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
