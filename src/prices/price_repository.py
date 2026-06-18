"""
Phase 5: Data-access layer for ``stock_prices``.

``PriceRepository`` owns all SQL reads and writes for the ``stock_prices``
table.  It follows the same dual-dialect pattern as ``ArticleRepository``:

PostgreSQL (production)
    Uses ``INSERT … ON CONFLICT DO UPDATE`` for true bulk upsert in a
    single round-trip.

SQLite / other (tests)
    Falls back to a SELECT-then-INSERT/UPDATE loop — functionally
    identical and fully portable.

Usage
-----
    from src.prices.price_repository import PriceRepository

    with db.get_session() as session:
        repo = PriceRepository(session, dialect_name=db.engine.dialect.name)
        result = repo.upsert_prices(records)
        print(result)  # e.g. "inserted=252 updated=0 skipped=0"
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from sqlalchemy.orm import Session

from src.storage.models import StockPrice
from src.storage.repository import UpsertResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceRepository:
    """
    Data-access object for ``stock_prices``.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.  The caller owns transaction management
        (commit / rollback) via ``DatabaseManager.get_session()``.
    dialect_name : str
        SQLAlchemy dialect name: ``"postgresql"`` or ``"sqlite"``.
        Controls which upsert implementation is used.
        Default: ``"postgresql"``.
    """

    def __init__(
        self,
        session: Session,
        dialect_name: str = "postgresql",
    ) -> None:
        self._session = session
        self._dialect = dialect_name.lower()

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert_prices(
        self,
        records: list[dict[str, Any]],
    ) -> UpsertResult:
        """
        Upsert a batch of daily OHLCV price rows into ``stock_prices``.

        Deduplication is on ``(ticker, trading_date)``.  On conflict, all
        price and volume columns are refreshed to the latest values.

        Parameters
        ----------
        records : list[dict[str, Any]]
            Normalised rows from ``YFinancePriceClient.fetch_prices()``.
            Each dict must contain: ``ticker``, ``trading_date``,
            ``open_price``, ``high_price``, ``low_price``, ``close_price``,
            ``adjusted_close``, ``volume``.

        Returns
        -------
        UpsertResult
            ``inserted``, ``updated``, and ``skipped`` counts.
        """
        if not records:
            return UpsertResult()

        logger.debug(
            f"Upserting {len(records)} price rows (dialect={self._dialect})"
        )

        if self._dialect == "postgresql":
            result = self._pg_upsert_prices(records)
        else:
            result = self._generic_upsert_prices(records)

        logger.debug(f"upsert_prices done: {result}")
        return result

    def _pg_upsert_prices(
        self,
        records: list[dict[str, Any]],
    ) -> UpsertResult:
        """PostgreSQL path — single-round-trip bulk upsert."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: PLC0415

        stmt = pg_insert(StockPrice).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_prices_ticker_date",
            set_={
                "open_price":     stmt.excluded.open_price,
                "high_price":     stmt.excluded.high_price,
                "low_price":      stmt.excluded.low_price,
                "close_price":    stmt.excluded.close_price,
                "adjusted_close": stmt.excluded.adjusted_close,
                "volume":         stmt.excluded.volume,
            },
        )
        self._session.execute(stmt)
        return UpsertResult(inserted=len(records))

    def _generic_upsert_prices(
        self,
        records: list[dict[str, Any]],
    ) -> UpsertResult:
        """Generic path — SELECT + INSERT/UPDATE per record (SQLite-safe)."""
        inserted = 0
        updated  = 0

        for rec in records:
            existing: Optional[StockPrice] = (
                self._session.query(StockPrice)
                .filter_by(
                    ticker=rec["ticker"],
                    trading_date=rec["trading_date"],
                )
                .first()
            )
            if existing is None:
                self._session.add(StockPrice(**rec))
                inserted += 1
            else:
                existing.open_price     = rec.get("open_price",     existing.open_price)
                existing.high_price     = rec.get("high_price",     existing.high_price)
                existing.low_price      = rec.get("low_price",      existing.low_price)
                existing.close_price    = rec.get("close_price",    existing.close_price)
                existing.adjusted_close = rec.get("adjusted_close", existing.adjusted_close)
                existing.volume         = rec.get("volume",         existing.volume)
                updated += 1

        self._session.flush()
        return UpsertResult(inserted=inserted, updated=updated)

    # ── Queries ───────────────────────────────────────────────────────────────

    def count_prices(self, ticker: Optional[str] = None) -> int:
        """
        Return the total number of price rows.

        Parameters
        ----------
        ticker : str | None
            When provided, only rows for this ticker are counted.
        """
        q = self._session.query(StockPrice)
        if ticker:
            q = q.filter(StockPrice.ticker == ticker.upper())
        return q.count()

    def get_prices_by_ticker(self, ticker: str) -> list[StockPrice]:
        """
        Return all price rows for *ticker*, ordered oldest-first.

        Parameters
        ----------
        ticker : str
            Stock symbol (case-insensitive; normalised to uppercase).
        """
        return (
            self._session.query(StockPrice)
            .filter(StockPrice.ticker == ticker.upper())
            .order_by(StockPrice.trading_date.asc())
            .all()
        )

    def get_latest_price(self, ticker: str) -> Optional[StockPrice]:
        """
        Return the most recent price row for *ticker*, or ``None`` if no
        data exists for that ticker.
        """
        return (
            self._session.query(StockPrice)
            .filter(StockPrice.ticker == ticker.upper())
            .order_by(StockPrice.trading_date.desc())
            .first()
        )

    def get_price_range(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[StockPrice]:
        """
        Return price rows for *ticker* in the **inclusive** date range
        ``[start_date, end_date]``, ordered oldest-first.

        Parameters
        ----------
        ticker : str
            Stock symbol (case-insensitive).
        start_date : date
            Inclusive lower bound of the trading date range.
        end_date : date
            Inclusive upper bound of the trading date range.
        """
        return (
            self._session.query(StockPrice)
            .filter(
                StockPrice.ticker == ticker.upper(),
                StockPrice.trading_date >= start_date,
                StockPrice.trading_date <= end_date,
            )
            .order_by(StockPrice.trading_date.asc())
            .all()
        )
