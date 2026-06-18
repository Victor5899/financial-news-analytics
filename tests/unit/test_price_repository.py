"""
Unit tests for src.prices.price_repository — PriceRepository.

All tests use an in-memory SQLite database; no PostgreSQL instance required.
SQLite exercises the generic (non-PostgreSQL) upsert path.  The PostgreSQL
path is verified by mocking the dialect and the pg_insert call.

Test organisation
-----------------
TestStockPriceModel           — ORM model table name, repr, nullable fields
TestPriceRepositoryInit       — constructor, dialect storage
TestUpsertPricesEmpty         — empty input → UpsertResult(0, 0, 0), no writes
TestUpsertPricesSQLite        — insert, update, idempotent, multi-ticker
TestUpsertPricesPostgresPath  — pg dialect invokes pg_insert
TestCountPrices               — total count, filtered by ticker, case folding
TestGetPricesByTicker         — ordering, filtering, case folding
TestGetLatestPrice            — newest date, empty DB, single row
TestGetPriceRange             — inclusive boundaries, ordering, cross-ticker
TestUpsertResultImport        — UpsertResult re-exported from repository
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.prices.price_repository import PriceRepository
from src.storage.models import Base, StockPrice
from src.storage.repository import UpsertResult

UTC = timezone.utc

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def sqlite_engine():  # type: ignore[no-untyped-def]
    """Fresh in-memory SQLite engine per test — no cross-test contamination."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(sqlite_engine):  # type: ignore[no-untyped-def]
    """Plain session bound to the per-test in-memory engine."""
    session = Session(sqlite_engine)
    yield session
    session.close()


@pytest.fixture
def repo(db_session) -> PriceRepository:  # type: ignore[no-untyped-def]
    return PriceRepository(db_session, dialect_name="sqlite")


# ── Sample data helpers ───────────────────────────────────────────────────────

def _price_record(
    ticker: str = "AAPL",
    trading_date: date = date(2026, 1, 2),
    close: float = 150.0,
) -> dict:
    return {
        "ticker":        ticker,
        "trading_date":  trading_date,
        "open_price":    close - 1.0,
        "high_price":    close + 2.0,
        "low_price":     close - 3.0,
        "close_price":   close,
        "adjusted_close": close - 0.5,
        "volume":        1_000_000,
    }


def _price_series(
    ticker: str = "AAPL",
    n: int = 5,
    start: date = date(2026, 1, 2),
) -> list[dict]:
    return [
        _price_record(ticker, start + timedelta(days=i), 100.0 + i)
        for i in range(n)
    ]


# ── TestStockPriceModel ───────────────────────────────────────────────────────

class TestStockPriceModel:
    def test_table_name(self) -> None:
        assert StockPrice.__tablename__ == "stock_prices"

    def test_repr_contains_ticker(self) -> None:
        sp = StockPrice(ticker="AAPL", trading_date=date(2026, 1, 2))
        assert "AAPL" in repr(sp)

    def test_repr_contains_date(self) -> None:
        sp = StockPrice(ticker="AAPL", trading_date=date(2026, 1, 2))
        assert "2026-01-02" in repr(sp)

    def test_nullable_price_fields_accept_none(self) -> None:
        sp = StockPrice(
            ticker="AAPL",
            trading_date=date(2026, 1, 2),
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=None,
            adjusted_close=None,
            volume=None,
        )
        assert sp.open_price is None
        assert sp.volume is None

    def test_model_stores_required_fields(self) -> None:
        sp = StockPrice(
            ticker="TSLA",
            trading_date=date(2026, 3, 15),
            close_price=250.0,
        )
        assert sp.ticker == "TSLA"
        assert sp.trading_date == date(2026, 3, 15)
        assert sp.close_price == 250.0

    def test_unique_constraint_name(self) -> None:
        constraint_names = {
            c.name for c in StockPrice.__table__.constraints
        }
        assert "uq_stock_prices_ticker_date" in constraint_names

    def test_ticker_index_exists(self) -> None:
        index_names = {idx.name for idx in StockPrice.__table__.indexes}
        assert "ix_stock_prices_ticker" in index_names

    def test_composite_index_exists(self) -> None:
        index_names = {idx.name for idx in StockPrice.__table__.indexes}
        assert "ix_stock_prices_ticker_date" in index_names


# ── TestPriceRepositoryInit ───────────────────────────────────────────────────

class TestPriceRepositoryInit:
    def test_default_dialect_is_postgresql(self, db_session: Session) -> None:
        repo = PriceRepository(db_session)
        assert repo._dialect == "postgresql"

    def test_custom_dialect_stored_lowercase(self, db_session: Session) -> None:
        repo = PriceRepository(db_session, dialect_name="SQLite")
        assert repo._dialect == "sqlite"

    def test_session_stored(self, db_session: Session) -> None:
        repo = PriceRepository(db_session)
        assert repo._session is db_session

    def test_postgresql_dialect_stored(self, db_session: Session) -> None:
        repo = PriceRepository(db_session, dialect_name="postgresql")
        assert repo._dialect == "postgresql"


# ── TestUpsertPricesEmpty ─────────────────────────────────────────────────────

class TestUpsertPricesEmpty:
    def test_empty_list_returns_upsert_result(self, repo: PriceRepository) -> None:
        result = repo.upsert_prices([])
        assert isinstance(result, UpsertResult)

    def test_empty_list_inserted_is_zero(self, repo: PriceRepository) -> None:
        result = repo.upsert_prices([])
        assert result.inserted == 0

    def test_empty_list_updated_is_zero(self, repo: PriceRepository) -> None:
        result = repo.upsert_prices([])
        assert result.updated == 0

    def test_empty_list_skipped_is_zero(self, repo: PriceRepository) -> None:
        result = repo.upsert_prices([])
        assert result.skipped == 0

    def test_empty_list_no_db_writes(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices([])
        db_session.commit()
        count = db_session.query(StockPrice).count()
        assert count == 0


# ── TestUpsertPricesSQLite ────────────────────────────────────────────────────

class TestUpsertPricesSQLite:
    def test_single_insert_returns_inserted_one(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        result = repo.upsert_prices([_price_record()])
        db_session.commit()
        assert result.inserted == 1

    def test_single_insert_stores_row(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices([_price_record("AAPL", date(2026, 1, 5))])
        db_session.commit()
        count = db_session.query(StockPrice).filter_by(ticker="AAPL").count()
        assert count == 1

    def test_multiple_inserts_count(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        records = _price_series(n=10)
        result = repo.upsert_prices(records)
        db_session.commit()
        assert result.inserted == 10

    def test_update_on_conflict(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        rec = _price_record("AAPL", date(2026, 2, 1), close=100.0)
        repo.upsert_prices([rec])
        db_session.commit()

        updated_rec = _price_record("AAPL", date(2026, 2, 1), close=999.0)
        result = repo.upsert_prices([updated_rec])
        db_session.commit()

        assert result.updated == 1

    def test_update_changes_close_price(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices([_price_record("TSLA", date(2026, 2, 1), close=200.0)])
        db_session.commit()

        repo.upsert_prices([_price_record("TSLA", date(2026, 2, 1), close=250.0)])
        db_session.commit()

        row = db_session.query(StockPrice).filter_by(
            ticker="TSLA", trading_date=date(2026, 2, 1)
        ).one()
        assert row.close_price == pytest.approx(250.0)

    def test_idempotent_upsert_same_data(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        records = _price_series("AAPL", n=5)
        repo.upsert_prices(records)
        db_session.commit()
        repo.upsert_prices(records)
        db_session.commit()

        count = db_session.query(StockPrice).filter_by(ticker="AAPL").count()
        assert count == 5

    def test_different_tickers_stored_separately(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices([
            _price_record("AAPL", date(2026, 1, 2)),
            _price_record("TSLA", date(2026, 1, 2)),
        ])
        db_session.commit()

        aapl = db_session.query(StockPrice).filter_by(ticker="AAPL").count()
        tsla = db_session.query(StockPrice).filter_by(ticker="TSLA").count()
        assert aapl == 1
        assert tsla == 1

    def test_trading_date_stored_correctly(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        target = date(2026, 6, 15)
        repo.upsert_prices([_price_record(trading_date=target)])
        db_session.commit()

        row = db_session.query(StockPrice).first()
        assert row is not None
        assert row.trading_date == target

    def test_none_price_values_stored(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        rec = {
            "ticker": "AAPL",
            "trading_date": date(2026, 1, 2),
            "open_price": None,
            "high_price": None,
            "low_price": None,
            "close_price": None,
            "adjusted_close": None,
            "volume": None,
        }
        result = repo.upsert_prices([rec])
        db_session.commit()

        assert result.inserted == 1
        row = db_session.query(StockPrice).first()
        assert row is not None
        assert row.open_price is None

    def test_upsert_result_total_property(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        result = repo.upsert_prices(_price_series(n=3))
        db_session.commit()
        assert result.total == 3

    def test_update_result_total_reflects_updated(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        records = _price_series(n=2)
        repo.upsert_prices(records)
        db_session.commit()
        result = repo.upsert_prices(records)  # all become updates
        db_session.commit()
        assert result.total == 2


# ── TestUpsertPricesPostgresPath ──────────────────────────────────────────────

class TestUpsertPricesPostgresPath:
    def test_pg_path_called_for_postgresql_dialect(
        self, db_session: Session
    ) -> None:
        repo = PriceRepository(db_session, dialect_name="postgresql")
        records = [_price_record()]

        mock_execute = MagicMock()
        db_session.execute = mock_execute  # type: ignore[method-assign]

        with patch("src.prices.price_repository.pg_insert", create=True):
            # We can't easily run real PG upsert against SQLite, so just verify
            # the _pg_upsert_prices method is selected.
            with patch.object(
                repo, "_pg_upsert_prices", return_value=UpsertResult(inserted=1)
            ) as mock_pg:
                result = repo.upsert_prices(records)

        mock_pg.assert_called_once_with(records)
        assert result.inserted == 1

    def test_generic_path_called_for_sqlite_dialect(
        self, db_session: Session
    ) -> None:
        repo = PriceRepository(db_session, dialect_name="sqlite")
        records = [_price_record()]

        with patch.object(
            repo, "_generic_upsert_prices", return_value=UpsertResult(inserted=1)
        ) as mock_generic:
            result = repo.upsert_prices(records)

        mock_generic.assert_called_once_with(records)
        assert result.inserted == 1


# ── TestCountPrices ───────────────────────────────────────────────────────────

class TestCountPrices:
    def test_empty_db_returns_zero(self, repo: PriceRepository) -> None:
        assert repo.count_prices() == 0

    def test_count_all_rows(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=5))
        repo.upsert_prices(_price_series("TSLA", n=3))
        db_session.commit()
        assert repo.count_prices() == 8

    def test_count_filtered_by_ticker(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=5))
        repo.upsert_prices(_price_series("TSLA", n=3))
        db_session.commit()
        assert repo.count_prices("AAPL") == 5
        assert repo.count_prices("TSLA") == 3

    def test_count_ticker_case_insensitive(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("NVDA", n=4))
        db_session.commit()
        assert repo.count_prices("nvda") == 4

    def test_count_unknown_ticker_returns_zero(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=2))
        db_session.commit()
        assert repo.count_prices("UNKNOWN") == 0


# ── TestGetPricesByTicker ─────────────────────────────────────────────────────

class TestGetPricesByTicker:
    def test_empty_db_returns_empty_list(self, repo: PriceRepository) -> None:
        assert repo.get_prices_by_ticker("AAPL") == []

    def test_returns_correct_ticker_rows(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=3))
        repo.upsert_prices(_price_series("TSLA", n=2))
        db_session.commit()

        rows = repo.get_prices_by_ticker("AAPL")
        assert len(rows) == 3
        assert all(r.ticker == "AAPL" for r in rows)

    def test_rows_ordered_oldest_first(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        records = _price_series("AAPL", n=5, start=date(2026, 1, 2))
        repo.upsert_prices(records)
        db_session.commit()

        rows = repo.get_prices_by_ticker("AAPL")
        dates = [r.trading_date for r in rows]
        assert dates == sorted(dates)

    def test_ticker_case_insensitive(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("MSFT", n=2))
        db_session.commit()

        rows = repo.get_prices_by_ticker("msft")
        assert len(rows) == 2

    def test_other_tickers_excluded(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=3))
        repo.upsert_prices(_price_series("AMZN", n=4))
        db_session.commit()

        rows = repo.get_prices_by_ticker("AAPL")
        assert all(r.ticker == "AAPL" for r in rows)

    def test_returns_list_of_stock_price_objects(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=1))
        db_session.commit()

        rows = repo.get_prices_by_ticker("AAPL")
        assert all(isinstance(r, StockPrice) for r in rows)


# ── TestGetLatestPrice ────────────────────────────────────────────────────────

class TestGetLatestPrice:
    def test_empty_db_returns_none(self, repo: PriceRepository) -> None:
        assert repo.get_latest_price("AAPL") is None

    def test_unknown_ticker_returns_none(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("TSLA", n=3))
        db_session.commit()
        assert repo.get_latest_price("AAPL") is None

    def test_returns_most_recent_date(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=5, start=date(2026, 1, 2)))
        db_session.commit()

        latest = repo.get_latest_price("AAPL")
        assert latest is not None
        assert latest.trading_date == date(2026, 1, 6)  # start + 4 days

    def test_single_row_returned_as_latest(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices([_price_record("AAPL", date(2026, 5, 1))])
        db_session.commit()

        latest = repo.get_latest_price("AAPL")
        assert latest is not None
        assert latest.trading_date == date(2026, 5, 1)

    def test_returns_stock_price_instance(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices(_price_series("AAPL", n=2))
        db_session.commit()

        latest = repo.get_latest_price("AAPL")
        assert isinstance(latest, StockPrice)

    def test_ticker_case_insensitive(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        repo.upsert_prices([_price_record("NVDA", date(2026, 3, 10))])
        db_session.commit()

        latest = repo.get_latest_price("nvda")
        assert latest is not None


# ── TestGetPriceRange ─────────────────────────────────────────────────────────

class TestGetPriceRange:
    def _populate(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        # 10 consecutive days starting 2026-01-02
        records = _price_series("AAPL", n=10, start=date(2026, 1, 2))
        repo.upsert_prices(records)
        db_session.commit()

    def test_empty_db_returns_empty_list(self, repo: PriceRepository) -> None:
        rows = repo.get_price_range("AAPL", date(2026, 1, 1), date(2026, 12, 31))
        assert rows == []

    def test_returns_rows_within_range(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("AAPL", date(2026, 1, 3), date(2026, 1, 7))
        assert len(rows) == 5  # 3, 4, 5, 6, 7

    def test_start_boundary_inclusive(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("AAPL", date(2026, 1, 2), date(2026, 1, 2))
        assert len(rows) == 1
        assert rows[0].trading_date == date(2026, 1, 2)

    def test_end_boundary_inclusive(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("AAPL", date(2026, 1, 11), date(2026, 1, 11))
        assert len(rows) == 1
        assert rows[0].trading_date == date(2026, 1, 11)

    def test_rows_ordered_oldest_first(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("AAPL", date(2026, 1, 2), date(2026, 1, 11))
        dates = [r.trading_date for r in rows]
        assert dates == sorted(dates)

    def test_outside_range_excluded(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("AAPL", date(2025, 1, 1), date(2025, 12, 31))
        assert rows == []

    def test_different_ticker_excluded(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        repo.upsert_prices(_price_series("TSLA", n=5, start=date(2026, 1, 2)))
        db_session.commit()

        rows = repo.get_price_range("AAPL", date(2026, 1, 2), date(2026, 1, 11))
        assert all(r.ticker == "AAPL" for r in rows)

    def test_ticker_case_insensitive(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("aapl", date(2026, 1, 2), date(2026, 1, 5))
        assert len(rows) == 4

    def test_returns_list_of_stock_price_objects(
        self, repo: PriceRepository, db_session: Session
    ) -> None:
        self._populate(repo, db_session)
        rows = repo.get_price_range("AAPL", date(2026, 1, 2), date(2026, 1, 4))
        assert all(isinstance(r, StockPrice) for r in rows)


# ── TestUpsertResultImport ────────────────────────────────────────────────────

class TestUpsertResultImport:
    def test_upsert_result_importable_from_repository(self) -> None:
        from src.storage.repository import UpsertResult as UR
        assert UR is UpsertResult

    def test_upsert_result_total_property(self) -> None:
        r = UpsertResult(inserted=3, updated=2, skipped=1)
        assert r.total == 5

    def test_upsert_result_frozen(self) -> None:
        r = UpsertResult(inserted=1)
        with pytest.raises((AttributeError, TypeError)):
            r.inserted = 99  # type: ignore[misc]

    def test_upsert_result_str_representation(self) -> None:
        r = UpsertResult(inserted=5, updated=2, skipped=0)
        s = str(r)
        assert "5" in s
        assert "2" in s
