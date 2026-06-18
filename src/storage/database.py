"""
Database connection management for financial-news-analytics.

``DatabaseManager`` wraps SQLAlchemy engine creation, connection pooling,
table DDL, and session lifecycle into a single, reusable object.

Usage
-----
    from src.storage.database import DatabaseManager

    db = DatabaseManager("postgresql://user:pass@localhost:5432/finews")
    db.verify_connection()   # raises DatabaseConnectionError on failure
    db.create_tables()       # CREATE TABLE IF NOT EXISTS

    with db.get_session() as session:
        session.add(some_orm_object)
        # commit is automatic on clean exit; rollback on exception

    db.dispose()  # release connection pool at shutdown
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.storage.models import Base
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class DatabaseError(Exception):
    """Base class for database-related errors."""


class DatabaseConnectionError(DatabaseError):
    """Raised when the database cannot be reached."""


class SchemaError(DatabaseError):
    """Raised when DDL operations fail."""


# ── Manager ───────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Manages the SQLAlchemy engine, session factory, and table DDL.

    Parameters
    ----------
    database_url : str
        SQLAlchemy-compatible connection URL.
        PostgreSQL example: ``postgresql://user:pass@localhost:5432/finews``
        SQLite in-memory  : ``sqlite:///:memory:``
    echo : bool
        If ``True``, all SQL statements are echoed to stdout. Useful for
        debugging. Default: ``False``.
    pool_size : int
        Number of persistent connections maintained in the pool.
        Ignored for SQLite. Default: ``5``.
    max_overflow : int
        Maximum connections allowed beyond ``pool_size`` during peak load.
        Ignored for SQLite. Default: ``10``.
    """

    def __init__(
        self,
        database_url: str,
        *,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        if not database_url:
            raise ValueError("database_url must not be empty")

        self._url = database_url
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    # ── Engine / session factory ──────────────────────────────────────────────

    @property
    def engine(self) -> Engine:
        """Lazy-initialised SQLAlchemy engine."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Lazy-initialised session factory bound to the engine."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._session_factory

    def _build_engine(self) -> Engine:
        is_sqlite = self._url.startswith("sqlite")

        kwargs: dict = {"echo": self._echo}
        if not is_sqlite:
            kwargs["pool_size"]     = self._pool_size
            kwargs["max_overflow"]  = self._max_overflow
            kwargs["pool_pre_ping"] = True   # drop stale connections before use
        else:
            # SQLite requires check_same_thread=False for test fixtures
            kwargs["connect_args"] = {"check_same_thread": False}

        logger.debug(f"Building database engine: {self._safe_url()}")
        return create_engine(self._url, **kwargs)

    # ── Connectivity ──────────────────────────────────────────────────────────

    def verify_connection(self) -> None:
        """
        Execute a lightweight ``SELECT 1`` to confirm the DB is reachable.

        Raises
        ------
        DatabaseConnectionError
            If the connection cannot be established.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Database connection verified: {self._safe_url()}")
        except Exception as exc:  # noqa: BLE001
            raise DatabaseConnectionError(
                f"Cannot connect to database at {self._safe_url()}: {exc}. "
                "Ensure PostgreSQL is running and DATABASE_URL in .env is correct."
            ) from exc

    # ── DDL ───────────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """
        Create all ORM-defined tables if they do not already exist.

        Uses ``checkfirst=True`` so re-running is always safe — no data is
        dropped or altered on existing databases.

        Raises
        ------
        SchemaError
            If DDL execution fails unexpectedly.
        """
        try:
            logger.info("Running CREATE TABLE IF NOT EXISTS for all models …")
            Base.metadata.create_all(self.engine, checkfirst=True)
            tables = sorted(Base.metadata.tables.keys())
            logger.info(f"Tables ready: {tables}")
        except Exception as exc:  # noqa: BLE001
            raise SchemaError(f"Failed to create tables: {exc}") from exc

    def drop_tables(self) -> None:
        """
        Drop all ORM-defined tables.

        **Destructive** — intended for test teardown only. Production code
        should use Alembic migrations instead.
        """
        logger.warning("Dropping all tables — all data will be lost!")
        Base.metadata.drop_all(self.engine)
        logger.info("All tables dropped.")

    # ── Session context manager ───────────────────────────────────────────────

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Yield a transactional database session.

        Commits automatically on clean exit. Rolls back and re-raises on any
        exception. The session is always closed in the ``finally`` block.

        Example
        -------
            with db.get_session() as session:
                session.add(article)
        """
        session: Session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def dispose(self) -> None:
        """Release all pooled connections. Call once at application shutdown."""
        if self._engine is not None:
            self._engine.dispose()
            logger.debug("Database connection pool disposed")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _safe_url(self) -> str:
        """Return the URL with any password redacted for safe logging."""
        try:
            from sqlalchemy.engine.url import make_url  # noqa: PLC0415
            return make_url(self._url).render_as_string(hide_password=True)
        except Exception:  # noqa: BLE001
            return "<db_url>"
