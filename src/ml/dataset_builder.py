"""
Phase 6: ML Dataset Builder for financial-news-analytics.

``MLDatasetBuilder`` combines Phase 4 engineered features with future stock
price movements to produce a supervised-learning dataset with binary and
multi-class labels.

Label groups
------------
Future close : next 1, 3, 5, and 7 trading days' closing prices
Return        : (future_close - close_today) / close_today
Binary labels : 1 if return > 0 else 0 (per horizon)
Direction     : BUY (return_5d > 0.02), SELL (return_5d < -0.02), HOLD otherwise

Usage
-----
    from src.ml.dataset_builder import MLDatasetBuilder
    from datetime import date
    from pathlib import Path

    builder = MLDatasetBuilder()
    df = builder.run(
        features_path=Path("data/features/feature_dataset_2026-06-16.csv"),
        target_date=date(2026, 6, 16),
        output_dir=Path("data/ml/"),
    )

Output column order
-------------------
    <all Phase 4 feature columns>,
    future_close_1d, future_close_3d, future_close_5d, future_close_7d,
    return_1d, return_3d, return_5d, return_7d,
    label_up_1d, label_up_3d, label_up_5d, label_up_7d,
    label_direction
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from src.storage.database import DatabaseManager
from src.storage.models import StockPrice
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Label column names ────────────────────────────────────────────────────────

LOOKAHEAD_DAYS: tuple[int, ...] = (1, 3, 5, 7)

LABEL_COLUMNS: list[str] = [
    "future_close_1d",
    "future_close_3d",
    "future_close_5d",
    "future_close_7d",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_7d",
    "label_up_1d",
    "label_up_3d",
    "label_up_5d",
    "label_up_7d",
    "label_direction",
]

BUY_THRESHOLD: float  = 0.02
SELL_THRESHOLD: float = -0.02


# ── Exceptions ────────────────────────────────────────────────────────────────

class MLDatasetError(Exception):
    """Base exception for all ML dataset errors."""


class DataLoadError(MLDatasetError):
    """Raised when feature or price data cannot be loaded."""


class LabelGenerationError(MLDatasetError):
    """Raised when label computation fails."""


# ── Private computation helpers ───────────────────────────────────────────────

def _compute_future_closes(
    ticker: str,
    target_date: date,
    price_map: dict[date, float],
    sorted_dates: list[date],
) -> dict[str, float | None]:
    """
    Look up N-trading-day-ahead closing prices for one ticker on target_date.

    Uses trading-day indexing — skips weekends and holidays automatically
    because only dates present in ``price_map`` are considered.

    Parameters
    ----------
    ticker : str
        Ticker symbol (used only for debug logging).
    target_date : date
        The reference trading date.
    price_map : dict[date, float]
        ``{trading_date: close_price}`` for this ticker.
    sorted_dates : list[date]
        Ascending list of trading dates for this ticker.

    Returns
    -------
    dict
        Keys ``future_close_1d``, ``future_close_3d``, ``future_close_5d``,
        ``future_close_7d``.  Value is ``None`` when insufficient future data
        is available.
    """
    if target_date not in price_map:
        return {f"future_close_{n}d": None for n in LOOKAHEAD_DAYS}

    try:
        idx = sorted_dates.index(target_date)
    except ValueError:
        return {f"future_close_{n}d": None for n in LOOKAHEAD_DAYS}

    result: dict[str, float | None] = {}
    for n in LOOKAHEAD_DAYS:
        future_idx = idx + n
        if future_idx < len(sorted_dates):
            future_date = sorted_dates[future_idx]
            result[f"future_close_{n}d"] = price_map.get(future_date)
        else:
            logger.debug(
                f"[{ticker}] Insufficient future trading days for "
                f"{n}d lookahead on {target_date}"
            )
            result[f"future_close_{n}d"] = None
    return result


def _compute_returns(
    close_today: float,
    future_closes: dict[str, float | None],
) -> dict[str, float | None]:
    """
    Compute percentage returns for each lookahead horizon.

    Formula: ``(future_close - close_today) / close_today``

    Parameters
    ----------
    close_today : float
        Closing price on the target date.
    future_closes : dict
        Output of :func:`_compute_future_closes`.

    Returns
    -------
    dict
        Keys ``return_1d`` … ``return_7d``.
        ``None`` when the future close is unavailable or ``close_today == 0``.
    """
    result: dict[str, float | None] = {}
    for n in LOOKAHEAD_DAYS:
        fc = future_closes.get(f"future_close_{n}d")
        if fc is None or close_today == 0:
            result[f"return_{n}d"] = None
        else:
            result[f"return_{n}d"] = round((fc - close_today) / close_today, 8)
    return result


def _compute_binary_labels(
    returns: dict[str, float | None],
) -> dict[str, int | None]:
    """
    Assign binary up/down labels for each return horizon.

    Rule: ``1`` if return > 0, ``0`` otherwise.
    ``None`` when the return is unavailable.
    """
    result: dict[str, int | None] = {}
    for n in LOOKAHEAD_DAYS:
        ret = returns.get(f"return_{n}d")
        if ret is None:
            result[f"label_up_{n}d"] = None
        else:
            result[f"label_up_{n}d"] = 1 if ret > 0 else 0
    return result


def _compute_direction_label(return_5d: float | None) -> str | None:
    """
    Assign a multi-class direction label from the 5-day return.

    Rules
    -----
    - ``return_5d > 0.02``  → ``"BUY"``
    - ``return_5d < -0.02`` → ``"SELL"``
    - Otherwise             → ``"HOLD"``
    - ``None``              → ``None`` (missing data)
    """
    if return_5d is None:
        return None
    if return_5d > BUY_THRESHOLD:
        return "BUY"
    if return_5d < SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


# ── MLDatasetBuilder ──────────────────────────────────────────────────────────

class MLDatasetBuilder:
    """
    Builds a supervised ML dataset from Phase 4 features and stock prices.

    Follows the same orchestration pattern as ``FeatureEngineer``:
    thin public API, private computation helpers, lazy database init.

    Parameters
    ----------
    database_url : str | None
        SQLAlchemy connection URL.  Defaults to ``settings.database_url``.

    Raises
    ------
    DataLoadError
        If no database URL is configured.
    """

    def __init__(self, database_url: str | None = None) -> None:
        resolved = database_url or settings.database_url
        if not resolved:
            raise DataLoadError(
                "No database URL configured. "
                "Set DATABASE_URL in your .env file or pass database_url explicitly.\n"
                "  Example: DATABASE_URL=postgresql://user:pass@localhost:5432/financial_news"
            )
        self._database_url: str = resolved
        self._db: DatabaseManager | None = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_db(self) -> DatabaseManager:
        """Lazily initialise and cache the DatabaseManager."""
        if self._db is None:
            self._db = DatabaseManager(self._database_url)
        return self._db

    # ── Public API ────────────────────────────────────────────────────────────

    def load_features(self, features_path: Path) -> pd.DataFrame:
        """
        Load the Phase 4 feature dataset from a CSV file.

        Parameters
        ----------
        features_path : Path
            Path to the feature CSV produced by ``generate_features.py``
            (e.g. ``data/features/feature_dataset_2026-06-16.csv``).

        Returns
        -------
        pd.DataFrame
            Feature rows with at least ``ticker`` and ``date`` columns.
            The ``date`` column is normalised to Python ``datetime.date`` objects.

        Raises
        ------
        DataLoadError
            If the file does not exist, cannot be parsed, is empty, or is
            missing required columns.
        """
        if not features_path.exists():
            raise DataLoadError(
                f"Feature file not found: {features_path}. "
                "Run scripts/generate_features.py first to produce it."
            )

        try:
            df = pd.read_csv(features_path)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"Failed to read feature CSV {features_path}: {exc}"
            ) from exc

        if df.empty:
            raise DataLoadError(
                f"Feature file is empty: {features_path}. "
                "Ensure generate_features.py produced valid output."
            )

        if "ticker" not in df.columns or "date" not in df.columns:
            raise DataLoadError(
                f"Feature file is missing required columns (ticker, date): {features_path}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        logger.info(
            f"Loaded {len(df)} feature rows "
            f"({df['ticker'].nunique()} tickers) from {features_path.name}"
        )
        return df

    def load_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load stock price rows from the database for label computation.

        Fetches ``(ticker, trading_date, close_price)`` in ``[start_date, end_date]``
        for the given tickers.  The window must extend at least
        ``max(LOOKAHEAD_DAYS)`` trading days beyond the feature target date.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols (case-insensitive; normalised to uppercase).
        start_date : date
            Inclusive lower bound for ``trading_date``.
        end_date : date
            Inclusive upper bound for ``trading_date``.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``trading_date``, ``close_price``.
            Returns an empty DataFrame when no prices are found.

        Raises
        ------
        DataLoadError
            On any database connectivity or query failure.
        """
        if not tickers:
            logger.warning("load_prices: no tickers requested — returning empty DataFrame")
            return pd.DataFrame(columns=["ticker", "trading_date", "close_price"])

        logger.debug(
            f"load_prices: tickers={tickers}  window={start_date} → {end_date}"
        )

        try:
            db = self._get_db()
            stmt = (
                select(
                    StockPrice.ticker.label("ticker"),
                    StockPrice.trading_date.label("trading_date"),
                    StockPrice.close_price.label("close_price"),
                )
                .where(StockPrice.ticker.in_([t.upper() for t in tickers]))
                .where(StockPrice.trading_date >= start_date)
                .where(StockPrice.trading_date <= end_date)
                .order_by(StockPrice.ticker, StockPrice.trading_date)
            )
            with db.engine.connect() as conn:
                df = pd.read_sql(stmt, conn)
        except MLDatasetError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"Failed to load prices from database: {exc}"
            ) from exc

        if df.empty:
            logger.warning(
                f"No price data found for tickers={tickers} "
                f"in window {start_date} → {end_date}"
            )
            return df

        df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date

        logger.info(
            f"Loaded {len(df)} price rows across "
            f"{df['ticker'].nunique()} ticker(s) "
            f"({start_date} → {end_date})"
        )
        return df

    def generate_labels(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute future-price labels for every (ticker, date) in features_df.

        For each feature row the method:

        1. Locates the closing price on the feature date.  If unavailable the
           row is **skipped** (warning logged) — there is nothing to anchor
           returns to.
        2. Looks up closing prices N trading days ahead (1, 3, 5, 7).
           Horizons whose future date is not yet in ``stock_prices`` are set
           to ``NULL``; the row is **not** skipped.
        3. Computes percentage returns for available horizons only
           (``NULL`` when the corresponding future close is ``NULL``).
        4. Assigns binary up/down labels for available returns
           (``NULL`` when the return is ``NULL``).
        5. Assigns a BUY/HOLD/SELL direction label from ``return_5d`` when
           available, otherwise ``NULL``.

        This design allows the dataset to be produced even when only a subset
        of lookahead data exists (e.g. only 1-day prices are available).

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of :meth:`load_features`.
        prices_df : pd.DataFrame
            Output of :meth:`load_prices`.  Must contain ``ticker``,
            ``trading_date``, and ``close_price`` columns.

        Returns
        -------
        pd.DataFrame
            Feature rows with all label columns appended.  Unavailable
            future-horizon columns contain ``NaN``.

        Raises
        ------
        LabelGenerationError
            When ``features_df`` or ``prices_df`` is empty on entry, or when
            every feature row is skipped due to a missing current-day close.
        """
        if features_df.empty:
            raise LabelGenerationError(
                "Cannot generate labels from an empty features DataFrame."
            )
        if prices_df.empty:
            raise LabelGenerationError(
                "Cannot generate labels: price DataFrame is empty. "
                "Ensure the stock_prices table is populated for the required tickers."
            )

        # Build per-ticker price index: {ticker: {date: close_price}}
        price_index: dict[str, dict[date, float]] = {}
        sorted_dates_index: dict[str, list[date]] = {}

        for ticker_val, group in prices_df.groupby("ticker"):
            valid = group.dropna(subset=["close_price"])
            if valid.empty:
                continue
            pm: dict[date, float] = {
                td: float(cp)
                for td, cp in zip(valid["trading_date"], valid["close_price"])
            }
            key = str(ticker_val)
            price_index[key] = pm
            sorted_dates_index[key] = sorted(pm.keys())

        labeled_rows: list[dict[str, Any]] = []
        skipped_count = 0

        for _, feat_row in features_df.iterrows():
            ticker = str(feat_row["ticker"])
            target_date = feat_row["date"]

            # Normalise date type robustly
            if isinstance(target_date, str):
                target_date = date.fromisoformat(target_date)
            elif hasattr(target_date, "date") and callable(target_date.date):
                target_date = target_date.date()

            pm = price_index.get(ticker, {})
            sorted_dates = sorted_dates_index.get(ticker, [])

            close_today = pm.get(target_date)
            if close_today is None:
                logger.warning(
                    f"[{ticker}] No close price for {target_date} "
                    "— skipping label generation for this row"
                )
                skipped_count += 1
                continue

            future_closes = _compute_future_closes(ticker, target_date, pm, sorted_dates)

            missing_closes = [k for k, v in future_closes.items() if v is None]
            if missing_closes:
                logger.debug(
                    f"[{ticker}] Partial future data on {target_date}: "
                    f"{missing_closes} will be NULL in the dataset"
                )

            returns = _compute_returns(close_today, future_closes)
            binary_labels = _compute_binary_labels(returns)
            direction = _compute_direction_label(returns.get("return_5d"))

            row: dict[str, Any] = feat_row.to_dict()
            row.update(future_closes)
            row.update(returns)
            row.update(binary_labels)
            row["label_direction"] = direction
            labeled_rows.append(row)

        if not labeled_rows:
            raise LabelGenerationError(
                f"No labeled rows generated — {skipped_count} row(s) skipped because "
                "the current-day close price was unavailable for every feature row. "
                "Ensure stock_prices contains data for the feature target date."
            )

        if skipped_count:
            logger.warning(
                f"generate_labels: {skipped_count} row(s) skipped "
                "(no current-day close price found)"
            )

        result_df = pd.DataFrame(labeled_rows)
        logger.info(
            f"Generated labels for {len(result_df)} row(s) "
            f"({skipped_count} skipped) across "
            f"{result_df['ticker'].nunique()} ticker(s)"
        )
        return result_df

    def build_dataset(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce canonical column ordering: features first, labels last.

        Parameters
        ----------
        labeled_df : pd.DataFrame
            Output of :meth:`generate_labels`.

        Returns
        -------
        pd.DataFrame
            Columns ordered: Phase 4 feature columns … label columns.
        """
        feature_cols = [c for c in labeled_df.columns if c not in LABEL_COLUMNS]
        final_cols = feature_cols + LABEL_COLUMNS
        available = [c for c in final_cols if c in labeled_df.columns]
        return labeled_df[available]

    def save_dataset(
        self,
        dataset_df: pd.DataFrame,
        output_dir: Path,
        date_tag: str,
    ) -> Path:
        """
        Save the ML dataset to a dated CSV file.

        The output filename follows the convention
        ``ml_dataset_<date_tag>.csv``.  The directory is created (including
        any missing parents) if it does not already exist.

        Parameters
        ----------
        dataset_df : pd.DataFrame
            Output of :meth:`build_dataset`.
        output_dir : Path
            Target directory.  Created automatically if absent.
        date_tag : str
            Date string appended to the filename (e.g. ``"2026-06-16"``).

        Returns
        -------
        Path
            Absolute path of the written CSV.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"ml_dataset_{date_tag}.csv"
        dataset_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(dataset_df)} ML dataset row(s) → {out_path}")
        return out_path

    def run(
        self,
        features_path: Path,
        target_date: date | None = None,
        output_dir: Path | None = None,
        lookahead_days: int = 14,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """
        Full pipeline: load features → load prices → generate labels → save.

        Parameters
        ----------
        features_path : Path
            Path to the Phase 4 feature CSV.
        target_date : date | None
            Feature date.  When ``None``, inferred from the feature CSV
            (the maximum date present in the ``date`` column).
        output_dir : Path | None
            Where to save the ML dataset CSV.  When ``None`` the dataset is
            returned but not written to disk.
        lookahead_days : int
            Extra calendar days to load from ``stock_prices`` beyond
            ``target_date``.  Default ``14`` safely covers 7 trading days
            for any calendar-week layout.
        dry_run : bool
            When ``True``, skip all I/O and return an empty DataFrame.

        Returns
        -------
        pd.DataFrame
            The final ML dataset, or an empty DataFrame when ``dry_run=True``.

        Raises
        ------
        DataLoadError
            On feature file or database load failure.
        LabelGenerationError
            When no labeled rows can be produced.
        """
        if dry_run:
            logger.info("Dry-run: skipping data load and label generation.")
            return pd.DataFrame()

        features_df = self.load_features(features_path)

        if target_date is None:
            dates_in_csv = features_df["date"].dropna().unique()
            if len(dates_in_csv) == 0:
                raise DataLoadError(
                    "Cannot infer target_date: feature CSV has no valid dates."
                )
            target_date = max(dates_in_csv)
            logger.info(f"Inferred target_date={target_date} from feature CSV")

        tickers = features_df["ticker"].str.upper().unique().tolist()
        price_end = target_date + timedelta(days=lookahead_days)

        prices_df = self.load_prices(
            tickers=tickers,
            start_date=target_date,
            end_date=price_end,
        )

        labeled_df = self.generate_labels(features_df, prices_df)
        dataset_df = self.build_dataset(labeled_df)

        if output_dir is not None:
            date_tag = target_date.strftime("%Y-%m-%d")
            self.save_dataset(dataset_df, output_dir, date_tag)

        return dataset_df

    def dispose(self) -> None:
        """Release all pooled database connections at application shutdown."""
        if self._db is not None:
            self._db.dispose()
            self._db = None
            logger.debug("MLDatasetBuilder: database connections disposed")
