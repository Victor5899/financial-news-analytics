"""
Phase 7: XGBoost model trainer for BUY / HOLD / SELL stock direction classification.

``ModelTrainer`` follows the same thin-orchestration pattern used by
``FeatureEngineer`` and ``MLDatasetBuilder`` in earlier phases: a single class
with named, chainable methods and private helpers.

Pipeline
--------
    trainer = ModelTrainer(dataset_path=..., model_out=..., ...)
    trainer.load_dataset()       # read ML CSV, drop unlabelled rows
    trainer.prepare_features()   # detect features, encode target, split
    trainer.train()              # fit XGBClassifier
    trainer.evaluate()           # compute full metrics suite
    trainer.save_model()         # persist model + metrics + importance

Usage
-----
    from src.model.trainer import ModelTrainer
    from pathlib import Path

    trainer = ModelTrainer(
        dataset_path=Path("data/ml/ml_dataset_2025-01-01_2026-06-17.csv"),
        model_out=Path("artifacts/models/xgboost_direction_model.joblib"),
        metrics_out=Path("artifacts/metrics/xgboost_metrics.json"),
        importance_out=Path("artifacts/plots/feature_importance.png"),
    )
    trainer.load_dataset().prepare_features().train().evaluate()
    trainer.save_model()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.model.evaluator import ModelEvaluator
from src.model.feature_importance import (
    compute_feature_importance,
    plot_feature_importance,
    save_importance_csv,
)
from src.model.model_io import save_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

#: Target column produced by Phase 6.
LABEL_COL: str = "label_direction"

#: Canonical direction classes — must match Phase 6 label generation.
DIRECTION_CLASSES: list[str] = ["BUY", "HOLD", "SELL"]

#: Exact column names that are metadata, not features.
_EXCLUDE_EXACT: frozenset[str] = frozenset({"ticker", "date"})

#: Column-name prefixes that mark label/lookahead columns, not features.
_EXCLUDE_PREFIXES: tuple[str, ...] = ("future_close_", "return_", "label_")

#: Default XGBoost hyperparameters — sensible starting point, no tuning.
_DEFAULT_PARAMS: dict[str, Any] = {
    "objective":        "multi:softprob",
    "num_class":        3,
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "mlogloss",
}


# ── Exceptions ────────────────────────────────────────────────────────────────

class ModelTrainingError(Exception):
    """Base exception for all model training errors."""


class DataPreparationError(ModelTrainingError):
    """Raised when dataset loading or feature preparation fails."""


class ModelNotTrainedError(ModelTrainingError):
    """Raised when a post-training method is called before ``train()``."""


# ── ModelTrainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """XGBoost direction classifier trainer.

    Follows a builder-style API: each public method returns ``self`` so calls
    can be chained.  Internal state is stored as private attributes and
    exposed through read-only properties.

    Parameters
    ----------
    dataset_path : Path
        Path to an ML dataset CSV produced by Phase 6
        (``data/ml/ml_dataset_*.csv``).  Both single-date and multi-date
        (historical backfill) datasets are supported.
    model_out : Path
        Destination for the serialised model artifact (``.joblib``).
    metrics_out : Path
        Destination for the evaluation metrics JSON.
    importance_out : Path
        Destination for the feature importance PNG plot.
    random_seed : int
        Random seed for the train/test split and XGBoost.  Default ``42``.
    label_col : str
        Name of the direction label column.  Default ``"label_direction"``.
    test_size : float
        Fraction of data reserved for the held-out test set.  Default ``0.2``.
    """

    def __init__(
        self,
        dataset_path: Path,
        model_out: Path,
        metrics_out: Path,
        importance_out: Path,
        random_seed: int = 42,
        label_col: str = LABEL_COL,
        test_size: float = 0.2,
    ) -> None:
        self._dataset_path: Path = dataset_path
        self._model_out: Path = model_out
        self._metrics_out: Path = metrics_out
        self._importance_out: Path = importance_out
        self._random_seed: int = random_seed
        self._label_col: str = label_col
        self._test_size: float = test_size

        self._df: pd.DataFrame | None = None
        self._X_train: np.ndarray | None = None
        self._X_test:  np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_test:  np.ndarray | None = None
        self._feature_columns: list[str] = []
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._model: XGBClassifier | None = None
        self._metrics: dict[str, Any] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load_dataset(self) -> "ModelTrainer":
        """Load the ML dataset CSV and drop rows without a direction label.

        Auto-detects whether the dataset is single-date or a multi-date
        historical backfill from the number of unique dates in the ``date``
        column.

        Returns
        -------
        ModelTrainer
            ``self`` (chainable).

        Raises
        ------
        DataPreparationError
            If the file is missing, unreadable, empty, or has no labelled rows.
        """
        path = self._dataset_path
        if not path.exists():
            raise DataPreparationError(
                f"Dataset not found: {path}. "
                "Run scripts/build_ml_dataset.py first to produce it."
            )

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise DataPreparationError(
                f"Failed to read ML dataset {path}: {exc}"
            ) from exc

        if df.empty:
            raise DataPreparationError(f"ML dataset is empty: {path}")

        # Detect dataset mode for informational logging only.
        if "date" in df.columns:
            n_dates = df["date"].nunique()
            mode = "historical" if n_dates > 1 else "single-date"
            logger.debug(f"Dataset mode detected: {mode} ({n_dates} unique date(s))")
        else:
            mode = "unknown"

        # Drop rows without a direction label.
        original_len = len(df)
        df = df.dropna(subset=[self._label_col])
        dropped = original_len - len(df)
        if dropped:
            logger.info(
                f"Dropped {dropped} rows with null '{self._label_col}' "
                f"({original_len} → {len(df)} rows)"
            )

        if df.empty:
            raise DataPreparationError(
                f"No labelled rows remaining in {path} after dropping nulls. "
                "Ensure the Phase 6 dataset was built with sufficient future data."
            )

        self._df = df
        logger.info(
            f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns "
            f"[mode={mode}, label_col={self._label_col!r}]"
        )
        return self

    def prepare_features(self) -> "ModelTrainer":
        """Detect feature columns, encode the target, and split train/test.

        Feature columns are automatically detected as all columns that are not
        in the metadata exclusion set (``ticker``, ``date``), do not start
        with a label/lookahead prefix (``future_close_``, ``return_``,
        ``label_``), and are not the target column itself.

        Uses a stratified 80/20 split with the configured random seed.

        Returns
        -------
        ModelTrainer
            ``self`` (chainable).

        Raises
        ------
        DataPreparationError
            If :meth:`load_dataset` has not been called, no feature columns
            are detected, or the feature matrix cannot be converted to float.
        """
        if self._df is None:
            raise DataPreparationError("Call load_dataset() before prepare_features().")

        df = self._df

        feature_cols: list[str] = [
            col for col in df.columns
            if col not in _EXCLUDE_EXACT
            and col != self._label_col
            and not any(col.startswith(p) for p in _EXCLUDE_PREFIXES)
        ]

        if not feature_cols:
            raise DataPreparationError(
                "No feature columns detected after excluding metadata and label columns. "
                f"Columns in dataset: {list(df.columns)}"
            )

        self._feature_columns = feature_cols

        try:
            X = df[feature_cols].values.astype(float)
        except ValueError as exc:
            raise DataPreparationError(
                f"Feature matrix contains non-numeric values: {exc}"
            ) from exc

        y_raw: np.ndarray = df[self._label_col].values

        # Fit encoder on the known canonical classes for consistent ordering.
        self._label_encoder.fit(DIRECTION_CLASSES)
        y: np.ndarray = self._label_encoder.transform(y_raw)

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X,
            y,
            test_size=self._test_size,
            random_state=self._random_seed,
            stratify=y,
        )

        nan_count = int(np.isnan(X).sum())
        if nan_count:
            logger.warning(
                f"Feature matrix contains {nan_count} NaN value(s). "
                "XGBoost will handle them natively via missing-value splits."
            )

        logger.info(
            f"Features   : {len(feature_cols)} columns | "
            f"Train rows : {len(self._X_train)} | "
            f"Test rows  : {len(self._X_test)}"
        )
        return self

    def train(self, params: dict[str, Any] | None = None) -> "ModelTrainer":
        """Fit the XGBoost classifier on the training split.

        Parameters
        ----------
        params : dict | None
            Optional parameter overrides merged on top of :data:`_DEFAULT_PARAMS`.
            ``random_state`` is always set to :attr:`_random_seed`.

        Returns
        -------
        ModelTrainer
            ``self`` (chainable).

        Raises
        ------
        DataPreparationError
            If :meth:`prepare_features` has not been called.
        """
        if self._X_train is None:
            raise DataPreparationError("Call prepare_features() before train().")

        merged: dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}
        merged["random_state"] = self._random_seed

        model = XGBClassifier(**merged)
        logger.info(
            f"Training XGBClassifier "
            f"(n_estimators={merged['n_estimators']}, "
            f"max_depth={merged['max_depth']}, "
            f"lr={merged['learning_rate']}) …"
        )
        model.fit(self._X_train, self._y_train)
        self._model = model
        logger.info("Training complete.")
        return self

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the trained model on the held-out test split.

        Decodes integer predictions back to label strings (``"BUY"`` /
        ``"HOLD"`` / ``"SELL"``) before computing metrics so that the
        classification report is human-readable.

        Returns
        -------
        dict
            Full metrics dict — see :func:`~src.model.metrics.compute_all_metrics`.

        Raises
        ------
        ModelNotTrainedError
            If :meth:`train` has not been called.
        """
        if self._model is None:
            raise ModelNotTrainedError("Call train() before evaluate().")

        y_pred_encoded: np.ndarray = self._model.predict(self._X_test)
        labels: list[str] = list(self._label_encoder.classes_)

        y_true_labels = self._label_encoder.inverse_transform(self._y_test)
        y_pred_labels = self._label_encoder.inverse_transform(y_pred_encoded)

        evaluator = ModelEvaluator(labels=labels)
        self._metrics = evaluator.evaluate(y_true_labels, y_pred_labels)
        evaluator.log_summary()
        return self._metrics

    def save_model(self) -> None:
        """Persist the model artifact, metrics JSON, and feature importance files.

        Saves three artefact types:

        1. **Model artifact** (``.joblib``) — model, label encoder, feature
           column list, and training metadata bundled together.
        2. **Metrics JSON** — full evaluation metrics from :meth:`evaluate`.
        3. **Feature importance PNG** — top-20 horizontal bar chart.
        4. **Feature importance CSV** — all features ranked by importance.

        Raises
        ------
        ModelNotTrainedError
            If :meth:`train` or :meth:`evaluate` has not been called.
        """
        if self._model is None:
            raise ModelNotTrainedError("Call train() before save_model().")
        if self._metrics is None:
            raise ModelNotTrainedError("Call evaluate() before save_model().")

        artifact: dict[str, Any] = {
            "model":           self._model,
            "label_encoder":   self._label_encoder,
            "feature_columns": self._feature_columns,
            "metadata": {
                "label_col":    self._label_col,
                "n_features":   len(self._feature_columns),
                "classes":      list(self._label_encoder.classes_),
                "random_seed":  self._random_seed,
                "dataset_path": str(self._dataset_path),
                "train_rows":   self.n_train,
                "test_rows":    self.n_test,
            },
        }

        save_model(artifact, self._model_out)
        from src.model.metrics import save_metrics  # noqa: PLC0415
        save_metrics(self._metrics, self._metrics_out)

        importance_df = compute_feature_importance(self._model, self._feature_columns)
        importance_csv = self._importance_out.parent / "feature_importance.csv"
        save_importance_csv(importance_df, importance_csv)
        plot_feature_importance(importance_df, self._importance_out)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_columns(self) -> list[str]:
        """Ordered list of feature column names used for training."""
        return list(self._feature_columns)

    @property
    def model(self) -> XGBClassifier | None:
        """Trained XGBClassifier, or ``None`` before :meth:`train` is called."""
        return self._model

    @property
    def metrics(self) -> dict[str, Any] | None:
        """Evaluation metrics dict, or ``None`` before :meth:`evaluate` is called."""
        return self._metrics

    @property
    def n_train(self) -> int:
        """Number of rows in the training split."""
        return int(len(self._X_train)) if self._X_train is not None else 0

    @property
    def n_test(self) -> int:
        """Number of rows in the test split."""
        return int(len(self._X_test)) if self._X_test is not None else 0

    @property
    def n_total(self) -> int:
        """Total labelled rows used (train + test)."""
        return self.n_train + self.n_test
