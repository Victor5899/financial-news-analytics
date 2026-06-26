"""
Phase 7: Model predictor — load a saved XGBoost artifact and predict direction.

``ModelPredictor`` is the inference counterpart to ``ModelTrainer``.  It loads
a self-contained artifact bundle (model + label encoder + feature columns) and
exposes three prediction surfaces:

* :meth:`predict_from_csv`      — input is a CSV file path
* :meth:`predict_from_dataframe`— input is a ``pd.DataFrame``
* :meth:`predict_from_vector`   — input is a single feature dict, list, or array

All surfaces return predicted direction strings (``"BUY"`` / ``"HOLD"`` /
``"SELL"``) plus per-class probabilities.

Usage
-----
    from src.model.predictor import ModelPredictor
    from pathlib import Path

    predictor = ModelPredictor(
        model_path=Path("artifacts/models/xgboost_direction_model.joblib")
    )
    predictor.load_model()

    # Predict from a CSV
    result_df = predictor.predict_from_csv(Path("data/ml/ml_dataset_2026-06-16.csv"))

    # Predict a single vector
    result = predictor.predict_from_vector({"sentiment_mean": 0.3, "volume": 1e6, ...})
    print(result["predicted_direction"])  # "BUY"
    print(result["probabilities"])        # {"BUY": 0.62, "HOLD": 0.25, "SELL": 0.13}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from src.model.model_io import load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Loads a saved XGBoost direction model and produces predictions.

    The loaded artifact bundle is fully self-contained: it carries the
    trained model, the label encoder, and the ordered feature column list,
    so the predictor does not depend on any training-time globals.

    Parameters
    ----------
    model_path : Path
        Path to a ``.joblib`` artifact produced by
        :meth:`~src.model.trainer.ModelTrainer.save_model`.
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path: Path = model_path
        self._artifact: dict[str, Any] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> "ModelPredictor":
        """Load the model artifact from disk.

        Returns
        -------
        ModelPredictor
            ``self`` (chainable).

        Raises
        ------
        ModelNotFoundError
            If the artifact file does not exist.
        ModelIOError
            If deserialisation fails.
        """
        self._artifact = load_model(self._model_path)
        meta = self._artifact.get("metadata", {})
        logger.info(
            f"Model loaded: {meta.get('n_features', '?')} features, "
            f"classes={meta.get('classes', '?')}, "
            f"trained_on={meta.get('train_rows', '?')} rows"
        )
        return self

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Raise if :meth:`load_model` has not been called."""
        if self._artifact is None:
            raise RuntimeError(
                "Model artifact not loaded. Call load_model() first."
            )

    def _align_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select and order the expected feature columns from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.  May contain extra columns (e.g. ``ticker``,
            ``date``) which are silently ignored.

        Returns
        -------
        np.ndarray
            Float array of shape ``(n_rows, n_features)`` in training order.

        Raises
        ------
        ValueError
            If any required feature column is absent from *df*.
        """
        self._ensure_loaded()
        feature_columns: list[str] = self._artifact["feature_columns"]  # type: ignore[index]
        missing = sorted(set(feature_columns) - set(df.columns))
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} required feature column(s): {missing}"
            )
        return df[feature_columns].values.astype(float)

    # ── Prediction surfaces ───────────────────────────────────────────────────

    def predict_from_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load a CSV file and return predictions as a DataFrame.

        Extra columns in the CSV (e.g. ``ticker``, ``date``) are preserved
        in the output alongside the prediction columns.

        Parameters
        ----------
        csv_path : Path
            Path to a CSV with feature columns.

        Returns
        -------
        pd.DataFrame
            Original CSV columns plus ``predicted_direction``,
            ``prob_BUY``, ``prob_HOLD``, ``prob_SELL``.

        Raises
        ------
        FileNotFoundError
            If *csv_path* does not exist.
        ValueError
            If required feature columns are absent.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path.name}")
        return self.predict_from_dataframe(df)

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict direction for each row in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input rows.  Must contain all feature columns used during training.
            Extra columns are preserved unchanged.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with four new columns appended:

            * ``predicted_direction`` — ``"BUY"``, ``"HOLD"``, or ``"SELL"``
            * ``prob_BUY``            — probability score for BUY
            * ``prob_HOLD``           — probability score for HOLD
            * ``prob_SELL``           — probability score for SELL

        Raises
        ------
        ValueError
            If required feature columns are absent from *df*.
        """
        self._ensure_loaded()
        X = self._align_features(df)
        model = self._artifact["model"]  # type: ignore[index]
        label_encoder = self._artifact["label_encoder"]  # type: ignore[index]

        y_pred_encoded: np.ndarray = model.predict(X)
        y_proba: np.ndarray = model.predict_proba(X)
        predicted: np.ndarray = label_encoder.inverse_transform(y_pred_encoded)
        classes: np.ndarray = label_encoder.classes_

        result = df.copy()
        result["predicted_direction"] = predicted
        for i, cls in enumerate(classes):
            result[f"prob_{cls}"] = y_proba[:, i]

        logger.info(f"Predicted {len(result)} row(s).")
        return result

    def predict_from_vector(
        self,
        feature_vector: Union[dict[str, float], list[float], np.ndarray],
    ) -> dict[str, Any]:
        """Predict direction for a single feature vector.

        Parameters
        ----------
        feature_vector : dict | list | ndarray
            * **dict** — keys must be feature column names; values are floats.
            * **list / ndarray** — values must be in the same order as the
              training feature columns (see :attr:`feature_columns`).

        Returns
        -------
        dict
            ``{"predicted_direction": str, "probabilities": {cls: float, ...}}``

        Raises
        ------
        ValueError
            If a dict key is missing, or a list/array has the wrong length.
        """
        self._ensure_loaded()
        feature_columns: list[str] = self._artifact["feature_columns"]  # type: ignore[index]
        model = self._artifact["model"]  # type: ignore[index]
        label_encoder = self._artifact["label_encoder"]  # type: ignore[index]

        if isinstance(feature_vector, dict):
            missing = sorted(set(feature_columns) - set(feature_vector))
            if missing:
                raise ValueError(
                    f"Feature vector is missing {len(missing)} key(s): {missing}"
                )
            x = np.array([feature_vector[f] for f in feature_columns], dtype=float)
        else:
            x = np.asarray(feature_vector, dtype=float)
            if x.shape[0] != len(feature_columns):
                raise ValueError(
                    f"Feature vector length mismatch: "
                    f"expected {len(feature_columns)}, got {x.shape[0]}"
                )

        x_2d = x.reshape(1, -1)
        y_pred_encoded = model.predict(x_2d)[0]
        y_proba: np.ndarray = model.predict_proba(x_2d)[0]
        predicted: str = label_encoder.inverse_transform([y_pred_encoded])[0]
        classes: np.ndarray = label_encoder.classes_
        probabilities = {str(cls): float(y_proba[i]) for i, cls in enumerate(classes)}

        return {"predicted_direction": predicted, "probabilities": probabilities}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_columns(self) -> list[str]:
        """Ordered feature column names from the loaded artifact.

        Raises
        ------
        RuntimeError
            If :meth:`load_model` has not been called.
        """
        self._ensure_loaded()
        return list(self._artifact["feature_columns"])  # type: ignore[index]

    @property
    def classes(self) -> list[str]:
        """Ordered class names from the loaded label encoder.

        Raises
        ------
        RuntimeError
            If :meth:`load_model` has not been called.
        """
        self._ensure_loaded()
        return list(self._artifact["label_encoder"].classes_)  # type: ignore[index]
