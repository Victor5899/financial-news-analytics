"""
Phase 7: Model persistence helpers — save and load XGBoost artifact bundles.

An artifact bundle is a plain Python dict containing:

    {
        "model":           <XGBClassifier>,
        "label_encoder":   <LabelEncoder>,
        "feature_columns": <list[str]>,
        "metadata":        <dict>,      # n_features, classes, train_rows, …
    }

Persisting the entire bundle in one joblib file keeps the predictor
self-contained: it only needs the path, not any external schema.

Usage
-----
    from src.model.model_io import save_model, load_model
    from pathlib import Path

    save_model(artifact, Path("artifacts/models/xgboost_direction_model.joblib"))
    artifact = load_model(Path("artifacts/models/xgboost_direction_model.joblib"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ModelIOError(Exception):
    """Base exception for model persistence errors."""


class ModelNotFoundError(ModelIOError):
    """Raised when the requested model artifact file does not exist."""


# ── Public API ────────────────────────────────────────────────────────────────

def save_model(artifact: dict[str, Any], path: Path) -> None:
    """Persist a model artifact bundle to disk using joblib compression.

    Creates any missing parent directories automatically.

    Parameters
    ----------
    artifact : dict
        Bundle produced by :class:`~src.model.trainer.ModelTrainer`.
        Must contain at least ``model``, ``label_encoder``,
        ``feature_columns``, and ``metadata`` keys.
    path : Path
        Destination file path (conventionally ``*.joblib``).

    Raises
    ------
    ModelIOError
        If joblib serialisation fails for any reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(artifact, path, compress=3)
    except Exception as exc:  # noqa: BLE001
        raise ModelIOError(f"Failed to save model artifact to {path}: {exc}") from exc
    logger.info(f"Model artifact saved → {path}")


def load_model(path: Path) -> dict[str, Any]:
    """Load a model artifact bundle from disk.

    Parameters
    ----------
    path : Path
        Path to a ``.joblib`` file produced by :func:`save_model`.

    Returns
    -------
    dict
        Artifact bundle with ``model``, ``label_encoder``,
        ``feature_columns``, and ``metadata`` keys.

    Raises
    ------
    ModelNotFoundError
        If the file does not exist.
    ModelIOError
        If joblib deserialisation fails.
    """
    if not path.exists():
        raise ModelNotFoundError(
            f"Model artifact not found: {path}. "
            "Run scripts/train_model.py first to produce it."
        )
    try:
        artifact: dict[str, Any] = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelIOError(f"Failed to load model artifact from {path}: {exc}") from exc
    logger.info(f"Model artifact loaded ← {path}")
    return artifact
