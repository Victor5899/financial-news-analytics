"""
Phase 7: Model evaluator — orchestrates metric computation, logging, and persistence.

``ModelEvaluator`` wraps the lower-level :mod:`~src.model.metrics` functions
into a stateful object that mirrors the orchestration style used elsewhere in
this project (thin public API, lazy caching, structured logging).

Usage
-----
    from src.model.evaluator import ModelEvaluator
    import numpy as np

    evaluator = ModelEvaluator(labels=["BUY", "HOLD", "SELL"])
    metrics = evaluator.evaluate(y_true, y_pred)
    evaluator.log_summary()
    evaluator.save(Path("artifacts/metrics/xgboost_metrics.json"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.model.metrics import compute_all_metrics, save_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Orchestrates classification metric computation, logging, and persistence.

    Parameters
    ----------
    labels : list[str]
        Ordered class names (e.g. ``["BUY", "HOLD", "SELL"]``).
        Must match the order used by the :class:`~sklearn.preprocessing.LabelEncoder`.
    """

    def __init__(self, labels: list[str]) -> None:
        self._labels: list[str] = labels
        self._metrics: dict[str, Any] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Compute and cache all classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth labels (decoded strings or encoded ints).
        y_pred : array-like
            Model predictions (same dtype as ``y_true``).

        Returns
        -------
        dict
            Full metrics dict — see :func:`~src.model.metrics.compute_all_metrics`.
        """
        self._metrics = compute_all_metrics(y_true, y_pred, self._labels)
        return self._metrics

    def save(self, path: Path) -> None:
        """Persist the cached metrics dict to a JSON file.

        Parameters
        ----------
        path : Path
            Destination JSON file.

        Raises
        ------
        RuntimeError
            If :meth:`evaluate` has not been called yet.
        """
        if self._metrics is None:
            raise RuntimeError("Call evaluate() before save().")
        save_metrics(self._metrics, path)

    def log_summary(self) -> None:
        """Emit key metric values at INFO level via the module logger.

        Raises
        ------
        RuntimeError
            If :meth:`evaluate` has not been called yet.
        """
        if self._metrics is None:
            raise RuntimeError("Call evaluate() before log_summary().")
        logger.info(f"  Accuracy   : {self._metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1   : {self._metrics['f1']['macro']:.4f}")
        logger.info(f"  Macro Prec : {self._metrics['precision']['macro']:.4f}")
        logger.info(f"  Macro Rec  : {self._metrics['recall']['macro']:.4f}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def metrics(self) -> dict[str, Any] | None:
        """Cached metrics dict, or ``None`` if :meth:`evaluate` not yet called."""
        return self._metrics

    @property
    def labels(self) -> list[str]:
        """Ordered class names this evaluator was configured with."""
        return self._labels
