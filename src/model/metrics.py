"""
Phase 7: Classification metric computation for the XGBoost direction model.

All public functions accept plain Python/NumPy arrays and return JSON-safe
Python primitives so that metric dicts can be serialised directly to disk.

Usage
-----
    from src.model.metrics import compute_all_metrics, save_metrics
    from pathlib import Path

    metrics = compute_all_metrics(y_true, y_pred, labels=["BUY", "HOLD", "SELL"])
    save_metrics(metrics, Path("artifacts/metrics/xgboost_metrics.json"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Individual metric helpers ─────────────────────────────────────────────────

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return overall classification accuracy as a plain Python float.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        Fraction of correctly classified samples in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute macro-averaged and per-class precision.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.
    labels : list[str]
        Ordered class names matching the label encoder classes.

    Returns
    -------
    dict
        ``{"macro": float, "per_class": {label: float, ...}}``
    """
    macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class = {lbl: float(v) for lbl, v in zip(labels, per_class_arr)}
    return {"macro": macro, "per_class": per_class}


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute macro-averaged and per-class recall.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.
    labels : list[str]
        Ordered class names matching the label encoder classes.

    Returns
    -------
    dict
        ``{"macro": float, "per_class": {label: float, ...}}``
    """
    macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_arr = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class = {lbl: float(v) for lbl, v in zip(labels, per_class_arr)}
    return {"macro": macro, "per_class": per_class}


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute macro-averaged and per-class F1 score.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.
    labels : list[str]
        Ordered class names matching the label encoder classes.

    Returns
    -------
    dict
        ``{"macro": float, "per_class": {label: float, ...}}``
    """
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class = {lbl: float(v) for lbl, v in zip(labels, per_class_arr)}
    return {"macro": macro, "per_class": per_class}


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> str:
    """Return a human-readable sklearn classification report string.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.
    labels : list[str]
        Display names for the classes (used as target_names).

    Returns
    -------
    str
        Multi-line report with precision, recall, F1, and support per class.
    """
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[list[int]]:
    """Return the confusion matrix as a nested list of Python ints.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    list[list[int]]
        Square matrix where ``cm[i][j]`` is the number of samples with true
        class ``i`` predicted as class ``j``.
    """
    return confusion_matrix(y_true, y_pred).tolist()


# ── Aggregate helper ──────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute the full suite of classification metrics in one call.

    All values are JSON-serialisable Python primitives.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.
    labels : list[str]
        Ordered class names (e.g. ``["BUY", "HOLD", "SELL"]``).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``classification_report``, ``confusion_matrix``, ``labels``.
    """
    return {
        "accuracy":               compute_accuracy(y_true, y_pred),
        "precision":              compute_precision(y_true, y_pred, labels),
        "recall":                 compute_recall(y_true, y_pred, labels),
        "f1":                     compute_f1(y_true, y_pred, labels),
        "classification_report":  compute_classification_report(y_true, y_pred, labels),
        "confusion_matrix":       compute_confusion_matrix(y_true, y_pred),
        "labels":                 labels,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Serialise a metrics dict to JSON at the given path.

    The ``classification_report`` string is stored under its own key.
    Directory is created automatically if it does not exist.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_all_metrics`.
    path : Path
        Destination JSON file (e.g. ``artifacts/metrics/xgboost_metrics.json``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build a fully serialisable copy — keep classification_report as a string.
    serialisable: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            serialisable[key] = val.tolist()
        else:
            serialisable[key] = val

    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    logger.info(f"Metrics saved → {path}")
