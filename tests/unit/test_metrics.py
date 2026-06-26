"""
Unit tests for src.model.metrics — classification metric computation.

All tests use small in-memory arrays.  No model, no filesystem, no DB.

Test organisation
-----------------
TestComputeAccuracy          — basic accuracy formula correctness
TestComputePrecision         — macro and per-class precision structure
TestComputeRecall            — macro and per-class recall structure
TestComputeF1                — macro and per-class F1 structure
TestComputeClassificationReport — report is a non-empty string
TestComputeConfusionMatrix   — shape and value correctness
TestComputeAllMetrics        — aggregate helper, all keys present
TestSaveMetrics              — JSON file creation and deserialisation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model.metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_f1,
    compute_precision,
    compute_recall,
    save_metrics,
)

_LABELS = ["BUY", "HOLD", "SELL"]

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth equals predictions — accuracy = 1.0."""
    y_true = np.array(["BUY", "HOLD", "SELL", "BUY", "HOLD"])
    y_pred = np.array(["BUY", "HOLD", "SELL", "BUY", "HOLD"])
    return y_true, y_pred


@pytest.fixture
def mixed_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Realistic mix of correct and incorrect predictions."""
    y_true = np.array(["BUY",  "BUY",  "HOLD", "HOLD", "SELL", "SELL"])
    y_pred = np.array(["BUY",  "HOLD", "BUY",  "HOLD", "SELL", "BUY"])
    return y_true, y_pred


# ── TestComputeAccuracy ───────────────────────────────────────────────────────

class TestComputeAccuracy:
    def test_perfect_accuracy(
        self, perfect_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = perfect_predictions
        assert compute_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_partial_accuracy(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        acc = compute_accuracy(y_true, y_pred)
        assert 0.0 < acc < 1.0

    def test_returns_python_float(
        self, perfect_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = perfect_predictions
        assert isinstance(compute_accuracy(y_true, y_pred), float)

    def test_zero_accuracy(self) -> None:
        y_true = np.array(["BUY",  "HOLD", "SELL"])
        y_pred = np.array(["SELL", "BUY",  "HOLD"])
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.0)


# ── TestComputePrecision ──────────────────────────────────────────────────────

class TestComputePrecision:
    def test_structure(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_precision(y_true, y_pred, _LABELS)
        assert "macro" in result
        assert "per_class" in result

    def test_per_class_keys(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_precision(y_true, y_pred, _LABELS)
        assert set(result["per_class"].keys()) == set(_LABELS)

    def test_perfect_precision(
        self, perfect_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = perfect_predictions
        result = compute_precision(y_true, y_pred, _LABELS)
        assert result["macro"] == pytest.approx(1.0)

    def test_values_are_python_floats(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_precision(y_true, y_pred, _LABELS)
        assert isinstance(result["macro"], float)
        for v in result["per_class"].values():
            assert isinstance(v, float)


# ── TestComputeRecall ─────────────────────────────────────────────────────────

class TestComputeRecall:
    def test_structure(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_recall(y_true, y_pred, _LABELS)
        assert "macro" in result and "per_class" in result

    def test_perfect_recall(
        self, perfect_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = perfect_predictions
        assert compute_recall(y_true, y_pred, _LABELS)["macro"] == pytest.approx(1.0)

    def test_per_class_keys(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_recall(y_true, y_pred, _LABELS)
        assert set(result["per_class"].keys()) == set(_LABELS)


# ── TestComputeF1 ─────────────────────────────────────────────────────────────

class TestComputeF1:
    def test_structure(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_f1(y_true, y_pred, _LABELS)
        assert "macro" in result and "per_class" in result

    def test_perfect_f1(
        self, perfect_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = perfect_predictions
        assert compute_f1(y_true, y_pred, _LABELS)["macro"] == pytest.approx(1.0)

    def test_f1_between_precision_and_recall(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        prec = compute_precision(y_true, y_pred, _LABELS)["macro"]
        rec  = compute_recall(y_true, y_pred, _LABELS)["macro"]
        f1   = compute_f1(y_true, y_pred, _LABELS)["macro"]
        # F1 is always <= max(precision, recall) and >= min(precision, recall)
        assert f1 <= max(prec, rec) + 1e-9
        assert f1 >= min(prec, rec) - 1e-9


# ── TestComputeClassificationReport ──────────────────────────────────────────

class TestComputeClassificationReport:
    def test_returns_non_empty_string(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        report = compute_classification_report(y_true, y_pred, _LABELS)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_class_names(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        report = compute_classification_report(y_true, y_pred, _LABELS)
        for label in _LABELS:
            assert label in report


# ── TestComputeConfusionMatrix ────────────────────────────────────────────────

class TestComputeConfusionMatrix:
    def test_returns_nested_list(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, list)
        assert all(isinstance(row, list) for row in cm)

    def test_diagonal_perfect_predictions(
        self, perfect_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = perfect_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        # All off-diagonal elements should be zero.
        for i, row in enumerate(cm):
            for j, val in enumerate(row):
                if i != j:
                    assert val == 0

    def test_row_sums_equal_class_counts(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        row_sums = [sum(row) for row in cm]
        assert sum(row_sums) == len(y_true)


# ── TestComputeAllMetrics ─────────────────────────────────────────────────────

class TestComputeAllMetrics:
    _REQUIRED_KEYS = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "classification_report",
        "confusion_matrix",
        "labels",
    }

    def test_all_keys_present(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_all_metrics(y_true, y_pred, _LABELS)
        assert self._REQUIRED_KEYS.issubset(set(result.keys()))

    def test_labels_preserved(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_all_metrics(y_true, y_pred, _LABELS)
        assert result["labels"] == _LABELS

    def test_accuracy_consistent_with_individual(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_all_metrics(y_true, y_pred, _LABELS)
        assert result["accuracy"] == pytest.approx(
            compute_accuracy(y_true, y_pred)
        )

    def test_json_serialisable(
        self, mixed_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = mixed_predictions
        result = compute_all_metrics(y_true, y_pred, _LABELS)
        # Should not raise.
        serialised = json.dumps(result)
        assert len(serialised) > 0


# ── TestSaveMetrics ───────────────────────────────────────────────────────────

class TestSaveMetrics:
    def test_creates_json_file(
        self,
        tmp_path: Path,
        mixed_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        y_true, y_pred = mixed_predictions
        metrics = compute_all_metrics(y_true, y_pred, _LABELS)
        out = tmp_path / "metrics.json"
        save_metrics(metrics, out)
        assert out.exists()

    def test_json_is_valid(
        self,
        tmp_path: Path,
        mixed_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        y_true, y_pred = mixed_predictions
        metrics = compute_all_metrics(y_true, y_pred, _LABELS)
        out = tmp_path / "metrics.json"
        save_metrics(metrics, out)
        loaded = json.loads(out.read_text())
        assert "accuracy" in loaded
        assert "confusion_matrix" in loaded

    def test_creates_parent_dirs(
        self,
        tmp_path: Path,
        mixed_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        y_true, y_pred = mixed_predictions
        metrics = compute_all_metrics(y_true, y_pred, _LABELS)
        out = tmp_path / "nested" / "deep" / "metrics.json"
        save_metrics(metrics, out)
        assert out.exists()

    def test_accuracy_round_trips(
        self,
        tmp_path: Path,
        perfect_predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        y_true, y_pred = perfect_predictions
        metrics = compute_all_metrics(y_true, y_pred, _LABELS)
        out = tmp_path / "m.json"
        save_metrics(metrics, out)
        loaded = json.loads(out.read_text())
        assert loaded["accuracy"] == pytest.approx(1.0)
