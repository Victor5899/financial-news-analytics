"""
Unit tests for src.model.model_io — model artifact persistence.

All tests use temporary directories and lightweight dummy artifacts.
No actual XGBoost model or external services are required.

Test organisation
-----------------
TestSaveModel         — creates file, creates parent dirs, raises on error
TestLoadModel         — loads correct artifact, raises ModelNotFoundError
TestRoundTrip         — save then load preserves artifact contents exactly
TestModelIOError      — exception hierarchy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.model.model_io import (
    ModelIOError,
    ModelNotFoundError,
    load_model,
    save_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_artifact() -> dict[str, Any]:
    """Minimal artifact dict that does not require a real XGBoost model."""
    return {
        "model":           "fake_model_object",
        "label_encoder":   "fake_encoder",
        "feature_columns": ["feat_a", "feat_b", "feat_c"],
        "metadata": {
            "n_features":   3,
            "classes":      ["BUY", "HOLD", "SELL"],
            "random_seed":  42,
            "train_rows":   80,
            "test_rows":    20,
        },
    }


# ── TestSaveModel ─────────────────────────────────────────────────────────────

class TestSaveModel:
    def test_creates_file(self, tmp_path: Path, dummy_artifact: dict[str, Any]) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        assert out.exists()

    def test_creates_parent_directories(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "nested" / "deep" / "model.joblib"
        save_model(dummy_artifact, out)
        assert out.exists()

    def test_file_is_non_empty(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        assert out.stat().st_size > 0

    def test_raises_model_io_error_on_failure(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        with patch("src.model.model_io.joblib.dump", side_effect=OSError("disk full")):
            with pytest.raises(ModelIOError, match="disk full"):
                save_model(dummy_artifact, out)


# ── TestLoadModel ─────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_raises_model_not_found_if_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.joblib"
        with pytest.raises(ModelNotFoundError, match="not found"):
            load_model(missing)

    def test_raises_model_io_error_on_corrupt_file(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.joblib"
        corrupt.write_bytes(b"this is not a valid joblib file")
        with pytest.raises(ModelIOError):
            load_model(corrupt)

    def test_loads_valid_artifact(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        loaded = load_model(out)
        assert isinstance(loaded, dict)

    def test_raises_model_io_error_on_joblib_failure(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        with patch("src.model.model_io.joblib.load", side_effect=RuntimeError("bad")):
            with pytest.raises(ModelIOError, match="bad"):
                load_model(out)


# ── TestRoundTrip ─────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_feature_columns_preserved(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        loaded = load_model(out)
        assert loaded["feature_columns"] == dummy_artifact["feature_columns"]

    def test_metadata_preserved(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        loaded = load_model(out)
        assert loaded["metadata"]["classes"] == dummy_artifact["metadata"]["classes"]
        assert loaded["metadata"]["n_features"] == dummy_artifact["metadata"]["n_features"]

    def test_model_object_preserved(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        loaded = load_model(out)
        assert loaded["model"] == dummy_artifact["model"]

    def test_all_keys_present_after_round_trip(
        self, tmp_path: Path, dummy_artifact: dict[str, Any]
    ) -> None:
        out = tmp_path / "model.joblib"
        save_model(dummy_artifact, out)
        loaded = load_model(out)
        for key in ("model", "label_encoder", "feature_columns", "metadata"):
            assert key in loaded


# ── TestModelIOError ──────────────────────────────────────────────────────────

class TestModelIOError:
    def test_model_not_found_is_model_io_error(self) -> None:
        assert issubclass(ModelNotFoundError, ModelIOError)

    def test_model_io_error_is_exception(self) -> None:
        assert issubclass(ModelIOError, Exception)

    def test_model_not_found_message(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.joblib"
        with pytest.raises(ModelNotFoundError) as exc_info:
            load_model(missing)
        assert str(missing) in str(exc_info.value)
