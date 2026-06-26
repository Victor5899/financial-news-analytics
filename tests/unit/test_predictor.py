"""
Unit tests for src.model.predictor — ModelPredictor.

Tests use a real but lightweight trained XGBoost model generated from a tiny
synthetic dataset to keep the suite self-contained and fast.

Test organisation
-----------------
TestModelPredictorInit          — constructor stores path, artifact starts None
TestLoadModel                   — loads valid artifact, raises on missing
TestPredictFromDataframe        — shape, column names, probability sums
TestPredictFromVector           — dict input, list input, wrong-length error
TestPredictFromCSV              — writes to CSV, raises on missing file
TestProperties                  — feature_columns and classes properties
TestMissingFeatures             — graceful errors when features are absent
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.model.model_io import save_model
from src.model.predictor import ModelPredictor
from src.model.trainer import DIRECTION_CLASSES, LABEL_COL


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_artifact(tmp_path: Path) -> dict:  # type: ignore[type-arg]
    """
    Train a minimal XGBClassifier on synthetic data and wrap it in an artifact
    bundle that ModelPredictor expects.
    """
    rng = np.random.default_rng(42)
    n = 90
    X = rng.uniform(0, 1, (n, 3)).astype(float)

    le = LabelEncoder()
    le.fit(DIRECTION_CLASSES)
    y = rng.choice(len(DIRECTION_CLASSES), size=n)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=5,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X, y)

    return {
        "model":           model,
        "label_encoder":   le,
        "feature_columns": ["feat_a", "feat_b", "feat_c"],
        "metadata": {
            "n_features":   3,
            "classes":      list(le.classes_),
            "random_seed":  42,
            "train_rows":   72,
            "test_rows":    18,
            "label_col":    LABEL_COL,
        },
    }


@pytest.fixture
def artifact_path(tmp_path: Path, trained_artifact: dict) -> Path:  # type: ignore[type-arg]
    """Save the artifact to a temp joblib file and return the path."""
    path = tmp_path / "model.joblib"
    save_model(trained_artifact, path)
    return path


@pytest.fixture
def loaded_predictor(artifact_path: Path) -> ModelPredictor:
    """Return a ModelPredictor that has already called load_model()."""
    predictor = ModelPredictor(model_path=artifact_path)
    predictor.load_model()
    return predictor


def _make_input_df(n: int = 5) -> pd.DataFrame:
    """Minimal DataFrame with the three feature columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 1, n),
            "feat_b": rng.uniform(0, 1, n),
            "feat_c": rng.uniform(0, 1, n),
        }
    )


# ── TestModelPredictorInit ────────────────────────────────────────────────────

class TestModelPredictorInit:
    def test_stores_model_path(self, tmp_path: Path) -> None:
        path = tmp_path / "model.joblib"
        predictor = ModelPredictor(model_path=path)
        assert predictor._model_path == path

    def test_artifact_is_none_initially(self, tmp_path: Path) -> None:
        predictor = ModelPredictor(model_path=tmp_path / "model.joblib")
        assert predictor._artifact is None

    def test_raises_if_predict_before_load(self, tmp_path: Path) -> None:
        predictor = ModelPredictor(model_path=tmp_path / "model.joblib")
        with pytest.raises(RuntimeError, match="load_model"):
            predictor.predict_from_dataframe(_make_input_df())


# ── TestLoadModel ─────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_load_succeeds(self, artifact_path: Path) -> None:
        predictor = ModelPredictor(model_path=artifact_path)
        result = predictor.load_model()
        assert predictor._artifact is not None

    def test_returns_self_for_chaining(self, artifact_path: Path) -> None:
        predictor = ModelPredictor(model_path=artifact_path)
        result = predictor.load_model()
        assert result is predictor

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        from src.model.model_io import ModelNotFoundError  # noqa: PLC0415
        predictor = ModelPredictor(model_path=tmp_path / "absent.joblib")
        with pytest.raises(ModelNotFoundError):
            predictor.load_model()


# ── TestPredictFromDataframe ──────────────────────────────────────────────────

class TestPredictFromDataframe:
    def test_returns_dataframe(self, loaded_predictor: ModelPredictor) -> None:
        result = loaded_predictor.predict_from_dataframe(_make_input_df())
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self, loaded_predictor: ModelPredictor) -> None:
        df = _make_input_df(n=7)
        result = loaded_predictor.predict_from_dataframe(df)
        assert len(result) == 7

    def test_predicted_direction_column_present(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        result = loaded_predictor.predict_from_dataframe(_make_input_df())
        assert "predicted_direction" in result.columns

    def test_probability_columns_present(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        result = loaded_predictor.predict_from_dataframe(_make_input_df())
        for cls in DIRECTION_CLASSES:
            assert f"prob_{cls}" in result.columns

    def test_directions_are_valid_classes(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        result = loaded_predictor.predict_from_dataframe(_make_input_df(n=20))
        assert set(result["predicted_direction"].unique()).issubset(set(DIRECTION_CLASSES))

    def test_probabilities_sum_to_one(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        result = loaded_predictor.predict_from_dataframe(_make_input_df())
        prob_cols = [f"prob_{cls}" for cls in DIRECTION_CLASSES]
        row_sums = result[prob_cols].sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0, abs=1e-5)

    def test_extra_columns_preserved(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        df = _make_input_df()
        df["ticker"] = "AAPL"
        result = loaded_predictor.predict_from_dataframe(df)
        assert "ticker" in result.columns

    def test_raises_on_missing_feature(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        df = _make_input_df().drop(columns=["feat_a"])
        with pytest.raises(ValueError, match="missing"):
            loaded_predictor.predict_from_dataframe(df)


# ── TestPredictFromVector ─────────────────────────────────────────────────────

class TestPredictFromVector:
    def test_dict_input_returns_direction(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        vec = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.8}
        result = loaded_predictor.predict_from_vector(vec)
        assert result["predicted_direction"] in DIRECTION_CLASSES

    def test_dict_input_returns_probabilities(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        vec = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.8}
        result = loaded_predictor.predict_from_vector(vec)
        assert "probabilities" in result
        assert set(result["probabilities"].keys()) == set(DIRECTION_CLASSES)

    def test_probabilities_sum_to_one(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        vec = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.8}
        result = loaded_predictor.predict_from_vector(vec)
        total = sum(result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_list_input(self, loaded_predictor: ModelPredictor) -> None:
        result = loaded_predictor.predict_from_vector([0.5, 0.3, 0.8])
        assert result["predicted_direction"] in DIRECTION_CLASSES

    def test_numpy_array_input(self, loaded_predictor: ModelPredictor) -> None:
        vec = np.array([0.5, 0.3, 0.8])
        result = loaded_predictor.predict_from_vector(vec)
        assert result["predicted_direction"] in DIRECTION_CLASSES

    def test_raises_on_wrong_length(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            loaded_predictor.predict_from_vector([0.5, 0.3])  # only 2 instead of 3

    def test_raises_on_missing_dict_key(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        with pytest.raises(ValueError, match="missing"):
            loaded_predictor.predict_from_vector({"feat_a": 0.5, "feat_b": 0.3})


# ── TestPredictFromCSV ────────────────────────────────────────────────────────

class TestPredictFromCSV:
    def test_returns_dataframe(
        self, loaded_predictor: ModelPredictor, tmp_path: Path
    ) -> None:
        csv = tmp_path / "input.csv"
        _make_input_df().to_csv(csv, index=False)
        result = loaded_predictor.predict_from_csv(csv)
        assert isinstance(result, pd.DataFrame)

    def test_raises_on_missing_csv(
        self, loaded_predictor: ModelPredictor, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            loaded_predictor.predict_from_csv(tmp_path / "absent.csv")

    def test_prediction_columns_in_output(
        self, loaded_predictor: ModelPredictor, tmp_path: Path
    ) -> None:
        csv = tmp_path / "input.csv"
        _make_input_df(n=3).to_csv(csv, index=False)
        result = loaded_predictor.predict_from_csv(csv)
        assert "predicted_direction" in result.columns
        for cls in DIRECTION_CLASSES:
            assert f"prob_{cls}" in result.columns


# ── TestProperties ────────────────────────────────────────────────────────────

class TestProperties:
    def test_feature_columns_property(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        cols = loaded_predictor.feature_columns
        assert cols == ["feat_a", "feat_b", "feat_c"]

    def test_classes_property(self, loaded_predictor: ModelPredictor) -> None:
        classes = loaded_predictor.classes
        assert set(classes) == set(DIRECTION_CLASSES)

    def test_properties_raise_before_load(self, tmp_path: Path) -> None:
        predictor = ModelPredictor(model_path=tmp_path / "model.joblib")
        with pytest.raises(RuntimeError):
            _ = predictor.feature_columns
        with pytest.raises(RuntimeError):
            _ = predictor.classes
