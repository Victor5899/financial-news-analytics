"""
Unit tests for src.model.trainer — ModelTrainer.

All tests use small synthetic DataFrames or temporary CSV files.
No PostgreSQL, no real XGBoost model, and (where possible) no actual
training to keep the suite fast.

Test organisation
-----------------
TestExceptionHierarchy       — exception class relationships
TestModelTrainerInit         — constructor stores paths and defaults
TestLoadDataset              — CSV loading, null-drop, error conditions
TestPrepareFeatures          — column detection, encoding, train/test split
TestTrain                    — XGBoost fitting with small synthetic data
TestEvaluate                 — metric dict structure after training
TestSaveModel                — artifact files created on disk
TestProperties               — read-only properties n_train, n_test, etc.
TestEndToEnd                 — complete pipeline on tiny synthetic dataset
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.trainer import (
    DIRECTION_CLASSES,
    LABEL_COL,
    DataPreparationError,
    ModelNotTrainedError,
    ModelTrainer,
    ModelTrainingError,
)


# ── Synthetic dataset helpers ─────────────────────────────────────────────────

def _make_feature_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """
    Create a synthetic ML dataset that mirrors the Phase 6 CSV schema.

    Includes typical metadata columns (ticker, date), all Phase 6 label
    columns (future_close_*, return_*, label_up_*, label_direction), and
    three numeric feature columns.
    """
    rng = np.random.default_rng(seed)
    directions = rng.choice(DIRECTION_CLASSES, size=n)

    df = pd.DataFrame(
        {
            "ticker":          ["AAPL"] * (n // 2) + ["TSLA"] * (n - n // 2),
            "date":            pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            # Feature columns — kept simple, three synthetic floats
            "sentiment_mean":  rng.uniform(-1, 1, n),
            "sentiment_std":   rng.uniform(0,  1, n),
            "article_count":   rng.integers(1, 20, n).astype(float),
            # Phase 6 label columns (all should be excluded from features)
            "future_close_1d": rng.uniform(100, 200, n),
            "future_close_5d": rng.uniform(100, 200, n),
            "return_1d":       rng.uniform(-0.05, 0.05, n),
            "return_5d":       rng.uniform(-0.05, 0.05, n),
            "label_up_1d":     rng.integers(0, 2, n),
            "label_up_5d":     rng.integers(0, 2, n),
            LABEL_COL:         directions,
        }
    )
    return df


def _make_csv(tmp_path: Path, n: int = 120, seed: int = 0) -> Path:
    """Write a synthetic dataset CSV to *tmp_path* and return the path."""
    csv_path = tmp_path / "ml_dataset_test.csv"
    _make_feature_df(n=n, seed=seed).to_csv(csv_path, index=False)
    return csv_path


def _make_trainer(tmp_path: Path, n: int = 120, seed: int = 0) -> ModelTrainer:
    """Return a ModelTrainer pointed at a synthetic dataset CSV."""
    csv_path = _make_csv(tmp_path, n=n, seed=seed)
    return ModelTrainer(
        dataset_path=csv_path,
        model_out=tmp_path / "model.joblib",
        metrics_out=tmp_path / "metrics.json",
        importance_out=tmp_path / "importance.png",
        random_seed=42,
    )


# ── TestExceptionHierarchy ────────────────────────────────────────────────────

class TestExceptionHierarchy:
    def test_data_preparation_is_training_error(self) -> None:
        assert issubclass(DataPreparationError, ModelTrainingError)

    def test_model_not_trained_is_training_error(self) -> None:
        assert issubclass(ModelNotTrainedError, ModelTrainingError)

    def test_model_training_error_is_exception(self) -> None:
        assert issubclass(ModelTrainingError, Exception)


# ── TestModelTrainerInit ──────────────────────────────────────────────────────

class TestModelTrainerInit:
    def test_stores_dataset_path(self, tmp_path: Path) -> None:
        csv = tmp_path / "dataset.csv"
        trainer = ModelTrainer(
            dataset_path=csv,
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        assert trainer._dataset_path == csv

    def test_default_random_seed(self, tmp_path: Path) -> None:
        trainer = ModelTrainer(
            dataset_path=tmp_path / "d.csv",
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        assert trainer._random_seed == 42

    def test_custom_random_seed(self, tmp_path: Path) -> None:
        trainer = ModelTrainer(
            dataset_path=tmp_path / "d.csv",
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
            random_seed=99,
        )
        assert trainer._random_seed == 99

    def test_initial_model_is_none(self, tmp_path: Path) -> None:
        trainer = ModelTrainer(
            dataset_path=tmp_path / "d.csv",
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        assert trainer.model is None

    def test_initial_metrics_is_none(self, tmp_path: Path) -> None:
        trainer = ModelTrainer(
            dataset_path=tmp_path / "d.csv",
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        assert trainer.metrics is None


# ── TestLoadDataset ───────────────────────────────────────────────────────────

class TestLoadDataset:
    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        trainer = ModelTrainer(
            dataset_path=tmp_path / "missing.csv",
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        with pytest.raises(DataPreparationError, match="not found"):
            trainer.load_dataset()

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("ticker,date,sentiment_mean,label_direction\n")
        trainer = ModelTrainer(
            dataset_path=empty_csv,
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        with pytest.raises(DataPreparationError, match="empty"):
            trainer.load_dataset()

    def test_drops_null_labels(self, tmp_path: Path) -> None:
        df = _make_feature_df(n=30)
        df.loc[df.index[:5], LABEL_COL] = None
        csv_path = tmp_path / "partial.csv"
        df.to_csv(csv_path, index=False)
        trainer = ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        trainer.load_dataset()
        assert len(trainer._df) == 25

    def test_returns_self_for_chaining(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        result = trainer.load_dataset()
        assert result is trainer

    def test_raises_if_all_labels_null(self, tmp_path: Path) -> None:
        df = _make_feature_df(n=10)
        df[LABEL_COL] = None
        csv_path = tmp_path / "all_null.csv"
        df.to_csv(csv_path, index=False)
        trainer = ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        with pytest.raises(DataPreparationError, match="No labelled rows"):
            trainer.load_dataset()

    def test_raises_if_prepare_called_without_load(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        with pytest.raises(DataPreparationError, match="load_dataset"):
            trainer.prepare_features()


# ── TestPrepareFeatures ───────────────────────────────────────────────────────

class TestPrepareFeatures:
    def test_detects_feature_columns(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        # Only sentiment_mean, sentiment_std, article_count should be features.
        assert set(trainer.feature_columns) == {
            "sentiment_mean", "sentiment_std", "article_count"
        }

    def test_excludes_metadata_columns(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        assert "ticker" not in trainer.feature_columns
        assert "date" not in trainer.feature_columns

    def test_excludes_label_columns(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        for col in trainer.feature_columns:
            assert not col.startswith("label_")
            assert not col.startswith("future_close_")
            assert not col.startswith("return_")

    def test_train_test_split_sizes(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=100)
        trainer.load_dataset().prepare_features()
        assert trainer.n_train == 80
        assert trainer.n_test == 20

    def test_n_total(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=100)
        trainer.load_dataset().prepare_features()
        assert trainer.n_total == 100

    def test_returns_self_for_chaining(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        result = trainer.load_dataset().prepare_features()
        assert result is trainer

    def test_raises_if_no_feature_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "ticker":        ["AAPL"] * 10,
                "date":          ["2025-01-01"] * 10,
                LABEL_COL:       ["BUY"] * 4 + ["HOLD"] * 3 + ["SELL"] * 3,
                "label_up_1d":   [1] * 10,
                "return_1d":     [0.01] * 10,
                "future_close_1d": [100.0] * 10,
            }
        )
        csv_path = tmp_path / "no_features.csv"
        df.to_csv(csv_path, index=False)
        trainer = ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "m.joblib",
            metrics_out=tmp_path / "m.json",
            importance_out=tmp_path / "i.png",
        )
        trainer.load_dataset()
        with pytest.raises(DataPreparationError, match="No feature columns"):
            trainer.prepare_features()


# ── TestTrain ─────────────────────────────────────────────────────────────────

class TestTrain:
    def test_model_is_set_after_training(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train()
        assert trainer.model is not None

    def test_raises_if_called_before_prepare(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset()
        with pytest.raises(DataPreparationError, match="prepare_features"):
            trainer.train()

    def test_returns_self_for_chaining(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        result = trainer.load_dataset().prepare_features().train()
        assert result is trainer

    def test_accepts_custom_params(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        # Should not raise with a smaller n_estimators.
        trainer.train(params={"n_estimators": 10})
        assert trainer.model is not None


# ── TestEvaluate ──────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_metrics_dict(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        metrics = trainer.load_dataset().prepare_features().train().evaluate()
        assert isinstance(metrics, dict)

    def test_required_keys_present(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        metrics = trainer.load_dataset().prepare_features().train().evaluate()
        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in metrics

    def test_accuracy_in_range(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        metrics = trainer.load_dataset().prepare_features().train().evaluate()
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_metrics_cached(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train().evaluate()
        assert trainer.metrics is not None

    def test_raises_if_called_before_train(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        with pytest.raises(ModelNotTrainedError, match="train"):
            trainer.evaluate()


# ── TestSaveModel ─────────────────────────────────────────────────────────────

class TestSaveModel:
    def test_model_file_created(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        assert (tmp_path / "model.joblib").exists()

    def test_metrics_file_created(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        assert (tmp_path / "metrics.json").exists()

    def test_importance_plot_created(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        assert (tmp_path / "importance.png").exists()

    def test_importance_csv_created(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        assert (tmp_path / "feature_importance.csv").exists()

    def test_raises_if_train_not_called(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        with pytest.raises(ModelNotTrainedError, match="train"):
            trainer.save_model()

    def test_raises_if_evaluate_not_called(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 5})
        with pytest.raises(ModelNotTrainedError, match="evaluate"):
            trainer.save_model()


# ── TestProperties ────────────────────────────────────────────────────────────

class TestProperties:
    def test_n_train_zero_before_prepare(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset()
        assert trainer.n_train == 0

    def test_n_test_zero_before_prepare(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset()
        assert trainer.n_test == 0

    def test_n_total_after_prepare(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=50)
        trainer.load_dataset().prepare_features()
        assert trainer.n_total == 50

    def test_feature_columns_copy(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features()
        cols = trainer.feature_columns
        cols.append("injected")
        assert "injected" not in trainer.feature_columns


# ── TestEndToEnd ──────────────────────────────────────────────────────────────

class TestEndToEnd:
    """Full pipeline with a tiny (120-row) synthetic dataset."""

    def test_full_pipeline_succeeds(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        assert trainer.model is not None
        assert trainer.metrics is not None
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "importance.png").exists()
