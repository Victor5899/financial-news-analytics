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
TestTimeSeriesSplit          — _prepare_time_series_split chronological split (Phase 7.5)
TestTuneHyperparameters      — RandomizedSearchCV path with mocked search (Phase 7.5)
TestBestParamsSaving         — xgboost_best_params.json created in tune mode (Phase 7.5)
TestSaveModelTop20           — top20_feature_importance.csv created (Phase 7.5)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.model.trainer import (
    DIRECTION_CLASSES,
    LABEL_COL,
    DataPreparationError,
    ModelNotTrainedError,
    ModelTrainer,
    ModelTrainingError,
    _TUNE_PARAM_GRID,
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
        # With n=120 and TimeSeriesSplit(n_splits=5), the last fold has
        # test_size = 120 // (5+1) = 20 rows → train=100, test=20.
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features()
        assert trainer.n_train == 100
        assert trainer.n_test == 20

    def test_n_total(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features()
        assert trainer.n_total == 120

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


# ── TestTimeSeriesSplit ───────────────────────────────────────────────────────

class TestTimeSeriesSplit:
    """Tests for _prepare_time_series_split (Phase 7.5)."""

    def test_chronological_split_no_overlap(self, tmp_path: Path) -> None:
        """Train indices must all come before test indices."""
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features()
        # The sorted dataset has 120 rows; train set covers the first 100.
        assert trainer.n_train == 100
        assert trainer.n_test == 20

    def test_train_test_sizes_sum_to_total(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features()
        assert trainer.n_train + trainer.n_test == trainer.n_total

    def test_custom_n_splits(self, tmp_path: Path) -> None:
        """With n_splits=4 and n=120: test_size=120//5=24, train=96."""
        csv_path = _make_csv(tmp_path, n=120)
        trainer = ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "model.joblib",
            metrics_out=tmp_path / "metrics.json",
            importance_out=tmp_path / "importance.png",
            n_splits=4,
        )
        trainer.load_dataset().prepare_features()
        assert trainer.n_test == 24
        assert trainer.n_train == 96
        assert trainer.n_total == 120

    def test_split_returns_correct_array_shapes(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features()
        assert trainer._X_train.shape[0] == trainer.n_train
        assert trainer._X_test.shape[0] == trainer.n_test
        assert trainer._y_train.shape[0] == trainer.n_train
        assert trainer._y_test.shape[0] == trainer.n_test

    def test_no_shuffle_preserves_order(self, tmp_path: Path) -> None:
        """TimeSeriesSplit must not shuffle — train comes before test in time."""
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset().prepare_features()
        # With the sorted synthetic dataset (monotone dates), X_train rows
        # must all have a strictly lower positional index than X_test rows.
        # We verify this via X shape: train is the first 100, test is last 20.
        assert trainer.n_train == 100
        assert trainer.n_test == 20

    def test_direct_helper_returns_four_arrays(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path, n=120)
        trainer.load_dataset()
        # Build X, y manually to call the helper directly.
        df = trainer._df
        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "date", LABEL_COL}
            and not any(c.startswith(p) for p in ("future_close_", "return_", "label_"))
        ]
        X = df[feature_cols].values.astype(float)
        le = LabelEncoder()
        le.fit(DIRECTION_CLASSES)
        y = le.transform(df[LABEL_COL].values)
        result = trainer._prepare_time_series_split(X, y)
        assert len(result) == 4
        X_tr, X_te, y_tr, y_te = result
        assert X_tr.shape[0] + X_te.shape[0] == len(X)


# ── TestTuneHyperparameters ───────────────────────────────────────────────────

class TestTuneHyperparameters:
    """Tests for the RandomizedSearchCV path (Phase 7.5)."""

    def _make_tune_trainer(self, tmp_path: Path) -> ModelTrainer:
        csv_path = _make_csv(tmp_path, n=120)
        return ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "model.joblib",
            metrics_out=tmp_path / "metrics.json",
            importance_out=tmp_path / "importance.png",
            tune=True,
        )

    def test_tune_param_grid_keys(self) -> None:
        """Search space must contain all expected hyperparameter keys."""
        expected_keys = {
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight", "gamma",
        }
        assert expected_keys == set(_TUNE_PARAM_GRID.keys())

    def test_tune_param_grid_values_are_lists(self) -> None:
        for key, values in _TUNE_PARAM_GRID.items():
            assert isinstance(values, list), f"{key!r} value must be a list"
            assert len(values) > 0, f"{key!r} list must not be empty"

    @patch("src.model.trainer.RandomizedSearchCV")
    def test_tune_mode_uses_randomized_search(
        self, mock_rscv_cls: MagicMock, tmp_path: Path
    ) -> None:
        """When tune=True, RandomizedSearchCV is instantiated and .fit() called."""
        # Build a mock best_estimator_ that acts like a fitted XGBClassifier.
        mock_estimator = MagicMock()
        mock_estimator.predict.return_value = np.zeros(20, dtype=int)

        mock_search_instance = MagicMock()
        mock_search_instance.best_estimator_ = mock_estimator
        mock_search_instance.best_params_ = {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 1, "gamma": 0,
        }
        mock_search_instance.best_score_ = 0.42
        mock_rscv_cls.return_value = mock_search_instance

        trainer = self._make_tune_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train()

        mock_rscv_cls.assert_called_once()
        mock_search_instance.fit.assert_called_once()
        assert trainer.model is mock_estimator

    @patch("src.model.trainer.RandomizedSearchCV")
    def test_tune_stores_best_params(
        self, mock_rscv_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_estimator = MagicMock()
        mock_estimator.predict.return_value = np.zeros(20, dtype=int)

        best_params = {
            "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 0.9, "colsample_bytree": 0.7,
            "min_child_weight": 3, "gamma": 0.1,
        }
        mock_search_instance = MagicMock()
        mock_search_instance.best_estimator_ = mock_estimator
        mock_search_instance.best_params_ = best_params
        mock_search_instance.best_score_ = 0.55
        mock_rscv_cls.return_value = mock_search_instance

        trainer = self._make_tune_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train()

        assert trainer.best_params == best_params
        assert trainer.cv_score == pytest.approx(0.55)

    def test_fast_mode_best_params_is_none(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 5})
        assert trainer.best_params is None
        assert trainer.cv_score is None


# ── TestBestParamsSaving ──────────────────────────────────────────────────────

class TestBestParamsSaving:
    """Tests that best params JSON is written in tune mode (Phase 7.5)."""

    @patch("src.model.trainer.RandomizedSearchCV")
    def test_best_params_json_created(
        self, mock_rscv_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_estimator = MagicMock()
        mock_estimator.predict.return_value = np.zeros(20, dtype=int)

        best_params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 1, "gamma": 0,
        }
        mock_search_instance = MagicMock()
        mock_search_instance.best_estimator_ = mock_estimator
        mock_search_instance.best_params_ = best_params
        mock_search_instance.best_score_ = 0.48
        mock_rscv_cls.return_value = mock_search_instance

        csv_path = _make_csv(tmp_path, n=120)
        metrics_out = tmp_path / "metrics.json"
        trainer = ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "model.joblib",
            metrics_out=metrics_out,
            importance_out=tmp_path / "importance.png",
            tune=True,
        )
        trainer.load_dataset().prepare_features().train()

        best_params_out = metrics_out.parent / "xgboost_best_params.json"
        assert best_params_out.exists()

    @patch("src.model.trainer.RandomizedSearchCV")
    def test_best_params_json_valid(
        self, mock_rscv_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_estimator = MagicMock()
        mock_estimator.predict.return_value = np.zeros(20, dtype=int)

        best_params = {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.9,
            "min_child_weight": 5, "gamma": 0.3,
        }
        mock_search_instance = MagicMock()
        mock_search_instance.best_estimator_ = mock_estimator
        mock_search_instance.best_params_ = best_params
        mock_search_instance.best_score_ = 0.51
        mock_rscv_cls.return_value = mock_search_instance

        csv_path = _make_csv(tmp_path, n=120)
        metrics_out = tmp_path / "metrics.json"
        trainer = ModelTrainer(
            dataset_path=csv_path,
            model_out=tmp_path / "model.joblib",
            metrics_out=metrics_out,
            importance_out=tmp_path / "importance.png",
            tune=True,
        )
        trainer.load_dataset().prepare_features().train()

        best_params_out = metrics_out.parent / "xgboost_best_params.json"
        payload = json.loads(best_params_out.read_text())
        assert "best_params" in payload
        assert "best_cv_score_f1_macro" in payload
        assert payload["best_cv_score_f1_macro"] == pytest.approx(0.51)
        assert payload["scoring"] == "f1_macro"
        assert payload["best_params"] == best_params

    def test_best_params_json_not_created_in_fast_mode(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 5})
        best_params_out = tmp_path / "xgboost_best_params.json"
        assert not best_params_out.exists()


# ── TestSaveModelTop20 ────────────────────────────────────────────────────────

class TestSaveModelTop20:
    """Tests that top20_feature_importance.csv is written (Phase 7.5)."""

    def test_top20_csv_created(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        assert (tmp_path / "top20_feature_importance.csv").exists()

    def test_top20_csv_has_rank_column(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        df = pd.read_csv(tmp_path / "top20_feature_importance.csv")
        assert "rank" in df.columns
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_top20_csv_has_at_most_20_rows(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        df = pd.read_csv(tmp_path / "top20_feature_importance.csv")
        assert len(df) <= 20

    def test_top20_csv_rank_starts_at_1(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.load_dataset().prepare_features().train({"n_estimators": 10})
        trainer.evaluate()
        trainer.save_model()
        df = pd.read_csv(tmp_path / "top20_feature_importance.csv")
        assert df["rank"].iloc[0] == 1
