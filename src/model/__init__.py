"""Phase 7: XGBoost Model Training & Stock Movement Prediction.

``ModelTrainer``   — loads an ML dataset, engineers features, trains and
                     evaluates an XGBoost classifier, and persists all artifacts.
``ModelPredictor`` — loads a saved model artifact and predicts direction for
                     new rows (CSV, DataFrame, or a single feature vector).
``ModelEvaluator`` — orchestrates metric computation, logging, and JSON export.

Typical usage
-------------
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
