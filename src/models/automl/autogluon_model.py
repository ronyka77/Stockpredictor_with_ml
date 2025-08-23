"""
AutoGluon Model Wrapper

Provides a thin adapter around AutoGluon TabularPredictor to conform to
our project BaseModel/ModelProtocol expectations, including:
- predict(X: pd.DataFrame) -> np.ndarray
- get_prediction_confidence(X: pd.DataFrame, method: str) -> np.ndarray

Also exposes fit with (X, y, X_val, y_val) by reconstructing label columns
for AutoGluon training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from src.models.base_model import BaseModel
from src.models.evaluation.threshold_evaluator import ModelProtocol
from src.utils.logger import get_logger


logger = get_logger(__name__)


class AutoGluonModel(BaseModel, ModelProtocol):
    def __init__(self, *,
                model_name: str = "autogluon",
                config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config or {})
        self.predictor: Optional[TabularPredictor] = None
        self.feature_names: Optional[list[str]] = None

    def _create_model(self, **kwargs) -> TabularPredictor:
        # AutoGluon predictor created during fit with label specified; no pre-instance here
        return None

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> "AutoGluonModel":
        label = self.config.get("label", "Future_Return_10")
        eval_metric = self.config.get("eval_metric", "rmse")
        presets = self.config.get("presets", "high_quality")
        ag_args_fit = self.config.get("ag_args_fit", {"num_cpus": 12, "num_gpus": 0})
        hyperparameters = self.config.get("hyperparameters", {
            "GBM": {}, "XGB": {}, "CAT": {}, "RF": {}, "XT": {}, "REALMLP": {},
            "KNN": None,
        })

        # Build train/valid DataFrames with label column
        train_df = X.copy()
        train_df[label] = y.values
        valid_df = None
        if X_val is not None and y_val is not None:
            valid_df = X_val.copy()
            valid_df[label] = y_val.values

        logger.info(f"Training AutoGluon with label={label}, eval_metric={eval_metric}")
        self.predictor = TabularPredictor(label=label, eval_metric=eval_metric)
        self.predictor.fit(
            train_data=train_df,
            tuning_data=valid_df,
            presets=presets,
            ag_args_fit=ag_args_fit,
            hyperparameters=hyperparameters,
            # Enable using provided validation in bagged mode per AutoGluon requirement
            use_bag_holdout=True if valid_df is not None else False,
        )

        # Capture feature names (exclude label)
        self.feature_names = [c for c in train_df.columns if c != label]
        self.model = self  # for BaseModel predict routing
        self.is_trained = True
        logger.info("AutoGluon training completed")
        return self

    # ModelProtocol requirement
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.predictor:
            raise ValueError("Predictor not trained/loaded")
        if self.feature_names is not None:
            X = X[self.feature_names]
        preds = self.predictor.predict(X)
        return preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)

    # ModelProtocol requirement
    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'variance') -> np.ndarray:
        if not self.predictor:
            raise ValueError("Predictor not trained/loaded")
        if self.feature_names is not None:
            X = X[self.feature_names]

        method = (method or "variance").lower()
        # Ensemble variance across base models as default
        if method == 'variance':
            model_names = self.predictor.get_model_names()
            if not model_names:
                # Fallback to magnitude
                base = np.abs(self.predict(X))
                return base / base.max() if base.max() > 0 else base

            all_preds = []
            for m in model_names:
                try:
                    p = self.predictor.predict(X, model=m)
                    p = p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)
                    all_preds.append(p.reshape(-1))
                except Exception:
                    continue
            if len(all_preds) < 2:
                base = np.abs(self.predict(X))
                return base / base.max() if base.max() > 0 else base
            stacked = np.vstack(all_preds)
            var = stacked.var(axis=0)
            conf = 1.0 / (1.0 + var)
            # Min-max normalize to [0,1]
            vmin, vmax = conf.min(), conf.max()
            if vmax > vmin:
                conf = (conf - vmin) / (vmax - vmin)
            return conf

        if method == 'simple':
            base = np.abs(self.predict(X))
            return base / base.max() if base.max() > 0 else base

        if method == 'margin':
            base = np.abs(self.predict(X))
            return np.tanh(base)

        raise ValueError(f"Unsupported confidence method: {method}")

    # MLflow logging helper
    def save_to_mlflow(self, params: Dict[str, Any], metrics: Dict[str, float], *,
                        experiment_name: Optional[str] = None) -> str:
        self.start_mlflow_run(experiment_name or f"stock_prediction_{self.model_name}")
        try:
            self.log_params(params)
            self.log_metrics(metrics)
            # Log leaderboard
            try:
                leaderboard = self.predictor.leaderboard(silent=True)
                leaderboard_path = "leaderboard.csv"
                leaderboard.to_csv(leaderboard_path, index=False)
                self.mlflow_integration.log_artifact(leaderboard_path)
            except Exception as e:
                logger.warning(f"Failed to log leaderboard: {e}")

            # Persist full AutoGluon predictor as artifact for downstream loading
            try:
                save_dir = "autogluon_model"
                self.predictor.save(save_dir)
                import mlflow as _mlflow
                _mlflow.log_artifacts(save_dir, artifact_path="autogluon_model")
                logger.info("Logged AutoGluon predictor artifacts under autogluon_model/")
            except Exception as e:
                logger.warning(f"Failed to log AutoGluon model artifacts: {e}")
        finally:
            self.end_mlflow_run()
        return self.run_id or ""


