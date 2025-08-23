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
from autogluon.core.metrics import make_scorer

from src.models.base_model import BaseModel
from src.models.evaluation.threshold_evaluator import ModelProtocol
from src.utils.logger import get_logger


logger = get_logger(__name__)


def default_mean_diff(y_true, y_pred, *args, **kwargs):
    """Top-level default metric function to compute mean(y_true - y_pred).
    Defined at module scope so it can be pickled when AutoGluon serializes the
    predictor.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return np.mean(y_t - y_p)

scorer = make_scorer(name="mean_diff", func=default_mean_diff, greater_is_better=True, optimum=1)

class EvalMetricWrapper:
    def __init__(self, func=None, needs_proba=False, name=None, greater_is_better_internal=True, worst_possible_value=None):
        """Wrap a metric function for AutoGluon. If no func provided, default to
        returning np.mean(y_true - y_pred, *args, **kwargs).
        Exposes common attributes AutoGluon checks (e.g., needs_proba, needs_quantile,
        greater_is_better_internal, name, worst_possible_value) to avoid attribute errors.
        """
        # Default metric: mean(y_true - y_pred) using module-level function (picklable)
        if func is None:
            func = default_mean_diff

        self.func = func
        # Attributes AutoGluon typically inspects
        self.needs_proba = needs_proba
        self.needs_pred = True
        self.needs_quantile = False
        # Some AutoGluon code checks for 'needs_class' on metric objects
        self.needs_class = False
        self.name = name or getattr(func, "__name__", "custom_metric")
        self.greater_is_better_internal = greater_is_better_internal
        # Expose both names for compatibility with Autogluon internal checks
        self.greater_is_better = greater_is_better_internal
        self.worst_possible_value = worst_possible_value
        # Provide `.error` and `.score` expected by some AutoGluon internals
        self.error = self.func
        self.score = self.func

    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.func(y_true, y_pred, *args, **kwargs)


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
        eval_metric = scorer


        # Build train/valid DataFrames with label column
        train_df = X.copy()
        train_df[label] = y.values
        valid_df = None
        if X_val is not None and y_val is not None:
            valid_df = X_val.copy()
            valid_df[label] = y_val.values

        combined_df = pd.concat([train_df, valid_df], axis=0)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        logger.info(f"combined_df: {len(combined_df)}")

        hyperparams = {
            "GBM": {},        # LightGBM defaults
            "XGB": {},        # XGBoost
            # "REALMLP": {},
            "TABICL": {}
        }

        logger.info(f"Training AutoGluon with label={label}, eval_metric={eval_metric}")
        self.predictor = TabularPredictor(label=label, 
                                eval_metric=eval_metric, 
                                problem_type='regression',
                                verbosity=3,
                                )
        self.predictor.fit(
            time_limit=7200,
            train_data=combined_df,
            presets='high',
            hyperparameters=hyperparams,
            dynamic_stacking=False,
            num_cpus=14,
            num_gpus=1,
            auto_stack=True,
            use_bag_holdout=True
        )
        results = self.predictor.fit_summary()
        logger.info(f"Model names: {self.predictor.model_names()}")
        logger.info(f"results: {results.to_dict()}")
        

        self.predictor.leaderboard(
            data=valid_df,
            extra_info=True,
            silent=True,
            display=True,
        )

        # self.predictor.distill(
        #     time_limit=7200,
        #     train_data=train_df,
        #     tuning_data=valid_df,
        #     presets=presets,
        #     hyperparameters=hyperparams,
        # )

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
                except Exception as e:
                    logger.warning(f"Failed to predict with model {m}: {e}")
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


