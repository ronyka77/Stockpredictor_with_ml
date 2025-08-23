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
from src.models.evaluation.threshold_evaluator import ModelProtocol, ThresholdEvaluator
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

scorer = make_scorer(name="mean_diff", score_func=default_mean_diff, greater_is_better=True, optimum=1)

class AutoGluonModel(BaseModel, ModelProtocol):
    def __init__(self, *,
                model_name: str = "autogluon",
                config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config or {})
        self.predictor: Optional[TabularPredictor] = None
        self.feature_names: Optional[list[str]] = None
        self.selected_model_name: Optional[str] = None
        self.optimal_threshold: Optional[float] = None

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

        # train_df = train_df.sort_values('date_int', ascending=False).head(200000).reset_index(drop=True)
        combined_df = pd.concat([train_df, valid_df], axis=0)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        logger.info(f"combined_df: {len(combined_df)}")

        hyperparams = {
            "REALMLP": {},
            "GBM": {},        
            "XGB": {},        
            "TABICL": {}
        }

        logger.info(f"Training AutoGluon with label={label}, eval_metric={eval_metric}")
        self.predictor = TabularPredictor(label=label, 
                                eval_metric=eval_metric, 
                                problem_type='regression',
                                verbosity=3)
        
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

        # self.predictor.distill(
        #     time_limit=7200,
        #     train_data=train_df,
        #     tuning_data=valid_df,
        #     presets=presets,
        #     hyperparameters=hyperparams,
        # )

        # Capture feature names (exclude label)
        self.feature_names = [c for c in train_df.columns if c != label]
        self.model = self
        self.is_trained = True
        logger.info("AutoGluon training completed")
        return self

    # ModelProtocol requirement
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.predictor:
            raise ValueError("Predictor not trained/loaded")
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        if self.selected_model_name is not None:
            preds = self.predictor.predict(X, model=self.selected_model_name)
        else:
            self.selected_model_name = self.predictor.model_best
            preds = self.predictor.predict(X, model=self.selected_model_name)
        return preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)

    # ModelProtocol requirement
    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'margin') -> np.ndarray:
        if not self.predictor:
            raise ValueError("Predictor not trained/loaded")

        if self.feature_names is not None:
            X = X[self.feature_names]

        method = method.lower()
        if method == 'variance':
            try:
                p = self.predictor.predict(X)
                p = p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)
                return p
            except Exception as e:
                logger.warning(f"Failed to predict with model {self.selected_model_name}: {e}")
                return np.ones(len(X))
        if method == 'simple':
            base = self.predict(X)
            return np.abs(base / base.max())
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

    def run_threshold_evaluation(self,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                confidence_method: str = 'margin',
                                threshold_range: tuple = (0.01, 0.99),
                                n_thresholds: int = 90,
                                investment_amount: float = 100.0) -> Dict[str, Any]:
        """Run centralized threshold optimization using ThresholdEvaluator.

        Returns the evaluator results dict.
        """
        if not self.predictor:
            raise ValueError("Predictor not trained/loaded")

        evaluator = ThresholdEvaluator(investment_amount=investment_amount)

        # Determine current prices
        if 'close' in X_test.columns:
            current_prices = X_test['close'].to_numpy()
        elif 'current_price' in X_test.columns:
            current_prices = X_test['current_price'].to_numpy()
        else:
            current_prices = np.ones(len(X_test))

        logger.info("Running threshold evaluation: method=%s, thresholds=%s, n=%d",
                    confidence_method, str(threshold_range), n_thresholds)
        try:
            results = evaluator.optimize_prediction_threshold(
                model=self,
                X_test=X_test,
                y_test=y_test,
                current_prices_test=current_prices,
                confidence_method=confidence_method,
                threshold_range=threshold_range,
                n_thresholds=n_thresholds,
            )
            logger.info("Threshold evaluation finished. status=%s", results.get('status', 'unknown'))
            return results
        except Exception as e:
            logger.error("Threshold evaluation failed: %s", e)
            return {'status': 'failed', 'message': str(e)}


    def load_from_dir(self, model_dir: str) -> "AutoGluonModel":
        logger.info("Loading AutoGluon predictor from %s", model_dir)
        self.predictor = TabularPredictor.load(model_dir)
        self.model = self
        self.feature_names = self.predictor.original_features
        self.is_trained = True
        logger.info("AutoGluon predictor loaded from %s", model_dir)
        return self


