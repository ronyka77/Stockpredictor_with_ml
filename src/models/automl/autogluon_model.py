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
        label = self.config.get("label", "Future_Return_10D")
        eval_metric = scorer

        # Build train/valid DataFrames with label column
        train_df = X.copy()
        train_df[label] = y.values
        valid_df = None
        if X_val is not None and y_val is not None:
            valid_df = X_val.copy()
            valid_df[label] = y_val.values

        combined_df = pd.concat([train_df, valid_df], axis=0)
        combined_df = combined_df.reset_index(drop=True)
        logger.info(f"combined_df: {len(combined_df)}")

        hyperparams = {
            # "FASTAI": {},
            "GBM": {'verbosity': -1},        
            # "TABM": {}, 
            "RF": {'verbose': 0},       
            # "CAT": {'task_type': 'GPU'}
            # "REALMLP": {},
        }

        logger.info(f"Training AutoGluon with label={label}, eval_metric={eval_metric}")
        self.predictor = TabularPredictor(label=label, 
                                eval_metric='mae', 
                                problem_type='regression',
                                verbosity=2)
        
        self.predictor.fit(
            time_limit=39600,
            train_data=train_df,
            tuning_data=valid_df,
            presets='best_quality',
            hyperparameters=hyperparams,
            dynamic_stacking=False,
            # num_cpus=14,
            num_gpus=1,
            auto_stack=True,
            num_bag_folds=10,
            use_bag_holdout=True,
            # ag_args={'fold_fitting_strategy': 'sequential_local'},
            ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'}
        )
        summary = self.predictor.fit_summary(show_plot=True)
        logger.info(summary)

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
            base_arr = np.asarray(base)
            # Compute max and guard against zero (or near-zero) to avoid div-by-zero
            try:
                max_val = base_arr.max()
            except Exception:
                # Fallback in case base_arr has unexpected structure
                max_val = float(np.max(base_arr))

            # Determine output dtype: preserve float dtype, otherwise cast to float
            out_dtype = base_arr.dtype if np.issubdtype(base_arr.dtype, np.floating) else float

            if np.isclose(max_val, 0):
                # Return zeros with same shape and an appropriate dtype
                return np.zeros_like(base_arr, dtype=out_dtype)

            return np.abs(base_arr / float(max_val))
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
        """
        Run centralized threshold optimization using ThresholdEvaluator.
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

        logger.info(f"Running threshold evaluation: method={confidence_method}, thresholds={str(threshold_range)}, n={n_thresholds}")
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
            logger.info(f"Threshold evaluation finished. status={results.get('status', 'unknown')}")
            return results
        except Exception as e:
            logger.error(f"Threshold evaluation failed: {e}")
            return {'status': 'failed', 'message': str(e)}

    def load_from_dir(self, model_dir: str) -> "AutoGluonModel":
        logger.info(f"Loading AutoGluon predictor from {model_dir}")
        self.predictor = TabularPredictor.load(model_dir)
        self.model = self
        self.feature_names = self.predictor.original_features
        self.is_trained = True
        logger.info(f"AutoGluon predictor loaded from {model_dir}")
        return self


