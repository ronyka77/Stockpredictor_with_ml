"""
AutoGluon Model Wrapper

Provides a thin adapter around AutoGluon TabularPredictor to conform to
our project BaseModel/ModelProtocol expectations, including:
- predict(X: pd.DataFrame) -> np.ndarray
- get_prediction_confidence(X: pd.DataFrame, method: str) -> np.ndarray

Also exposes fit with (X, y, X_val, y_val) by reconstructing label columns
for AutoGluon training.
"""

import gc
import os
import psutil
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer


from src.models.base_model import BaseModel
from src.models.evaluation.threshold_evaluator import ModelProtocol, ThresholdEvaluator
from src.utils.logger import get_logger


logger = get_logger(__name__)

# Standardized error messages
PREDICTOR_NOT_TRAINED = "Predictor not trained/loaded"


def default_mean_diff(y_true, y_pred, *args, **kwargs):
    """Top-level default metric function to compute mean(y_true - y_pred).
    Defined at module scope so it can be pickled when AutoGluon serializes the
    predictor.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return np.mean(y_p - y_t)


def conditional_mean_absolute_error(y_true, y_pred, *args, **kwargs):
    """Mean absolute error with conditional zeroing rules.

    Rules:
    - If y_true > 0 and y_pred > y_true then contribution = 0
    - If y_true < 0 and y_pred < y_true then contribution = 0

    Returns the mean of absolute errors after applying the rules.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    # Ensure same shape
    if y_t.shape != y_p.shape:
        y_p = np.reshape(y_p, y_t.shape)

    abs_errors = np.abs(y_t - y_p)

    # Apply conditional zeroing
    mask_positive_over = (y_t > 0) & (y_t > y_p)
    mask_negative_under = (y_t < 0) & (y_p < y_t)

    abs_errors[mask_positive_over] = 0.0
    abs_errors[mask_negative_under] = 0.0

    return float(np.mean(abs_errors))


scorer = make_scorer(
    name="mean_diff", score_func=default_mean_diff, greater_is_better=True, optimum=1
)

mae_scorer = make_scorer(
    name="conditional_mean_absolute_error",
    score_func=conditional_mean_absolute_error,
    greater_is_better=False,
    optimum=0,
)


class AutoGluonModel(BaseModel, ModelProtocol):
    def __init__(
        self, *, model_name: str = "autogluon", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name=model_name, config=config or {})
        self.predictor: Optional[TabularPredictor] = None
        self.feature_names: Optional[list[str]] = None
        self.selected_model_name: Optional[str] = None
        self.optimal_threshold: Optional[float] = None

    def _create_model(self, **kwargs) -> TabularPredictor:
        # AutoGluon predictor created during fit with label specified; no pre-instance here
        return None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "AutoGluonModel":
        label = self.config.get("label", "Future_Return_10D")
        groups = self.config.get("groups", "year")
        eval_metric = mae_scorer
        train_df = X.copy()

        train_df[label] = y.values
        # Backwards-compatible handling for callers using the old keyword name `X_val`
        if x_val is None and "X_val" in kwargs:
            x_val = kwargs.pop("X_val")
        if y_val is None and "y_val" in kwargs:
            y_val = kwargs.pop("y_val")

        valid_df = None
        if x_val is not None and y_val is not None:
            valid_df = x_val.copy()
            valid_df[label] = y_val.values

        hyperparams = {
            "FASTAI": {},
            "GBM": {},
            "XGB": {},
            # "TABM": {}, # noqa: E501
            # "RF": {}, # noqa: E501
            # "CAT": {'task_type': 'GPU'} # noqa: E501
            # "REALMLP": {}, # noqa: E501
        }

        logger.info(f"Training AutoGluon with label={label}, eval_metric={eval_metric}")
        self.predictor = TabularPredictor(
            label=label,
            eval_metric=eval_metric,
            problem_type="regression",
            verbosity=2,
        )
        collected = gc.collect()
        logger.info(f"Garbage collected: {collected}")
        proc = psutil.Process(os.getpid())
        logger.info(f"RSS MB: {proc.memory_info().rss / 1024**2}")

        self.predictor.fit(
            time_limit=39600,
            train_data=train_df,
            tuning_data=valid_df,
            presets="best_quality",
            hyperparameters=hyperparams,
            dynamic_stacking=False,
            num_gpus=1,
            num_stack_levels=2,
            num_bag_folds=4,
            use_bag_holdout=True,
            fit_strategy="parallel",
            ag_args_ensemble={"fold_fitting_strategy": "parallel_local"},
        )
        summary = self.predictor.fit_summary(show_plot=True)
        logger.info(summary)

        # Capture feature names (exclude label)
        self.feature_names = [c for c in train_df.columns if c != label and c != groups]
        self.model = self
        self.is_trained = True
        logger.info("AutoGluon training completed")
        return self

    # ModelProtocol requirement
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.predictor:
            raise ValueError(PREDICTOR_NOT_TRAINED)
        if self.feature_names is not None:
            X = X[self.feature_names]

        if self.selected_model_name is not None:
            preds = self.predictor.predict(X, model=self.selected_model_name)
        else:
            self.selected_model_name = self.predictor.model_best
            preds = self.predictor.predict(X, model=self.selected_model_name)
        return preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)

    # ModelProtocol requirement
    def get_prediction_confidence(
        self, X: pd.DataFrame, method: str = "margin"
    ) -> np.ndarray:
        if not self.predictor:
            raise ValueError(PREDICTOR_NOT_TRAINED)

        if self.feature_names is not None:
            X = X[self.feature_names]

        method = method.lower()
        if method == "variance":
            try:
                p = self.predictor.predict(X)
                p = p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)
                return p
            except Exception as e:
                logger.warning(
                    f"Failed to predict with model {self.selected_model_name}: {e}"
                )
                return np.ones(len(X))
        if method == "simple":
            base = self.predict(X)
            base_arr = np.asarray(base)
            # Compute max and guard against zero (or near-zero) to avoid div-by-zero
            try:
                max_val = base_arr.max()
            except Exception:
                # Fallback in case base_arr has unexpected structure
                max_val = float(np.max(base_arr))

            # Determine output dtype: preserve float dtype, otherwise cast to float
            out_dtype = (
                base_arr.dtype if np.issubdtype(base_arr.dtype, np.floating) else float
            )

            if np.isclose(max_val, 0):
                # Return zeros with same shape and an appropriate dtype
                return np.zeros_like(base_arr, dtype=out_dtype)

            return np.abs(base_arr / float(max_val))
        if method == "margin":
            base = np.abs(self.predict(X))
            return np.tanh(base)

        raise ValueError(f"Unsupported confidence method: {method}")

    def run_threshold_evaluation(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        confidence_method: str = "margin",
        threshold_range: tuple = (0.01, 0.99),
        n_thresholds: int = 90,
        investment_amount: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Run centralized threshold optimization using ThresholdEvaluator.
        Returns the evaluator results dict.
        """
        if not self.predictor:
            raise ValueError(PREDICTOR_NOT_TRAINED)

        evaluator = ThresholdEvaluator(investment_amount=investment_amount)

        # Determine current prices
        if "close" in X_test.columns:
            current_prices = X_test["close"].to_numpy()
        elif "current_price" in X_test.columns:
            current_prices = X_test["current_price"].to_numpy()
        else:
            current_prices = np.ones(len(X_test))

        logger.info(
            f"Running threshold evaluation: method={confidence_method}, thresholds={str(threshold_range)}, n={n_thresholds}"
        )
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
            return results
        except Exception as e:
            logger.error(f"Threshold evaluation failed: {e}")
            return {"status": "failed", "message": str(e)}

    def load_from_dir(self, model_dir: str) -> "AutoGluonModel":
        logger.info(f"Loading AutoGluon predictor from {model_dir}")
        self.predictor = TabularPredictor.load(model_dir)
        self.model = self
        self.feature_names = self.predictor.original_features
        self.is_trained = True
        logger.info(f"AutoGluon predictor loaded from {model_dir}")
        return self
