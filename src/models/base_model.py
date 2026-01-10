"""
Base Model Class

This module provides an abstract base class for all machine learning models
in the stock prediction system, with built-in MLflow tracking, persistence,
and evaluation capabilities.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import ast
from datetime import datetime

from src.models.evaluation.threshold_evaluator import ThresholdEvaluator
from src.utils.core.logger import get_logger
from src.utils.mlops.mlflow_integration import MLflowIntegration

logger = get_logger(__name__)


class DefaultHyperparameterConfig:
    """Small default hyperparameter config provider used when a model
    implementation does not provide its own hyperparameter configuration.
    """

    def get_default_params(self) -> dict:
        # Conservative defaults that work for lightweight tests
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}


class BaseModel(ABC):
    """
    Abstract base class for all ML models with MLflow integration
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        """
        Initialize base model

        Args:
            model_name: Name of the model for tracking
            config: Model configuration parameters
            threshold_evaluator: Optional shared ThresholdEvaluator instance
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.feature_importance = None
        self.training_history = {}

        # MLflow tracking
        self.mlflow_integration = MLflowIntegration()
        self.experiment_name = f"stock_prediction_{model_name}"
        self.run_id = None

        # Use provided ThresholdEvaluator or create a new one
        if threshold_evaluator is not None:
            self.threshold_evaluator = threshold_evaluator
        else:
            self.threshold_evaluator = ThresholdEvaluator()

        # Unified thresholding defaults
        self.base_threshold = 0.5
        # Provide a default hyperparameter_config object so gradient-boosting
        # implementations can call `self.hyperparameter_config.get_default_params()`
        # without having to instantiate their own config in tests.
        self.hyperparameter_config = DefaultHyperparameterConfig()

        logger.info(f"Initialized {model_name} model")

    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """
        Create the underlying model instance

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BaseModel":
        """
        Train the model

        Args:
            X: Training features
            y: Training targets
            x_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features for prediction

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if self.feature_names is not None:
            # Ensure feature order matches training
            X = X[self.feature_names]

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions (for classification models)

        Args:
            X: Features for prediction

        Returns:
            Probability predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("Model does not support probability predictions")

        if self.feature_names is not None:
            X = X[self.feature_names]

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained or self.feature_importance is None:
            return None

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.feature_importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self, experiment_name: str) -> str:
        """
        Save model to MLflow

        Args:
            experiment_name: MLflow experiment name

        Returns:
            MLflow run ID
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Set the experiment
        self.mlflow_integration.setup_experiment(experiment_name)

        # Start a new run if not already started
        if self.run_id is None:
            run = self.mlflow_integration.start_run()
            self.run_id = run.info.run_id

        # Log model parameters and metadata
        params = {
            "model_name": self.model_name,
            "config": str(self.config),
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "is_trained": self.is_trained,
        }
        if self.feature_names:
            params["feature_names"] = str(self.feature_names)
        if self.feature_importance is not None:
            params["has_feature_importance"] = True
        self.log_params(params)

        # Log the actual model
        self.log_model(flavor="sklearn")

        logger.info(f"Model saved to MLflow with run ID: {self.run_id}")
        return self.run_id

    def load_model(self, run_id: str) -> "BaseModel":
        """
        Load model from MLflow

        Args:
            run_id: MLflow run ID to load the model from

        Returns:
            Self for method chaining
        """

        # Load the model from MLflow
        model_uri = f"runs:/{run_id}/model"
        self.model = self.mlflow_integration.load_sklearn_model(model_uri)

        # Load run parameters
        run_info = self.mlflow_integration.get_run(run_id)
        params = run_info.data.params

        self.model_name = params.get("model_name", "Unknown")

        # Secure parsing for config
        config_str = params.get("config", "{}")
        try:
            parsed_config = ast.literal_eval(config_str) if config_str != "{}" else {}
            self.config = parsed_config if isinstance(parsed_config, dict) else {}
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse config from MLflow params; using empty dict")
            self.config = {}

        # Secure parsing for feature_names
        feature_names_str = params.get("feature_names", "[]")
        try:
            if feature_names_str != "[]":
                parsed_features = ast.literal_eval(feature_names_str)
                self.feature_names = parsed_features if isinstance(parsed_features, list) else None
            else:
                self.feature_names = None
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse feature_names from MLflow params; leaving as None")
            self.feature_names = None

        self.is_trained = params.get("is_trained", "True") == "True"
        self.run_id = run_id

        logger.info(f"Model loaded from MLflow run ID: {run_id}")
        return self

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_prices: Optional[np.ndarray] = None,
        confidence_method: str = "leaf_depth",
    ) -> Dict[str, float]:
        """
        Evaluate model performance with optional threshold-based evaluation

        Args:
            X: Features for evaluation
            y: True targets
            metrics_calculator: CustomMetrics instance
            current_prices: Current stock prices for profit calculation (optional)
            threshold_evaluator: ThresholdEvaluator instance for advanced evaluation (optional)
            use_threshold_optimization: Whether to use threshold optimization
            confidence_method: Method for calculating confidence scores

        Returns:
            Dictionary of metric scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        metrics = {}
        # Advanced threshold-based evaluation
        if current_prices is not None:
            try:
                # Perform threshold optimization
                threshold_results = self.threshold_evaluator.optimize_prediction_threshold(
                    model=self,
                    x_test=X,
                    y_test=y,
                    current_prices_test=current_prices,
                    confidence_method=confidence_method,
                )

                if threshold_results.get("status") == "success":
                    # Extract results directly from threshold optimization
                    best_result = threshold_results["best_result"]

                    # Add threshold-based metrics
                    metrics.update(
                        {
                            "threshold_optimized": True,
                            "optimal_threshold": threshold_results["optimal_threshold"],
                            "threshold_profit": best_result.get("test_profit_per_investment", 0.0),
                            "threshold_custom_accuracy": best_result.get(
                                "test_custom_accuracy", 0.0
                            ),
                            "threshold_investment_success_rate": best_result.get(
                                "test_investment_success_rate", 0.0
                            ),
                            "threshold_samples_kept_ratio": best_result.get(
                                "test_samples_kept_ratio", 0.0
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Threshold evaluation failed: {e}")
                metrics["threshold_evaluation_error"] = str(e)

        return metrics

    def optimize_prediction_threshold(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        current_prices_test: np.ndarray,
        confidence_method: str = "leaf_depth",
        threshold_range: Tuple[float, float] = (0.01, 0.99),
        n_thresholds: int = 90,
    ) -> Dict[str, Any]:
        """
        Unified wrapper: optimize prediction threshold via central evaluator.
        """
        results = self.threshold_evaluator.optimize_prediction_threshold(
            model=self,
            x_test=x_test,
            y_test=y_test,
            current_prices_test=current_prices_test,
            confidence_method=confidence_method,
            threshold_range=threshold_range,
            n_thresholds=n_thresholds,
        )
        # Store optimal settings if available
        if results.get("status") == "success":
            self.optimal_threshold = results.get("optimal_threshold")
            self.confidence_method = results.get("confidence_method", confidence_method)
        return results

    @staticmethod
    def _optuna_safe_set_user_attr(trial: Any, key: str, value: Any) -> None:
        """Best-effort Optuna trial user_attr setter.

        Notes:
        - Must remain JSON-serializable for sqlite-backed Optuna storage.
        - Never raise from tracking helpers.
        """
        try:
            if trial is None or not key:
                return
            trial.set_user_attr(str(key), value)
        except Exception:
            return

    @staticmethod
    def _optuna_to_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _optuna_to_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    def _optuna_log_dataset_sizes(self, trial: Any, x_train: Any, x_test: Any) -> None:
        """Log simple dataset sizes for dashboarding."""
        try:
            n_features = self._optuna_to_int(getattr(x_train, "shape", [None, None])[1])
            n_train = self._optuna_to_int(len(x_train))
            n_test = self._optuna_to_int(len(x_test))
        except Exception:
            return

        if n_features is not None:
            self._optuna_safe_set_user_attr(trial, "n_features", n_features)
        if n_train is not None:
            self._optuna_safe_set_user_attr(trial, "n_train", n_train)
        if n_test is not None:
            self._optuna_safe_set_user_attr(trial, "n_test", n_test)

    def _optuna_log_trial_core_metrics(
        self,
        trial: Any,
        *,
        profit_per_investment: Any = None,
        best_iteration: Any = None,
        best_score: Any = None,
    ) -> None:
        """Log core per-trial metrics into Optuna user attrs."""
        ppi = self._optuna_to_float(profit_per_investment)
        if ppi is not None:
            self._optuna_safe_set_user_attr(trial, "profit_per_investment", ppi)
        bi = self._optuna_to_int(best_iteration)
        if bi is not None:
            self._optuna_safe_set_user_attr(trial, "best_iteration", bi)
        bs = self._optuna_to_float(best_score)
        if bs is not None:
            self._optuna_safe_set_user_attr(trial, "best_score", bs)

    def _optuna_log_threshold_metrics(
        self,
        trial: Any,
        *,
        optimal_threshold: Any = None,
        samples_kept_ratio: Any = None,
        investment_success_rate: Any = None,
        custom_accuracy: Any = None,
        total_profit: Any = None,
        profitable_investments: Any = None,
        prefix: str = "threshold_",
    ) -> None:
        """Log threshold-optimization metrics into Optuna user attrs."""
        opt_thr = self._optuna_to_float(optimal_threshold)
        if opt_thr is not None:
            self._optuna_safe_set_user_attr(trial, f"{prefix}optimal_threshold", opt_thr)

        skr = self._optuna_to_float(samples_kept_ratio)
        if skr is not None:
            self._optuna_safe_set_user_attr(trial, f"{prefix}samples_kept_ratio", skr)

        isr = self._optuna_to_float(investment_success_rate)
        if isr is not None:
            self._optuna_safe_set_user_attr(trial, f"{prefix}investment_success_rate", isr)

        ca = self._optuna_to_float(custom_accuracy)
        if ca is not None:
            self._optuna_safe_set_user_attr(trial, f"{prefix}custom_accuracy", ca)

        tp = self._optuna_to_float(total_profit)
        if tp is not None:
            self._optuna_safe_set_user_attr(trial, f"{prefix}total_profit", tp)

        pi = self._optuna_to_int(profitable_investments)
        if pi is not None:
            self._optuna_safe_set_user_attr(trial, f"{prefix}profitable_investments", pi)

    def create_optuna_study(
        self,
        *,
        prediction_horizon: Optional[int] = None,
        direction: str = "maximize",
        objective: str = "profit_per_investment",
        objective_slug: str = "profit",
        split_date: Optional[str] = None,
        target_column: Optional[str] = None,
        storage_url: Optional[str] = None,
        study_name: Optional[str] = None,
        load_if_exists: bool = True,
        sampler: Any = None,
        pruner: Any = None,
        n_features: Optional[int] = None,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        user_attrs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create/load a persistent Optuna study for this model.

        This is a convenience hook to keep Optuna setup consistent across models.
        """
        from src.utils.mlops.optuna_utils import create_or_load_study

        ph = prediction_horizon
        if ph is None:
            try:
                ph = int(self.config.get("prediction_horizon", 0) or 0)
            except Exception:
                ph = 0

        if n_features is None and self.feature_names is not None:
            try:
                n_features = int(len(self.feature_names))
            except Exception:
                n_features = None

        return create_or_load_study(
            model_name=self.model_name,
            prediction_horizon=int(ph),
            direction=direction,
            objective=objective,
            objective_slug=objective_slug,
            split_date=split_date,
            target_column=target_column,
            storage_url=storage_url,
            study_name=study_name,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner,
            n_features=n_features,
            n_train=n_train,
            n_test=n_test,
            user_attrs=user_attrs,
        )

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False,
        threshold: Optional[float] = None,
        confidence_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified wrapper: make predictions with confidence-based filtering via central evaluator.
        """
        # Defaults: prefer optimal_threshold if available to preserve behavior
        if threshold is None:
            threshold = getattr(self, "optimal_threshold", getattr(self, "base_threshold", 0.5))
        if confidence_method is None:
            confidence_method = getattr(
                self, "confidence_method", getattr(self, "default_confidence_method", "leaf_depth")
            )

        return self.threshold_evaluator.predict_with_threshold(
            model=self,
            X=X,
            threshold=threshold,
            confidence_method=confidence_method,
            return_confidence=return_confidence,
        )

    def start_mlflow_run(self, experiment_name: Optional[str] = None) -> None:
        """
        Start MLflow tracking run

        Args:
            experiment_name: Name of the experiment (optional)
        """
        if experiment_name:
            self.experiment_name = experiment_name

        try:
            # Setup experiment using MLflow integration
            self.mlflow_integration.setup_experiment(self.experiment_name)

            # Start run
            run_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = self.mlflow_integration.start_run(run_name=run_name)
            self.run_id = run.info.run_id

            logger.info(f"Started MLflow run: {self.run_id}")
        except Exception as e:
            logger.warning(f"MLflow run start failed: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow

        Args:
            params: Parameters to log
        """
        try:
            self.mlflow_integration.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow param logging failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow

        Args:
            metrics: Metrics to log
            step: Step number (optional)
        """
        try:
            self.mlflow_integration.log_metrics(metrics, step=step)
            logger.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"MLflow metric logging failed: {e}")

    def log_model(self, flavor: str = "sklearn") -> None:
        """
        Log model to MLflow

        Args:
            flavor: MLflow model flavor (sklearn, xgboost, etc.)
        """
        try:
            if self.is_trained:
                self.mlflow_integration.log_model(
                    self.model,
                    artifact_path=self.model_name,
                    flavor=flavor,
                    registered_model_name=f"{self.model_name}_stock_prediction",
                )
                logger.info(f"Logged {flavor} model to MLflow")
        except Exception as e:
            logger.warning(f"MLflow model logging failed: {e}")

    def end_mlflow_run(self) -> None:
        """
        End MLflow tracking run
        """
        try:
            import mlflow

            if mlflow.active_run():
                mlflow.end_run()
                logger.info(f"Ended MLflow run: {self.run_id}")
        except Exception as e:
            logger.warning(f"MLflow run end failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information summary

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "config": self.config,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "run_id": self.run_id,
            "training_history": self.training_history,
        }
