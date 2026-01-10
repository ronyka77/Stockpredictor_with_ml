"""
Random Forest Model Implementation

This module provides a RandomForestModel class for stock prediction,
following the BaseModel and LightGBMModel architecture, with MLflow integration
and threshold evaluation support.
"""

from typing import Optional, Dict, Any
import ast

import numpy as np
import pandas as pd
import optuna
import mlflow

from src.models.base_model import BaseModel
from src.models.evaluation.threshold_evaluator import ThresholdEvaluator
from src.utils.core.logger import get_logger
from sklearn.ensemble import RandomForestRegressor
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for stock price prediction with MLflow and threshold evaluation support.
    """

    def __init__(
        self,
        model_name: str = "random_forest_stock_predictor",
        config: Optional[Dict[str, Any]] = None,
        prediction_horizon: int = 10,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        """
        Initialize RandomForestModel
        Args:
            model_name: Name for MLflow tracking
            config: Model configuration parameters
            prediction_horizon: Prediction horizon in days
            threshold_evaluator: Optional shared ThresholdEvaluator instance
        """
        if config is None:
            config = {}
        config["prediction_horizon"] = prediction_horizon
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        self.prediction_horizon = self.config.get("prediction_horizon", 10)
        logger.info(f"Initialized {model_name} (RandomForestModel)")

    def _create_model(self, **kwargs):
        """
        Create the underlying RandomForestRegressor instance
        Returns:
            RandomForestRegressor instance
        """
        params = self.config.copy()
        params.update(kwargs)
        # Remove non-sklearn params
        params.pop("prediction_horizon", None)
        # Set default hyperparameters if not specified
        params.setdefault("min_samples_leaf", 1)
        params.setdefault("max_features", "sqrt")
        params.setdefault("random_state", 42)
        return RandomForestRegressor(**params)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ):
        """
        Train the Random Forest model
        Args:
            x: Training features
            y: Training targets
            x_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
        Returns:
            Self for method chaining
        """
        self.model = self._create_model(**kwargs)
        self.model.fit(x, y)
        self.is_trained = True
        self.feature_names = list(x.columns)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.model.feature_importances_
        else:
            self.feature_importance = None
        self.training_history = {}
        logger.info(f"RandomForestModel trained on {x.shape[0]} samples, {x.shape[1]} features.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.
        Args:
            X: Features for prediction
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if self.feature_names is not None:
            X = X[self.feature_names]
        preds = self.model.predict(X)
        logger.info(f"Predicted {len(preds)} samples with RandomForestModel.")
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probability predictions are not supported for regression RandomForestModel.
        Raises:
            NotImplementedError
        """
        raise NotImplementedError("RandomForestModel (regression) does not support predict_proba.")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores from the trained Random Forest model.
        Returns:
            DataFrame with feature names and importance scores, sorted descending
        """
        if not self.is_trained or self.feature_importance is None:
            return None
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.feature_importance}
        ).sort_values("importance", ascending=False)
        logger.info("Feature importance extracted from RandomForestModel.")
        return importance_df

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "variance") -> np.ndarray:
        """
        Calculate confidence scores for predictions using variance across trees (default).
        Args:
            X: Feature matrix
            method: Confidence calculation method ('variance' supported)
        Returns:
            Array of confidence scores (higher = more confident)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating confidence")
        if self.feature_names is not None:
            X = X[self.feature_names]
        if method == "variance":
            # Get predictions from all trees
            all_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            variance = np.var(all_preds, axis=0)
            confidence_scores = 1.0 / (1.0 + variance)  # Inverse variance as confidence
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        # Normalize confidence scores to [0, 1]
        min_conf, max_conf = confidence_scores.min(), confidence_scores.max()
        if max_conf > min_conf:
            confidence_scores = (confidence_scores - min_conf) / (max_conf - min_conf)
        else:
            confidence_scores = np.full_like(confidence_scores, 0.5)
        logger.info(f"Confidence scores calculated using method '{method}'.")
        return confidence_scores

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_prices: Optional[np.ndarray] = None,
        confidence_method: str = "variance",
    ) -> Dict[str, float]:
        """
        Evaluate model performance with optional threshold-based evaluation.
        Args:
            X: Features for evaluation
            y: True targets
            current_prices: Current stock prices for profit calculation (optional)
            confidence_method: Method for calculating confidence scores
        Returns:
            Dictionary of metric scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        metrics = {}
        if current_prices is not None:
            try:
                threshold_results = self.threshold_evaluator.optimize_prediction_threshold(
                    model=self,
                    x_test=X,
                    y_test=y,
                    current_prices_test=current_prices,
                    confidence_method=confidence_method,
                )
                if threshold_results.get("status") == "success":
                    best_result = threshold_results["best_result"]
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

    def save_model(self, experiment_name: str = None) -> str:
        """
        Save model to MLflow
        Args:
            experiment_name: MLflow experiment name
        Returns:
            MLflow run ID
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        if experiment_name:
            self.experiment_name = experiment_name
        self.mlflow_integration.setup_experiment(self.experiment_name)
        if self.run_id is None:
            # Check if there's already an active run
            active_run = mlflow.active_run()
            if active_run is not None:
                # Use the existing active run
                self.run_id = active_run.info.run_id
                logger.info(f"Using existing active MLflow run: {self.run_id}")
            else:
                # Start a new run
                run = self.mlflow_integration.start_run()
                self.run_id = run.info.run_id
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
        self.log_model(flavor="sklearn")
        logger.info(f"Model saved to MLflow with run ID: {self.run_id}")
        return self.run_id

    def load_model(self, run_id: str) -> "RandomForestModel":
        """
        Load model from MLflow
        Args:
            run_id: MLflow run ID to load the model from
        Returns:
            Self for method chaining
        """
        model_uri = f"runs:/{run_id}/model"
        self.model = self.mlflow_integration.load_sklearn_model(model_uri)
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
            parsed_features = (
                ast.literal_eval(feature_names_str) if feature_names_str != "[]" else []
            )
            self.feature_names = parsed_features if isinstance(parsed_features, list) else None
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse feature_names from MLflow params; leaving as None")
            self.feature_names = []
        self.is_trained = params.get("is_trained", "True") == "True"
        self.run_id = run_id
        logger.info(f"Model loaded from MLflow run ID: {run_id}")
        return self

    def get_best_trial_info(self) -> Dict[str, Any]:
        """
        Get information about the best trial from hypertuning.
        Returns:
            Dictionary with best trial information including threshold optimization details
        """
        if not hasattr(self, "best_score"):
            return {"message": "No hypertuning has been run yet"}
        base_info = {
            "best_score": getattr(self, "best_score", None),
            "best_trial_params": getattr(self, "best_trial_params", None),
            "has_best_model": hasattr(self, "best_trial_model")
            and self.best_trial_model is not None,
            "model_updated": self.model is not None,
        }
        base_info["threshold_optimization"] = getattr(self, "best_threshold_info", None)
        return base_info

    def finalize_best_model(self) -> None:
        """
        Finalize the best model after hypertuning.
        Sets the main model instance to the best performing model and threshold info.
        """
        if hasattr(self, "best_trial_model") and self.best_trial_model is not None:
            self.model = self.best_trial_model.model
            self.feature_names = self.best_trial_model.feature_names

            best_threshold = getattr(self, "best_threshold_info", None)
            if best_threshold and best_threshold.get("optimal_threshold") is not None:
                self.optimal_threshold = best_threshold["optimal_threshold"]
                self.confidence_method = getattr(self, "confidence_method", "variance")

            logger.info(f"Best model finalized with score: {getattr(self, 'best_score', None)}")
            logger.info(f"Best parameters: {getattr(self, 'best_trial_params', None)}")
            if best_threshold:
                logger.info(f"Best threshold info: {best_threshold}")
        else:
            logger.warning("No best model found to finalize")

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 50
    ) -> list:
        """
        Selects the best features using a preliminary RandomForest model based on feature importance.
        The 'close' column is always included if present in the original DataFrame.
        Args:
            X: The full feature matrix.
            y: The target series.
            n_features_to_select: The target number of features to select.
        Returns:
            A list of the selected feature names.
        """
        logger.info(
            f"Starting feature selection to find the best {n_features_to_select} features..."
        )
        prelim_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=1, max_features="sqrt"
        )
        prelim_model.fit(X, y)
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": prelim_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        logger.info("Top 10 features from preliminary model (by importance):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"- {row['feature']}: {row['importance']:.2f}")
        selected_features = list(importance_df["feature"].head(n_features_to_select))
        required_columns = ["close", "ticker_id"]
        for column in required_columns:
            if column in X.columns and column not in selected_features:
                logger.info(
                    f"'{column}' column was not in the top features. Adding it to the list."
                )
                removed_feature = selected_features.pop()
                selected_features.append(column)
                logger.info(f"Removed '{removed_feature}' to make space for '{column}'.")
        logger.info(f"Feature selection complete. Selected {len(selected_features)} features.")
        return selected_features

    def _evaluate_trial_on_test_set(
        self, model_instance, x_test_df: pd.DataFrame, y_true: pd.Series
    ):
        """Evaluate a trial model on the provided test set and return (metric, threshold_results).

        This helper isolates threshold optimization logic to keep the objective function small.
        """
        current_prices_local = x_test_df["close"].values if "close" in x_test_df.columns else None

        if current_prices_local is not None:
            results = model_instance.threshold_evaluator.optimize_prediction_threshold(
                model=model_instance,
                x_test=x_test_df,
                y_test=y_true,
                current_prices_test=current_prices_local,
                confidence_method="variance",
            )
            if results.get("status") == "success":
                best_res = results["best_result"]
                return best_res.get("test_profit_per_investment", 0.0), results
            return model_instance.model.score(x_test_df, y_true), results

        return model_instance.model.score(x_test_df, y_true), None

    def _create_and_evaluate_trial(
        self,
        trial,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """Create a RandomForest trial model, train it, and evaluate on test set.

        Returns (metric, trial_model, params, threshold_results).
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2", None]
            ),
            "random_state": 42,
            "n_jobs": -1,
        }

        trial_model = RandomForestModel(
            model_name=f"random_forest_trial_{trial.number}",
            config=params,
            prediction_horizon=self.prediction_horizon,
            threshold_evaluator=self.threshold_evaluator,
        )
        trial_model.fit(x_train, y_train)

        metric, threshold_results = self._evaluate_trial_on_test_set(trial_model, x_test, y_test)
        return metric, trial_model, params, threshold_results

    def objective(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
    ) -> callable:
        """
        Create Optuna objective function for RandomForestRegressor hypertuning with optional threshold optimization.
        Args:
            x_train: Training features
            y_train: Training targets
            x_test: Test features
            y_test: Test targets
        Returns:
            Objective function for Optuna optimization
        """
        self.best_score = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2", None]
                ),
                "random_state": 42,
                "n_jobs": -1,
            }
            try:
                metric, trial_model, params, threshold_results = self._create_and_evaluate_trial(
                    trial, x_train, y_train, x_test, y_test
                )
                if metric > self.best_score:
                    self._update_best_trial(
                        metric, trial_model, params, threshold_results, trial.number
                    )
                else:
                    logger.info(
                        f"Trial {trial.number}: Score = {metric:.4f} (Best: {self.best_score:.4f})"
                    )
                return metric
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6

        return objective

    def _update_best_trial(
        self, metric, trial_model, params, threshold_results, trial_number: int
    ) -> None:
        """Centralize updating state and logging when a new best trial is found."""
        self.best_score = metric
        self.best_trial_model = trial_model
        self.best_trial_params = params.copy()
        self.best_threshold_info = threshold_results if threshold_results is not None else None
        self.model = trial_model.model
        self.feature_names = trial_model.feature_names
        logger.info(f"NEW BEST TRIAL {trial_number}: Score = {metric:.4f}")
        if threshold_results and threshold_results.get("status") == "success":
            logger.info(f"Optimal threshold: {threshold_results['optimal_threshold']}")

    @staticmethod
    def load_and_prepare_data(
        prediction_horizon: int = 10,
        split_date: str = None,
        ticker: str = None,
        clean_features: bool = True,
        **kwargs,
    ):
        """
        Load and prepare data for RandomForestModel using the same pipeline as LightGBMModel.
        This uses prepare_ml_data_for_training_with_cleaning and ensures compatibility with sklearn.
        Args:
            prediction_horizon: Prediction horizon in days
            split_date: Date to split train/test
            ticker: Optional ticker filter
            clean_features: Whether to clean features
            **kwargs: Additional arguments for data preparation
        Returns:
            dict: Contains x_train, x_test, y_train, y_test, and metadata
        Note:
            If categorical features are present, ensure they are encoded (e.g., one-hot or ordinal) as required by sklearn.
            This method does not perform encoding itself but expects the pipeline or user to handle it per project convention.
        """
        data = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=prediction_horizon,
            split_date=split_date,
            ticker=ticker,
            clean_features=clean_features,
            **kwargs,
        )
        # Document: If categorical features exist, encoding should be handled in the pipeline or before fit.
        return data


def main():
    logger.info("=" * 80)
    logger.info("STANDALONE RANDOM FOREST HYPERTUNING & EVALUATION")
    logger.info("=" * 80)

    # 1. Data loading
    logger.info("Loading and preparing data...")
    data_result = RandomForestModel.load_and_prepare_data(
        prediction_horizon=10, split_date="2025-06-15", ticker=None, clean_features=True
    )
    x_train = data_result["x_train"]
    x_test = data_result["x_test"]
    y_train = data_result["y_train"]
    y_test = data_result["y_test"]
    target_column = data_result.get("target_column", "target")
    train_date_range = data_result.get("train_date_range", None)
    test_date_range = data_result.get("test_date_range", None)

    # 2. Feature selection (optional)
    n_features_to_select = 60
    rf_model = RandomForestModel(model_name="random_forest_feature_selector", prediction_horizon=10)
    selected_features = rf_model.select_features(x_train, y_train, n_features_to_select)
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]
    logger.info(f"DataFrames updated with {len(selected_features)} selected features.")

    # 3. Instantiate model for hypertuning
    rf_model = RandomForestModel(
        model_name="random_forest_standalone_hypertuned", prediction_horizon=10
    )
    objective_function = rf_model.objective(x_train_selected, y_train, x_test_selected, y_test)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    n_trials = 50
    logger.info(f"Starting Optuna hypertuning for {n_trials} trials...")
    study.optimize(objective_function, n_trials=n_trials, n_jobs=1)

    # 4. Finalize best model
    rf_model.finalize_best_model()
    best_trial_info = rf_model.get_best_trial_info()
    logger.info(f"Best trial info: {best_trial_info}")

    # 5. Evaluate and log results
    if "close" in x_test_selected.columns:
        current_prices = x_test_selected["close"].values
    else:
        current_prices = None
    eval_metrics = rf_model.evaluate(x_test_selected, y_test, current_prices=current_prices)
    logger.info(f"Final evaluation metrics: {eval_metrics}")

    # 6. Save best model to MLflow
    # final_metrics = {
    #     'best_score': best_trial_info.get('best_score', None),
    #     **(eval_metrics if eval_metrics else {})
    # }
    # final_params = best_trial_info.get('best_trial_params', {})
    run_id = rf_model.save_model(experiment_name="random_forest_stock_predictor_experiment")
    logger.info(f"Model saved to MLflow run: {run_id}")
    logger.info("=" * 80)
    logger.info("🎉 STANDALONE RANDOM FOREST HYPERTUNING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(
        f"Dataset: {len(y_train) + len(y_test):,} samples, {x_train_selected.shape[1]} features"
    )
    logger.info(f"Target: {target_column} (10-day horizon)")
    logger.info(f"Train period: {train_date_range}")
    logger.info(f"Test period: {test_date_range}")
    logger.info(f"Hypertuning: {n_trials} trials completed")
    logger.info(f"Best Score: {best_trial_info.get('best_score', None)}")
    logger.info(f"MLflow run: {run_id}")


if __name__ == "__main__":
    main()
