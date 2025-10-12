"""
MLP Main Module

This module contains the main function and MLflow integration for standalone MLP execution.
Includes the complete pipeline for hypertuning and evaluation.
"""

import torch
import torch.nn as nn
import pandas as pd
import mlflow
import mlflow.pytorch
import optuna
import pickle
import os
from datetime import datetime
from typing import Dict, Any
import numpy as np  # noqa: F401 (used in smoke test via alias)
from sklearn.preprocessing import StandardScaler  # noqa: F401 (used in smoke test via alias)

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLFlowManager
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.models.time_series.mlp.mlp_evaluation import MLPEvaluationMixin
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin

logger = get_logger(__name__)
experiment_name = "mlp_stock_predictor"


class MLPWrapper(nn.Module):
    """
    PyTorch Module wrapper for MLPPredictorWithMLflow to enable MLflow compatibility.

    This wrapper preserves all functionality of the original predictor while making it
    compatible with MLflow's PyTorch model logging requirements.
    """

    def __init__(self, predictor):
        """
        Initialize the wrapper with a MLPPredictorWithMLflow instance.

        Args:
            predictor: MLPPredictorWithMLflow instance to wrap
        """
        super().__init__()
        self.predictor = predictor
        self.model = predictor.model
        if self.model is None:
            raise ValueError("Predictor must have a trained model")

        # Preserve all predictor attributes for compatibility
        self.scaler = getattr(predictor, "scaler", None)
        self.feature_names = getattr(predictor, "feature_names", [])
        self.config = getattr(predictor, "config", {})
        self.device = getattr(
            predictor,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model_name = getattr(predictor, "model_name", "MLP")
        self.threshold_evaluator = getattr(predictor, "threshold_evaluator", None)
        self.is_trained = getattr(predictor, "is_trained", False)

        # Preserve threshold optimization attributes
        self.optimal_threshold = getattr(predictor, "optimal_threshold", 0.5)
        self.confidence_method = getattr(predictor, "confidence_method", "variance")
        self.best_threshold_info = getattr(predictor, "best_threshold_info", {})

        logger.info("✅ MLPWrapper initialized with all predictor attributes preserved")

    def forward(self, x):
        """
        Forward pass through the PyTorch model.

        Args:
            x: Input tensor

        Returns:
            Model predictions
        """
        return self.model(x)

    def predict(self, features_df):
        """
        Make predictions using the wrapped predictor's logic.

        Args:
            features_df: Input features DataFrame

        Returns:
            Predictions as numpy array
        """
        return self.predictor.predict(features_df)

    def get_prediction_confidence(self, features_df, method="variance"):
        """
        Get prediction confidence scores using the wrapped predictor's logic.

        Args:
            features_df: Input features DataFrame
            method: Confidence calculation method

        Returns:
            Confidence scores as numpy array
        """
        return self.predictor.get_prediction_confidence(features_df, method)

    def predict_with_threshold(
        self, X, return_confidence=False, threshold=None, confidence_method=None
    ):
        """
        Make predictions with confidence-based filtering.

        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            threshold: Confidence threshold
            confidence_method: Confidence method

        Returns:
            Dictionary with predictions and filtering info
        """
        return self.predictor.predict_with_threshold(
            X, return_confidence, threshold, confidence_method
        )


class MLPPredictorWithMLflow(MLPPredictor, MLPEvaluationMixin, MLPOptimizationMixin):
    """
    MLPPredictor with MLflow integration capabilities.
    """

    def save_model(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        x_eval: pd.DataFrame,
        experiment_name: str = None,
        scaler=None,
    ) -> str:
        """
        Save the trained MLP model to MLflow with all necessary components

        Args:
            metrics: Model evaluation metrics to log
            params: Model parameters to log
            x_eval: Evaluation features for signature generation
            experiment_name: Experiment name (uses default if None)
            scaler: Fitted StandardScaler instance to save with model

        Returns:
            MLflow run ID where the model was saved
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")

        if experiment_name is None:
            experiment_name = "mlp_stock_predictor"

        try:
            # End any existing run before starting a new one
            try:
                mlflow.end_run()
                logger.info("Ended existing MLflow run before starting new one")
            except Exception as e:
                logger.info(f"No existing run to end: {e}")

            # Set up MLflow tracking
            mlflow.set_experiment(experiment_name)

            # Start a new run
            with mlflow.start_run(
                run_name=f"mlp_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
            ) as run:
                # Log parameters (excluding base params if desired)
                params_to_log = {
                    k: v
                    for k, v in params.items()
                    if k
                    not in [
                        "device",
                        "objective",
                        "verbosity",
                        "seed",
                        "nthread",
                        "verbose",
                    ]
                }
                mlflow.log_params(params_to_log)
                mlflow.log_metrics(metrics)

                if scaler is not None:
                    try:
                        # Create temporary directory and save scaler with specific name
                        temp_dir = "src/models/scalers"
                        os.makedirs(temp_dir, exist_ok=True)
                        scaler_path = os.path.join(temp_dir, "scaler.pkl")

                        with open(scaler_path, "wb") as f:
                            pickle.dump(scaler, f)

                        mlflow.log_artifact(scaler_path, artifact_path="preprocessor")

                        # Clean up temp files
                        os.unlink(scaler_path)
                        logger.info("✅ Feature scaler saved to MLflow artifacts")
                    except Exception as scaler_error:
                        logger.warning(f"⚠️ Could not save scaler: {str(scaler_error)}")
                else:
                    logger.info(
                        "ℹ️ No scaler provided - model will use raw features for prediction"
                    )

                # Create input example using the DataFrame x_eval
                input_example = x_eval.iloc[:5].copy()

                # Identify and convert integer columns to float64
                if hasattr(input_example, "dtypes"):
                    for col in input_example.columns:
                        if input_example[col].dtype.kind == "i":
                            logger.info(
                                f"Converting integer column '{col}' to float64 for signature"
                            )
                            input_example[col] = input_example[col].astype("float64")

                # Infer signature using direct model inference on CPU to avoid device issues
                try:
                    self.model.eval()
                    try:
                        original_device = next(self.model.parameters()).device
                    except Exception:
                        original_device = torch.device("cpu")
                    self.model.to(torch.device("cpu"))
                    with torch.no_grad():
                        input_tensor = torch.as_tensor(
                            input_example.values, dtype=torch.float32
                        )
                        predictions_example = self.model(input_tensor).cpu().numpy()
                    signature = mlflow.models.infer_signature(
                        input_example, predictions_example
                    )
                finally:
                    try:
                        if original_device is not None:
                            self.model.to(original_device)
                    except Exception as e:
                        logger.error(f"Error setting model to original device: {e}")
                        pass

                # Persist feature names explicitly for robust loading
                try:
                    feature_names = list(x_eval.columns)
                    mlflow.log_dict(
                        {"feature_names": feature_names},
                        "preprocessor/feature_names.json",
                    )
                    logger.info(
                        "✅ Feature names saved to MLflow artifacts (preprocessor/feature_names.json)"
                    )
                except Exception as fn_err:
                    logger.warning(f"⚠️ Could not save feature names: {str(fn_err)}")

                # Log the raw PyTorch model at a stable artifact path
                try:
                    try:
                        original_device = next(self.model.parameters()).device
                    except Exception:
                        original_device = torch.device("cpu")
                    self.model.to(torch.device("cpu"))
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                    )
                    logger.info(
                        "✅ PyTorch model successfully logged to MLflow at artifact path 'model'"
                    )
                except Exception as e:
                    logger.error(f"❌ Model logging failed: {str(e)}")
                    raise
                finally:
                    try:
                        if original_device is not None:
                            self.model.to(original_device)
                    except Exception as e:
                        logger.error(f"Error setting model to original device: {e}")
                        pass

                logger.info(
                    f"✅ MLP model saved successfully. Run ID: {run.info.run_id}"
                )
                return run.info.run_id

        except Exception as e:
            logger.error(f"Error saving model to MLflow: {str(e)}")
            return None

    def load_model(self, run_id: str, experiment_name: str = None) -> bool:
        """
        Load a trained MLP model from MLflow based on the given run ID

        Args:
            run_id: MLflow run ID to load the model from
            experiment_name: Experiment name (uses default if None)

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if experiment_name is None:
                experiment_name = "mlp_stock_predictor"

            # Set the experiment
            mlflow.set_experiment(experiment_name)

            # Load model directly from the run's artifact path (stable URI)
            try:
                model_uri = f"runs:/{run_id}/model"
                self.model = mlflow.pytorch.load_model(model_uri)
                self.model.to(self.device)
                logger.info("✅ Model loaded successfully from runs URI")
            except Exception as model_error:
                logger.error(
                    f"❌ Failed to load model from runs URI: {str(model_error)}"
                )
                return False

            # Load preprocessor artifacts
            scaler_loaded = False
            feature_names_loaded = False
            try:
                scaler_local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/preprocessor/scaler.pkl"
                )
                with open(scaler_local_path, "rb") as f:
                    self.scaler = pickle.load(f)
                scaler_loaded = True
                logger.info(
                    "✅ Feature scaler loaded from MLflow artifacts (preprocessor/scaler.pkl)"
                )
            except Exception as scaler_error:
                logger.info(
                    f"ℹ️ No scaler artifact found or failed to load: {scaler_error}"
                )

            try:
                feature_names_local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/preprocessor/feature_names.json"
                )
                import json

                with open(feature_names_local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "feature_names" in data:
                    self.feature_names = data["feature_names"]
                    feature_names_loaded = True
                    logger.info(
                        f"✅ Loaded {len(self.feature_names)} feature names from artifacts"
                    )
            except Exception as fn_error:
                logger.info(
                    f"ℹ️ No feature_names artifact found or failed to load: {fn_error}"
                )

            # Fallback: attempt to extract from model signature if not loaded
            if not feature_names_loaded:
                try:
                    self._extract_feature_names_from_model(model_uri, "runs model")
                except Exception:
                    pass

            # Mark as trained
            self.is_trained = True

            logger.info("✅ Model loaded successfully")
            if run_id:
                logger.info(f"   Run ID: {run_id}")
            logger.info(f"   Experiment: {experiment_name}")
            logger.info(f"   Scaler loaded: {scaler_loaded}")
            logger.info(
                f"   Feature names loaded: {len(self.feature_names) if hasattr(self, 'feature_names') and self.feature_names else 0}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    def _extract_feature_names_from_model(self, model_uri: str, source: str):
        """
        Extract feature names from model signature

        Args:
            model_uri: Model URI to extract signature from
            source: Source description for logging
        """
        feature_names_from_signature = []
        try:
            # Direct approach: get signature from model URI
            logger.info(f"Attempting to load model signature from: {model_uri}")

            # Load model info to get signature
            from mlflow.models import get_model_info

            model_info = get_model_info(model_uri)
            logger.info(f"Model info loaded: {model_info is not None}")

            if model_info and model_info.signature:
                logger.info(
                    f"Model signature found: {model_info.signature is not None}"
                )

                if model_info.signature.inputs:
                    logger.info(
                        f"Signature inputs found: {len(model_info.signature.inputs.inputs) if hasattr(model_info.signature.inputs, 'inputs') else 'No inputs attr'}"
                    )

                    # Extract feature names from signature inputs
                    # Handle different signature input formats
                    if hasattr(model_info.signature.inputs, "inputs"):
                        # Schema format
                        for input_spec in model_info.signature.inputs.inputs:
                            if hasattr(input_spec, "name") and input_spec.name:
                                feature_names_from_signature.append(input_spec.name)
                    elif hasattr(model_info.signature.inputs, "schema"):
                        # Alternative schema format
                        if hasattr(model_info.signature.inputs.schema, "input_names"):
                            feature_names_from_signature = (
                                model_info.signature.inputs.schema.input_names
                            )
                    else:
                        logger.info(
                            f"Signature inputs type: {type(model_info.signature.inputs)}"
                        )
                        logger.info(
                            f"Signature inputs attributes: {dir(model_info.signature.inputs)}"
                        )

                    if feature_names_from_signature:
                        self.feature_names = feature_names_from_signature
                        logger.info(
                            f"✅ Loaded {len(self.feature_names)} feature names from {source} signature"
                        )
                    else:
                        logger.warning(
                            f"⚠️ No feature names found in {source} signature inputs"
                        )
                else:
                    logger.warning(f"⚠️ {source} signature has no inputs")
            else:
                logger.warning(f"⚠️ No model signature found in {source} model info")

        except Exception as signature_error:
            logger.warning(
                f"⚠️ Could not extract feature names from {source} signature: {str(signature_error)}"
            )
            logger.info(
                f"Signature error details: {type(signature_error).__name__}: {signature_error}"
            )

            # Fallback: Try to extract feature names from the loaded model's input layer
            try:
                logger.info(
                    f"🔄 Attempting fallback: extracting feature names from {source} model architecture..."
                )
                if hasattr(self.model, "input_size"):
                    # If the model has an input_size attribute, we can infer the number of features
                    input_size = self.model.input_size
                    logger.info(f"Model input size: {input_size}")
                    # Note: We can't get feature names from architecture, but we can at least validate the size
                    logger.warning(
                        f"⚠️ Cannot extract feature names from {source} model architecture - will need to be provided externally"
                    )
            except Exception as fallback_error:
                logger.warning(
                    f"⚠️ Fallback feature extraction for {source} also failed: {str(fallback_error)}"
                )


def main():
    """
    Standalone MLP hypertuning and evaluation using load_all_data
    """
    logger.info("=" * 80)
    logger.info("🎯 STANDALONE MLP HYPERTUNING & EVALUATION")
    logger.info("=" * 80)

    try:
        # 0. Setup MLflow experiment tracking
        logger.info("0. Setting up MLflow experiment tracking...")
        mlflow_manager = MLFlowManager()
        experiment_id = mlflow_manager.setup_experiment(experiment_name)

        logger.info(f"✅ MLflow experiment setup completed: {experiment_id}")
        logger.info(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # Define prediction horizon
        prediction_horizon = 10
        number_of_trials = 200
        # n_features_to_select = 80

        # OPTION 1: Use the enhanced data preparation function with cleaning (direct import)
        data_result = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=prediction_horizon,
            split_date="2025-06-15",
            ticker=None,
            clean_features=True,
        )

        # Extract prepared data
        x_train = data_result["x_train"]
        x_test = data_result["x_test"]
        y_train = data_result["y_train"]
        y_test = data_result["y_test"]
        target_column = data_result["target_column"]
        train_date_range = data_result["train_date_range"]
        test_date_range = data_result["test_date_range"]

        logger.info(
            f"✅ Data loaded successfully y_test range: {y_test.min()} to {y_test.max()} avg: {y_test.mean()}"
        )

        # Remove outliers from target variables to improve prediction quality
        logger.info("🧹 Removing outliers from target variables...")

        # Calculate percentiles for outlier removal (keep middle 95%)
        y_train_lower = y_train.quantile(0.05)
        y_train_upper = y_train.quantile(0.95)

        # Create outlier masks
        y_train_outlier_mask = (y_train >= y_train_lower) & (y_train <= y_train_upper)

        # Apply outlier removal
        x_train_clean_outliers = x_train[y_train_outlier_mask]
        y_train_clean = y_train[y_train_outlier_mask]

        # Log outlier removal results
        logger.info("📊 Target outlier removal results:")
        logger.info(
            f"   Training: {len(y_train)} → {len(y_train_clean)} samples ({len(y_train) - len(y_train_clean)} outliers removed)"
        )
        logger.info(
            f"   Training target range: {y_train_clean.min():.6f} to {y_train_clean.max():.6f} (avg: {y_train_clean.mean():.6f})"
        )

        # Update data with cleaned versions
        x_train = x_train_clean_outliers
        y_train = y_train_clean

        # Validate and clean the data
        x_train_clean = MLPDataUtils.validate_and_clean_data(x_train)
        x_test_clean = MLPDataUtils.validate_and_clean_data(x_test)

        # Extract cleaned data
        x_train = x_train_clean
        x_test = x_test_clean

        # Create SINGLE MLPPredictorWithMLflow instance for entire pipeline
        mlp_model = MLPPredictorWithMLflow(
            model_name="mlp_complete_pipeline",
            config={"input_size": len(x_train.columns)},
        )

        # 2. Perform feature selection using the same instance
        # selected_features = mlp_model.select_features(x_train_scaled, y_train, n_features_to_select)
        numerical_features = []
        for col in x_train.columns:
            if x_train[col].dtype in [
                "float64",
                "float32",
                "int64",
                "int32",
                "float",
                "int",
            ]:
                numerical_features.append(col)

        selected_features = numerical_features
        logger.info(
            f"✅ Using {len(selected_features)} numerical features - no feature selection applied"
        )
        # Create new DataFrames with only the selected features
        x_train_selected = x_train[selected_features]
        x_test_selected = x_test[selected_features]
        logger.info(
            f"   DataFrames updated with {len(selected_features)} selected features."
        )

        # Update the model's input size for the selected features
        mlp_model.config["input_size"] = len(selected_features)

        # Fit a single StandardScaler on training data only to prevent data leakage
        x_train_selected, scaler = MLPDataUtils.scale_data(x_train_selected, None, True)
        # Remove rows with the highest 50 date_int values
        if "date_int" in x_test_selected.columns:
            threshold = x_test_selected["date_int"].copy()
            threshold = threshold.drop_duplicates().max() - 15
            logger.info(f"📅 Threshold: {threshold}")
            mask = x_test_selected["date_int"] < threshold
            x_test_selected, y_test = x_test_selected[mask], y_test[mask]
            logger.info(
                f"📅 Removed rows with date_int >= {threshold} (kept {len(x_test_selected)} samples)"
            )
        else:
            logger.warning("⚠️ 'date_int' column not found - skipping date filtering")

        x_test_scaled, _ = MLPDataUtils.scale_data(x_test_selected, scaler, False)
        # Store the fitted scaler in the SAME model instance
        mlp_model.scaler = scaler

        # Create objective function using the SAME MLP model instance with selected features
        objective_function = mlp_model.objective(
            x_train_selected,
            y_train,
            x_test_selected,
            x_test_scaled,
            y_test,
            fitted_scaler=scaler,
        )
        sampler = optuna.samplers.RandomSampler(seed=42)

        # Run optimization
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective_function, n_trials=number_of_trials, n_jobs=1)

        # Get best results from study
        best_params = study.best_params
        best_profit = (
            study.best_value
        )  # This is now threshold-optimized profit per investment

        logger.info(
            "🎯 Hyperparameter optimization with threshold optimization completed!"
        )
        logger.info(
            f"✅ Best Threshold-Optimized Profit per Investment: ${best_profit:.2f}"
        )
        logger.info(f"✅ Best parameters: {best_params}")

        # Finalize the best model (ensure mlp_model contains the best performing model)
        mlp_model.finalize_best_model()

        # Get best trial info for verification (now includes threshold info)
        best_trial_info = mlp_model.get_best_trial_info()
        logger.info(f"✅ Best trial info: {best_trial_info}")

        # Use the SAME instance throughout - no need to create final_model
        final_model = mlp_model

        # Extract current prices for evaluation
        final_current_prices = x_test_selected["close"].values

        # Evaluate with the optimal threshold from hyperparameter optimization
        optimal_threshold = getattr(final_model, "optimal_threshold", 0.5)
        confidence_method = getattr(final_model, "confidence_method", "variance")

        threshold_performance = (
            final_model.threshold_evaluator.evaluate_threshold_performance(
                model=final_model,
                x_test=x_test_selected,
                y_test=y_test,
                current_prices_test=final_current_prices,
                threshold=optimal_threshold,
                confidence_method=confidence_method,
            )
        )

        # Also get unfiltered baseline for comparison
        baseline_predictions = final_model.predict(x_test_selected)
        baseline_profit = final_model.threshold_evaluator.calculate_profit_score(
            y_test.values, baseline_predictions, final_current_prices
        )
        baseline_profit_per_investment = baseline_profit / len(y_test)

        logger.info("📊 Final Results Comparison:")
        logger.info(
            f"   Baseline (unfiltered) profit per investment: ${baseline_profit_per_investment:.2f}"
        )
        logger.info(
            f"   Threshold-optimized profit per investment: ${threshold_performance['profit_per_investment']:.2f}"
        )
        logger.info(
            f"   Improvement ratio: {threshold_performance['profit_per_investment'] / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0:.2f}x"
        )
        logger.info(
            f"   Samples kept: {threshold_performance['samples_evaluated']}/{len(x_test_selected)} ({threshold_performance['samples_kept_ratio']:.1%})"
        )
        logger.info(
            f"   Investment success rate: {threshold_performance['investment_success_rate']:.3f}"
        )

        # Use threshold-optimized metrics for final evaluation
        final_profit_per_investment = threshold_performance["profit_per_investment"]
        final_total_profit = threshold_performance["total_profit"]
        final_investment_success_rate = threshold_performance["investment_success_rate"]
        final_samples_kept = threshold_performance["samples_evaluated"]

        # Traditional metrics on filtered data
        final_mse = threshold_performance["mse"]
        final_mae = threshold_performance["mae"]
        final_r2 = threshold_performance["r2_score"]

        # Store threshold results for MLflow logging
        threshold_metrics = {
            "final_optimal_threshold": final_model.best_threshold_info[
                "optimal_threshold"
            ],
            "final_samples_kept_ratio": threshold_performance["samples_kept_ratio"],
            "final_investment_success_rate": final_investment_success_rate,
            "final_baseline_profit_per_investment": baseline_profit_per_investment,
            "final_improvement_ratio": threshold_performance["profit_per_investment"]
            / baseline_profit_per_investment
            if baseline_profit_per_investment != 0
            else 0,
        }

        logger.info("📊 Final Optimized Results:")
        logger.info(f"   Total Profit: ${final_total_profit:.2f}")
        logger.info(f"   Profit per Investment: ${final_profit_per_investment:.2f}")
        logger.info(
            f"   Samples Used: {final_samples_kept}/{len(x_test_selected)} (threshold-filtered)"
        )
        logger.info(f"   Traditional MSE: {final_mse:.4f}")
        logger.info(f"   Traditional MAE: {final_mae:.4f}")
        logger.info(f"   Traditional R²: {final_r2:.4f}")

        # Prepare comprehensive metrics for logging
        final_metrics = {
            "final_total_profit": final_total_profit,
            "final_profit_per_investment": final_profit_per_investment,
            "final_mse": final_mse,
            "final_mae": final_mae,
            "final_r2": final_r2,
            "final_test_samples": len(y_test),
            "hypertuning_best_profit": best_profit,
            "hypertuning_trials_completed": number_of_trials,
            "features_selected": len(selected_features),
        }

        # Add threshold optimization metrics if successful
        final_metrics.update(threshold_metrics)

        # Prepare comprehensive parameters for logging
        final_params = best_params.copy()
        final_params.update(
            {
                "prediction_horizon": prediction_horizon,
                "hypertuning_trials": number_of_trials,
                "target_column": target_column,
                "split_date": data_result["split_date"],
                "feature_count": data_result["feature_count"],
                "train_samples": data_result["train_samples"],
                "test_samples": data_result["test_samples"],
                "threshold_method": final_model.confidence_method,
                "threshold_optimization_during_hypertuning": True,
                "optimal_threshold_from_hypertuning": final_model.best_threshold_info[
                    "optimal_threshold"
                ],
            }
        )

        # Use the universal logging function via save_model method
        saved_run_id = final_model.save_model(
            metrics=final_metrics,
            params=final_params,
            x_eval=x_test_selected,
            experiment_name=experiment_name,
            scaler=scaler,
        )

        logger.info(
            f"✅ Model saved using updated save_model method. Run ID: {saved_run_id}"
        )

        logger.info("=" * 80)
        logger.info("🎉 STANDALONE MLP HYPERTUNING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(
            f"📊 Dataset: {data_result['train_samples'] + data_result['test_samples']:,} samples, {data_result['feature_count']} features"
        )
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📅 Train period: {train_date_range}")
        logger.info(f"📅 Test period: {test_date_range}")
        logger.info(
            f"🔧 Hypertuning: {number_of_trials} trials completed (optimizing for profit)"
        )
        logger.info(f"📈 Final Total Profit: ${final_total_profit:.2f}")
        logger.info(
            f"📈 Average Profit per Investment: ${final_profit_per_investment:.2f}"
        )
        logger.info(f"📈 Traditional MSE: {final_mse:.4f}")
        logger.info(f"💾 Model saved to MLflow run: {saved_run_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in standalone MLP hypertuning: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def smoke_test_save_and_load():
    """
    Quick smoke test: create a tiny MLP, save with scaler & feature names, then load and verify.
    """
    logger.info("= " * 40)
    logger.info("🔬 Running MLP save/load smoke test")
    logger.info("= " * 40)

    # Set up experiment quickly
    experiment = "mlp_stock_predictor_test"
    mlflow.set_experiment(experiment)

    # Create minimal data
    feature_names = ["f1", "f2", "f3", "f4"]
    import numpy as _np

    x_eval = pd.DataFrame(_np.random.RandomState(42).rand(16, 4), columns=feature_names)

    # Minimal model
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    # Fitted scaler (for artifact check)
    from sklearn.preprocessing import StandardScaler as _Std

    scaler = _Std().fit(x_eval.to_numpy())

    # Create predictor instance
    predictor = MLPPredictorWithMLflow(
        model_name="mlp_smoke_test",
        config={"input_size": 4},
    )
    predictor.model = model
    predictor.device = torch.device("cpu")

    # Save
    run_id = predictor.save_model(
        metrics={"mse": 0.0},
        params={"layers": "4-8-1"},
        x_eval=x_eval,
        experiment_name=experiment,
        scaler=scaler,
    )

    if not run_id:
        raise RuntimeError("Smoke test failed: no run_id returned from save_model")

    logger.info(f"🧪 Saved run_id: {run_id}")

    # Load into a fresh instance
    loader = MLPPredictorWithMLflow(
        model_name="mlp_smoke_test_loader",
        config={"input_size": 4},
    )
    loader.device = torch.device("cpu")
    ok = loader.load_model(run_id=run_id, experiment_name=experiment)
    if not ok:
        raise RuntimeError("Smoke test failed: load_model returned False")

    # Basic checks
    assert loader.model is not None, "Loaded model is None"
    assert hasattr(loader, "scaler") and loader.scaler is not None, (
        "Scaler was not loaded"
    )
    assert hasattr(loader, "feature_names") and loader.feature_names, (
        "Feature names were not loaded"
    )
    assert loader.feature_names == feature_names, "Feature names mismatch after load"

    logger.info(
        "✅ Smoke test passed: model, scaler, and feature names loaded correctly"
    )


if __name__ == "__main__":
    main()
