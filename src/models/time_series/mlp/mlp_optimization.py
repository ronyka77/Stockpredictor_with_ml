"""
MLP Optimization Module

This module contains hyperparameter optimization methods for the MLP model.
Includes Optuna integration, objective functions, and optimization utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import torch
from sklearn.preprocessing import StandardScaler

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPOptimizationMixin:
    """
    Mixin class providing hyperparameter optimization functionality for MLPPredictor.
    """

    def objective(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        x_test_scaled: Optional[pd.DataFrame],
        y_test: pd.Series,
        fitted_scaler=StandardScaler,
    ) -> callable:
        """
        Create Optuna objective function for hyperparameter optimization with threshold optimization

        Args:
            x_train: Training features
            y_train: Training targets
            x_test: Test features
            y_test: Test targets
            fitted_scaler: Pre-fitted StandardScaler instance (optional)

        Returns:
            Objective function for Optuna optimization with threshold optimization
        """
        # Initialize tracking variables for best model (optimizing for investment success rate)
        self.best_investment_success_rate = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None
        self.feature_names = x_train.columns

        # Store the fitted scaler for use in trials
        if fitted_scaler is not None:
            self.fitted_scaler = fitted_scaler
            logger.info("✅ Pre-fitted scaler received for optimization trials")

        def objective(trial):
            """Objective function for Optuna optimization with threshold optimization for each trial"""

            # Get layer_sizes as string and convert to list
            layer_sizes_str = trial.suggest_categorical(
                "layer_sizes",
                (
                    "64,32",
                    "128,64",
                    "256,128,64",
                    "512,256,128,64",
                    "128,64,32",
                    "256,128,64,32",
                ),
            )

            # Convert string back to list of integers
            layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(",")]

            # Model creation parameters (relevant for _create_model)
            model_params = {
                "input_size": len(x_train.columns),  # Add input_size from training data
                "layer_sizes": layer_sizes,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 5e-2, log=True
                ),
                "dropout": trial.suggest_float("dropout", 0.05, 0.5),
                "optimizer": "adamw",  # Fixed - generally the best choice
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-2, log=True
                ),
                "activation": "relu",  # Fixed - most reliable activation
                "epochs": trial.suggest_int("epochs", 5, 30),
                "batch_norm": True,  # Fixed - generally beneficial
                "residual": False,  # Fixed - not needed for MLPs
                "gradient_clip": trial.suggest_float(
                    "gradient_clip", 0.1, 5.0, log=True
                ),  # Keep variable
                "early_stopping_patience": 15,  # Fixed - good default
                "lr_scheduler": "cosine",  # Fixed - often the best scheduler
            }

            # Data loading parameters (not relevant for model creation)
            data_params = {
                "batch_size": trial.suggest_categorical(
                    "batch_size", (256, 512, 1024, 2048)
                ),
                "num_workers": 4,  # Fixed based on CPU cores
                # Only pin memory if CUDA is available to avoid warnings on CPU
                "pin_memory": torch.cuda.is_available(),
            }

            try:
                # Create a new model instance for this trial
                trial_model = MLPPredictor(
                    model_name=f"mlp_trial_{trial.number}",
                    config=model_params,
                    threshold_evaluator=self.threshold_evaluator,
                )

                # Disable MLflow for trial models to avoid clutter
                trial_model.disable_mlflow = True

                # Create the model using _create_model with only model-relevant parameters
                trial_model.model = trial_model._create_model(params=model_params)
                trial_model.model.to(trial_model.device)

                # Store the fitted scaler for later use in the trial model
                self.current_trial_scaler = (
                    getattr(self, "fitted_scaler", None) or fitted_scaler
                )

                # Create DataLoaders using the cleaned and scaled data
                num_workers = data_params.get("num_workers", 4)
                pin_memory = data_params.get("pin_memory", True)

                logger.info(
                    f"🚀 Creating DataLoaders with num_workers={num_workers}, pin_memory={pin_memory}"
                )

                # Create DataLoaders using the cleaned and scaled data
                train_loader = MLPDataUtils.create_dataloader_from_dataframe(
                    x_train,
                    y_train,
                    data_params["batch_size"],
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                val_loader = MLPDataUtils.create_dataloader_from_dataframe(
                    x_test_scaled,
                    y_test,
                    data_params["batch_size"],
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )

                # Train the model
                trial_model.fit(
                    train_loader,
                    val_loader,
                    scaler=self.fitted_scaler,
                    feature_names=self.feature_names,
                )

                # Extract current prices for test sets
                test_current_prices = (
                    x_test["close"].values
                    if "close" in x_test.columns
                    else np.ones(len(x_test))
                )

                # Run threshold optimization for this trial
                logger.info(f"Running threshold optimization for trial {trial.number}")

                threshold_results = trial_model.optimize_prediction_threshold(
                    x_test=x_test,
                    y_test=y_test,
                    current_prices_test=test_current_prices,
                    confidence_method="variance",
                )

                # Use threshold-optimized investment success rate
                optimized_profit_score = threshold_results["best_result"][
                    "test_profit_per_investment"
                ]

                # Store additional threshold info for logging
                threshold_info = {
                    "optimal_threshold": threshold_results["optimal_threshold"],
                    "samples_kept_ratio": threshold_results["best_result"][
                        "test_samples_ratio"
                    ],
                    "investment_success_rate": threshold_results["best_result"][
                        "investment_success_rate"
                    ],
                    "test_profit_per_investment": threshold_results["best_result"][
                        "test_profit_per_investment"
                    ],
                    "custom_accuracy": threshold_results["best_result"][
                        "test_custom_accuracy"
                    ],
                    "total_threshold_profit": threshold_results["best_result"][
                        "test_profit"
                    ],
                    "profitable_investments": threshold_results["best_result"][
                        "profitable_investments"
                    ],
                }

                # Log threshold optimization results for this trial
                logger.info(f"Trial {trial.number} threshold optimization:")
                logger.info(
                    f"  Optimal threshold: {threshold_info['optimal_threshold']:.3f}"
                )
                logger.info(
                    f"  Samples kept: {threshold_info['samples_kept_ratio']:.1%}"
                )
                logger.info(
                    f"  Optimized profit per investment: {optimized_profit_score:.3f}"
                )

                # Check if this is the best trial so far
                if optimized_profit_score > self.best_investment_success_rate:
                    self.best_investment_success_rate = optimized_profit_score
                    self.best_trial_model = trial_model
                    # Combine model and data parameters for storing best trial info
                    combined_params = {**model_params, **data_params}
                    self.best_trial_params = combined_params.copy()
                    self.best_threshold_info = threshold_info.copy()

                    # Update self.model with the best trial model
                    self.model = trial_model.model
                    self.feature_names = trial_model.feature_names

                    # Store the optimal threshold information
                    if threshold_info["optimal_threshold"] is not None:
                        self.optimal_threshold = threshold_info["optimal_threshold"]
                        self.confidence_method = "variance"

                    logger.info(
                        f"🎯 NEW BEST TRIAL {trial.number}: Profit Per Investment = {optimized_profit_score:.3f}"
                    )
                    if threshold_info.get("optimal_threshold") is not None:
                        logger.info(
                            f"   Optimal threshold: {threshold_info.get('optimal_threshold'):.3f}"
                        )
                        logger.info(
                            f"   Samples kept: {threshold_info.get('samples_kept_ratio', 0):.1%}"
                        )
                        logger.info(
                            f"   Investment success rate: {threshold_info.get('investment_success_rate', 0):.3f}"
                        )
                        logger.info(
                            f"   Custom accuracy: {threshold_info.get('custom_accuracy', 0):.3f}"
                        )

                    self.previous_best = optimized_profit_score
                else:
                    logger.info(
                        f"Trial {trial.number}: Profit Per Investment = {optimized_profit_score:.3f} (Best: {self.best_investment_success_rate:.3f})"
                    )

                return optimized_profit_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6

        return objective

    def get_best_trial_info(self) -> Dict[str, Any]:
        """
        Get information about the best trial from hyperparameter optimization with threshold info

        Returns:
            Dictionary with best trial information including threshold optimization details
        """
        if not hasattr(self, "best_investment_success_rate"):
            return {"message": "No hyperparameter optimization has been run yet"}

        base_info = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "best_trial_params": self.best_trial_params,
            "has_best_model": self.best_trial_model is not None,
            # Use getattr to avoid AttributeError when `model` wasn't set on the mixin
            "model_updated": getattr(self, "model", None) is not None,
        }

        # Add scaler information if available
        if hasattr(self, "scaler") and self.scaler is not None:
            base_info.update(
                {
                    "scaler_info": {
                        "scaler_type": "StandardScaler",
                        "scaler_available": True,
                        "scaler_fitted": hasattr(self.scaler, "mean_")
                        and hasattr(self.scaler, "scale_"),
                    }
                }
            )
        else:
            base_info.update(
                {
                    "scaler_info": {
                        "scaler_type": "None",
                        "scaler_available": False,
                        "scaler_fitted": False,
                    }
                }
            )

        # Add threshold optimization information if available
        if (
            hasattr(self, "best_threshold_info")
            and self.best_threshold_info is not None
        ):
            base_info.update(
                {
                    "threshold_optimization": {
                        "optimal_threshold": self.best_threshold_info.get(
                            "optimal_threshold"
                        ),
                        "samples_kept_ratio": self.best_threshold_info.get(
                            "samples_kept_ratio"
                        ),
                        "investment_success_rate": self.best_threshold_info.get(
                            "investment_success_rate"
                        ),
                        "custom_accuracy": self.best_threshold_info.get(
                            "custom_accuracy"
                        ),
                        "total_threshold_profit": self.best_threshold_info.get(
                            "total_threshold_profit"
                        ),
                        "profitable_investments": self.best_threshold_info.get(
                            "profitable_investments"
                        ),
                        "confidence_method": getattr(
                            self, "confidence_method", "variance"
                        ),
                    }
                }
            )
        else:
            base_info["threshold_optimization"] = None

        return base_info

    def finalize_best_model(self) -> None:
        """
        Finalize the best model after hyperparameter optimization with threshold optimization
        This ensures the main model instance contains the best performing model and threshold info
        """
        if hasattr(self, "best_trial_model") and self.best_trial_model is not None:
            # Copy the best model's state to this instance
            self.model = self.best_trial_model.model
            self.feature_names = self.best_trial_model.feature_names

            # Transfer the scaler from the best trial model
            if (
                hasattr(self.best_trial_model, "scaler")
                and self.best_trial_model.scaler is not None
            ):
                self.scaler = self.best_trial_model.scaler
                logger.info("✅ StandardScaler transferred from best trial model")
            else:
                logger.warning("⚠️ No scaler found in best trial model")

            # Copy threshold optimization information if available
            if (
                hasattr(self, "best_threshold_info")
                and self.best_threshold_info is not None
            ):
                if self.best_threshold_info.get("optimal_threshold") is not None:
                    self.optimal_threshold = self.best_threshold_info[
                        "optimal_threshold"
                    ]
                    self.confidence_method = getattr(
                        self, "confidence_method", "variance"
                    )

            # Log the finalization
            logger.info(
                f"✅ Best model finalized with investment success rate: {self.best_investment_success_rate:.3f}"
            )
            logger.info(f"✅ Best parameters: {self.best_trial_params}")

            # Log threshold information if available
            if (
                hasattr(self, "best_threshold_info")
                and self.best_threshold_info is not None
            ):
                threshold_info = self.best_threshold_info
                if threshold_info.get("optimal_threshold") is not None:
                    logger.info(
                        f"✅ Optimal threshold: {threshold_info['optimal_threshold']:.3f}"
                    )
                    logger.info(
                        f"✅ Samples kept ratio: {threshold_info.get('samples_kept_ratio', 0):.1%}"
                    )
                    logger.info(
                        f"✅ Investment success rate: {threshold_info.get('investment_success_rate', 0):.3f}"
                    )
                    logger.info(
                        f"✅ Custom accuracy: {threshold_info.get('custom_accuracy', 0):.3f}"
                    )
                    logger.info(
                        f"✅ Profitable investments: {threshold_info.get('profitable_investments', 0)}"
                    )
                else:
                    logger.info(
                        "✅ No threshold optimization was successful for the best trial"
                    )

            # Log to MLflow if enabled
            if not getattr(self, "disable_mlflow", False):
                # Log best hyperparameters
                self.log_params(
                    {f"best_{k}": v for k, v in self.best_trial_params.items()}
                )

                # Log best metrics
                metrics_to_log = {
                    "best_investment_success_rate": self.best_investment_success_rate,
                    "hypertuning_completed": 1,
                }

                # Add scaler information
                if hasattr(self, "scaler") and self.scaler is not None:
                    metrics_to_log.update(
                        {"scaler_used": 1, "scaler_type": "StandardScaler"}
                    )
                    logger.info("✅ StandardScaler information logged to MLflow")
                else:
                    metrics_to_log.update({"scaler_used": 0, "scaler_type": "None"})
                    logger.warning("⚠️ No scaler information to log to MLflow")

                # Add threshold metrics if available
                if (
                    hasattr(self, "best_threshold_info")
                    and self.best_threshold_info is not None
                ):
                    threshold_info = self.best_threshold_info
                    if threshold_info.get("optimal_threshold") is not None:
                        metrics_to_log.update(
                            {
                                "best_optimal_threshold": threshold_info[
                                    "optimal_threshold"
                                ],
                                "best_samples_kept_ratio": threshold_info[
                                    "samples_kept_ratio"
                                ],
                                "best_investment_success_rate": threshold_info[
                                    "investment_success_rate"
                                ],
                                "best_custom_accuracy": threshold_info[
                                    "custom_accuracy"
                                ],
                                "best_total_threshold_profit": threshold_info[
                                    "total_threshold_profit"
                                ],
                                "best_profitable_investments": threshold_info[
                                    "profitable_investments"
                                ],
                            }
                        )

                self.log_metrics(metrics_to_log)
        else:
            logger.warning("⚠ No best model found to finalize")


# Extend MLPPredictor with optimization mixin
class MLPPredictorWithOptimization(MLPPredictor, MLPOptimizationMixin):
    """
    MLPPredictor with hyperparameter optimization capabilities.
    """

    pass
