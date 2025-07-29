"""
MLP Optimization Module

This module contains hyperparameter optimization methods for the MLP model.
Includes Optuna integration, objective functions, and optimization utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPOptimizationMixin:
    """
    Mixin class providing hyperparameter optimization functionality for MLPPredictor.
    """
    
    def _create_model_for_tuning(self, params: Optional[Dict[str, Any]] = None, 
                                model_name: Optional[str] = None) -> 'MLPPredictor':
        """
        Create a new MLPPredictor instance with specified parameters for hyperparameter tuning
        
        Args:
            params: Dictionary of MLP parameters
            model_name: Optional custom name for the new model
            
        Returns:
            New MLPPredictor instance configured with the provided parameters
        """
        if params is None:
            params = {}
            
        if model_name is None:
            model_name = f"{self.model_name}_trial"
        
        # Create new model instance with the same threshold evaluator
        config = self.config.copy()
        config.update(params)
        
        new_model = MLPPredictor(
            model_name=model_name,
            config=config,
            threshold_evaluator=self.threshold_evaluator
        )
        
        return new_model

    def objective(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> callable:
        """
        Create Optuna objective function for hyperparameter optimization with threshold optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features  
            y_test: Test targets
            
        Returns:
            Objective function for Optuna optimization with threshold optimization
        """
        # Initialize tracking variables for best model (optimizing for investment success rate)
        self.best_investment_success_rate = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None
        
        def objective(trial):
            """Objective function for Optuna optimization with threshold optimization for each trial"""
            params = {
                'layer_sizes': trial.suggest_categorical('layer_sizes', 
                    [[64, 32], [128, 64], [256, 128, 64], [512, 256, 128, 64], [128, 64, 32], [256, 128, 64, 32]]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop', 'sgd']),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'gelu']),
                'epochs': trial.suggest_int('epochs', 20, 100),
                'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
                'residual': trial.suggest_categorical('residual', [True, False]),
                'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 5.0, log=True),
                'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 15),
                'lr_scheduler': trial.suggest_categorical('lr_scheduler', [None, 'cosine', 'step', 'plateau'])
            }
            
            try:
                # Create model with trial parameters using the _create_model_for_tuning method
                trial_model = self._create_model_for_tuning(
                    params=params,
                    model_name=f"mlp_trial_{trial.number}"
                )
                
                # Disable MLflow for trial models to avoid clutter
                trial_model.disable_mlflow = True
                
                # Prepare data for training
                train_loader, val_loader = self._prepare_data_for_training(X_train, y_train, X_test, y_test, params['batch_size'])
                
                # Train the model
                trial_model.fit(train_loader, val_loader)
                
                # Extract current prices for test sets
                test_current_prices = X_test['close'].values if 'close' in X_test.columns else np.ones(len(X_test))
                
                # Run threshold optimization for this trial
                logger.debug(f"Running threshold optimization for trial {trial.number}")
                
                threshold_results = trial_model.optimize_prediction_threshold(
                    X_test=X_test,
                    y_test=y_test,
                    current_prices_test=test_current_prices,
                    confidence_method='variance', 
                    threshold_range=(0.1, 0.9),
                    n_thresholds=80  
                )
                
                # Use threshold-optimized investment success rate
                optimized_profit_score = threshold_results['best_result']['test_profit_per_investment']
                
                # Store additional threshold info for logging
                threshold_info = {
                    'optimal_threshold': threshold_results['optimal_threshold'],
                    'samples_kept_ratio': threshold_results['best_result']['test_samples_ratio'],
                    'investment_success_rate': threshold_results['best_result']['investment_success_rate'],
                    'test_profit_per_investment': threshold_results['best_result']['test_profit_per_investment'],
                    'custom_accuracy': threshold_results['best_result']['test_custom_accuracy'],
                    'total_threshold_profit': threshold_results['best_result']['test_profit'],
                    'profitable_investments': threshold_results['best_result']['profitable_investments']
                }
                
                # Log threshold optimization results for this trial
                logger.debug(f"Trial {trial.number} threshold optimization:")
                logger.debug(f"  Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                logger.debug(f"  Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                logger.debug(f"  Optimized profit per investment: {optimized_profit_score:.3f}")
                
                # Check if this is the best trial so far
                if optimized_profit_score > self.best_investment_success_rate:
                    self.best_investment_success_rate = optimized_profit_score
                    self.best_trial_model = trial_model
                    self.best_trial_params = params.copy()
                    self.best_threshold_info = threshold_info.copy()
                    
                    # Update self.model with the best trial model
                    self.model = trial_model.model
                    self.feature_names = trial_model.feature_names
                    
                    # Store the optimal threshold information
                    if threshold_info['optimal_threshold'] is not None:
                        self.optimal_threshold = threshold_info['optimal_threshold']
                        self.confidence_method = 'variance'
                    
                    logger.info(f"🎯 NEW BEST TRIAL {trial.number}: Profit Per Investment = {optimized_profit_score:.3f}")
                    if threshold_info['optimal_threshold'] is not None:
                        logger.info(f"   Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                        logger.info(f"   Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                        logger.info(f"   Investment success rate: {threshold_info['investment_success_rate']:.3f}")
                        logger.info(f"   Custom accuracy: {threshold_info['custom_accuracy']:.3f}")
                    
                    self.previous_best = optimized_profit_score
                else:
                    logger.debug(f"Trial {trial.number}: Profit Per Investment = {optimized_profit_score:.3f} (Best: {self.best_investment_success_rate:.3f})")
                
                return optimized_profit_score
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6
        
        return objective

    def _prepare_data_for_training(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                    X_test: pd.DataFrame, y_test: pd.Series, 
                                    batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training by creating DataLoaders
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
        
        return MLPDataUtils.create_train_val_dataloaders(
            X_train, y_train, X_test, y_test, batch_size
        )

    def get_best_trial_info(self) -> Dict[str, Any]:
        """
        Get information about the best trial from hyperparameter optimization with threshold info
        
        Returns:
            Dictionary with best trial information including threshold optimization details
        """
        if not hasattr(self, 'best_investment_success_rate'):
            return {"message": "No hyperparameter optimization has been run yet"}
        
        base_info = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "best_trial_params": self.best_trial_params,
            "has_best_model": self.best_trial_model is not None,
            "model_updated": self.model is not None
        }
        
        # Add threshold optimization information if available
        if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
            base_info.update({
                "threshold_optimization": {
                    "optimal_threshold": self.best_threshold_info.get('optimal_threshold'),
                    "samples_kept_ratio": self.best_threshold_info.get('samples_kept_ratio'),
                    "investment_success_rate": self.best_threshold_info.get('investment_success_rate'),
                    "custom_accuracy": self.best_threshold_info.get('custom_accuracy'),
                    "total_threshold_profit": self.best_threshold_info.get('total_threshold_profit'),
                    "profitable_investments": self.best_threshold_info.get('profitable_investments'),
                    "confidence_method": getattr(self, 'confidence_method', 'variance')
                }
            })
        else:
            base_info["threshold_optimization"] = None
        
        return base_info

    def finalize_best_model(self) -> None:
        """
        Finalize the best model after hyperparameter optimization with threshold optimization
        This ensures the main model instance contains the best performing model and threshold info
        """
        if hasattr(self, 'best_trial_model') and self.best_trial_model is not None:
            # Copy the best model's state to this instance
            self.model = self.best_trial_model.model
            self.feature_names = self.best_trial_model.feature_names
            
            # Copy threshold optimization information if available
            if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                if self.best_threshold_info.get('optimal_threshold') is not None:
                    self.optimal_threshold = self.best_threshold_info['optimal_threshold']
                    self.confidence_method = getattr(self, 'confidence_method', 'variance')
            
            # Log the finalization
            logger.info(f"✅ Best model finalized with investment success rate: {self.best_investment_success_rate:.3f}")
            logger.info(f"✅ Best parameters: {self.best_trial_params}")
            
            # Log threshold information if available
            if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                threshold_info = self.best_threshold_info
                if threshold_info.get('optimal_threshold') is not None:
                    logger.info(f"✅ Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                    logger.info(f"✅ Samples kept ratio: {threshold_info['samples_kept_ratio']:.1%}")
                    logger.info(f"✅ Investment success rate: {threshold_info['investment_success_rate']:.3f}")
                    logger.info(f"✅ Custom accuracy: {threshold_info['custom_accuracy']:.3f}")
                    logger.info(f"✅ Profitable investments: {threshold_info['profitable_investments']}")
                else:
                    logger.info("✅ No threshold optimization was successful for the best trial")
            
            # Log to MLflow if enabled
            if not getattr(self, 'disable_mlflow', False):
                # Log best hyperparameters
                self.log_params({f"best_{k}": v for k, v in self.best_trial_params.items()})
                
                # Log best metrics
                metrics_to_log = {
                    "best_investment_success_rate": self.best_investment_success_rate,
                    "hypertuning_completed": 1
                }
                
                # Add threshold metrics if available
                if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                    threshold_info = self.best_threshold_info
                    if threshold_info.get('optimal_threshold') is not None:
                        metrics_to_log.update({
                            "best_optimal_threshold": threshold_info['optimal_threshold'],
                            "best_samples_kept_ratio": threshold_info['samples_kept_ratio'],
                            "best_investment_success_rate": threshold_info['investment_success_rate'],
                            "best_custom_accuracy": threshold_info['custom_accuracy'],
                            "best_total_threshold_profit": threshold_info['total_threshold_profit'],
                            "best_profitable_investments": threshold_info['profitable_investments']
                        })
                
                self.log_metrics(metrics_to_log)
        else:
            logger.warning("⚠ No best model found to finalize")


# Extend MLPPredictor with optimization mixin
class MLPPredictorWithOptimization(MLPPredictor, MLPOptimizationMixin):
    """
    MLPPredictor with hyperparameter optimization capabilities.
    """
    pass 