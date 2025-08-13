"""
LightGBM Model Implementation

This module provides a complete LightGBM model implementation for stock prediction
with hyperparameter optimization, threshold optimization, and MLflow integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from datetime import datetime
import os

from src.models.base_model import BaseModel
from src.models.evaluation import ThresholdEvaluator, CustomMetrics
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLFlowManager
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning

logger = get_logger(__name__)
experiment_name = "lightgbm_stock_predictor"
threshold_evaluator = ThresholdEvaluator()


class LightGBMModel(BaseModel):
    """
    LightGBM model for stock price prediction with advanced features
    """
    
    def __init__(self, 
                model_name: str = "lightgbm_stock_predictor",
                config: Optional[Dict[str, Any]] = None,
                prediction_horizon: int = 10,
                threshold_evaluator: Optional[ThresholdEvaluator] = None):
        """
        Initialize LightGBM model
        
        Args:
            model_name: Name for MLflow tracking
            config: Model configuration parameters
            prediction_horizon: Prediction horizon in days
            threshold_evaluator: Optional shared ThresholdEvaluator instance
        """
        # Add prediction_horizon to config
        if config is None:
            config = {}
        config['prediction_horizon'] = prediction_horizon
        
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        
        # Initialize LightGBM-specific parameters
        self.prediction_horizon = self.config.get('prediction_horizon', 10)
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
        self.eval_metric = self.config.get('eval_metric', 'rmse')
        self.base_threshold = 0.5
        self.default_confidence_method = 'leaf_depth'
        
        # Calculate CPU usage limit (75% of available cores by default)
        total_cores = os.cpu_count() or 4
        self.max_cpu_cores = max(1, int(total_cores * 0.75))
        logger.info(f"🔧 CPU Usage Limit: {self.max_cpu_cores}/{total_cores} cores (75%)")
        
        # Initialize central evaluators
        self.custom_metrics = CustomMetrics()
    
    def set_cpu_usage_limit(self, percentage: float = 0.75) -> None:
        """
        Set the CPU usage limit for LightGBM training and prediction
        
        Args:
            percentage: CPU usage percentage (0.0 to 1.0, default 0.75 for 75%)
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("CPU usage percentage must be between 0.0 and 1.0")
        
        total_cores = os.cpu_count() or 4
        self.max_cpu_cores = max(1, int(total_cores * percentage))
        logger.info(f"🔧 CPU Usage Limit Updated: {self.max_cpu_cores}/{total_cores} cores ({percentage:.1%})")
    
    def _create_model(self, params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None, **kwargs) -> 'LightGBMModel':
        """
        Create a new LightGBM model instance with specified parameters
        
        Args:
            params: Dictionary of LightGBM parameters
            model_name: Optional custom name for the new model
            **kwargs: Additional parameters
        
        Returns:
            New LightGBMModel instance configured with the provided parameters
        """
        if params is None:
            params = {}
            
        if model_name is None:
            model_name = f"{self.model_name}_configured"
        
        # Create new model instance with the same prediction horizon and evaluator settings
        config = params.copy()
        config.update({
            'investment_amount': self.threshold_evaluator.investment_amount
        })
        
        new_model = LightGBMModel(
            model_name=model_name,
            config=config,
            prediction_horizon=self.prediction_horizon,
            threshold_evaluator=self.threshold_evaluator
        )
        
        # Copy any relevant settings from the current model
        new_model.early_stopping_rounds = self.early_stopping_rounds
        new_model.eval_metric = self.eval_metric
        
        logger.info(f"Created new LightGBM model '{model_name}' with parameters: {params}")
        
        return new_model
    
    def _identify_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        Identify categorical features in the dataset
        
        Args:
            X: Feature matrix
            
        Returns:
            List of categorical feature names
        """
        categorical_features = []
        
        # Explicitly add 'ticker_id' if it exists
        if 'ticker_id' in X.columns:
            categorical_features.append('ticker_id')
            
        for col in X.columns:
            # Skip ticker_id as it's already added
            if col == 'ticker_id':
                continue

            # Check for object/string columns
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            # Check for integer columns with low cardinality (potential categorical)
            elif X[col].dtype in ['int64', 'int32'] and X[col].nunique() < 50:
                categorical_features.append(col)
        
        return categorical_features
    
    def _validate_training_data(self, X: pd.DataFrame):
        """
        Validates the training data to ensure it is not constant, which would
        lead to a model that makes identical predictions.
        """
        constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) == 1]
        
        if not constant_cols:
            return

        logger.error("🚨 Data validation failed: Constant features detected!")
        logger.error(f"   Number of constant features: {len(constant_cols)} out of {len(X.columns)}")
        logger.error(f"   Constant feature names (sample): {constant_cols[:10]}")

        # Check the ratio of constant features
        constant_ratio = len(constant_cols) / len(X.columns)
        if constant_ratio > 0.5: # If more than 50% of features are constant, this is critical.
            raise ValueError(f"CRITICAL: {constant_ratio:.1%} of features are constant. "
                            "This will lead to identical predictions. "
                            "Please check the data preparation pipeline (`prepare_ml_data_for_training_with_cleaning`). "
                            "Aborting training.")
        else:
            logger.warning("A significant number of features are constant. Model performance will likely be poor.")
    
    def _prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[lgb.Dataset, lgb.Dataset]:
        """
        Prepare data for LightGBM training using pre-split train/test data
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values
            X_test: Test feature matrix (used for validation during training)
            y_test: Test target values (used for validation during training)
            
        Returns:
            Training and test Dataset objects
        """
        # Store feature names and identify categorical features
        self.feature_names = list(X_train.columns)
        self.categorical_features = self._identify_categorical_features(X_train)
        
        # Create LightGBM datasets directly from the pre-split data
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features
        )
        
        test_data = lgb.Dataset(
            X_test, 
            label=y_test,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features,
            reference=train_data
        )
        
        return train_data, test_data
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_test: pd.DataFrame, y_test: pd.Series,
            params: Optional[Dict[str, Any]] = None) -> 'LightGBMModel':
        """
        Train the LightGBM model using pre-split train/test data
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values
            X_test: Test feature matrix (used for validation during training)
            y_test: Test target values (used for validation during training)
            params: Custom parameters (overrides defaults)
            
        Returns:
            Self for method chaining
        """
        # Validate training data before proceeding
        self._validate_training_data(X_train)
        
        # Prepare data
        train_data, test_data = self._prepare_data(X_train, y_train, X_test, y_test)
        
        # Use provided parameters or defaults
        if params is None:
            params = self.hyperparameter_config.get_default_params()
        
        # Extract n_estimators before adding to LightGBM params
        n_estimators = params.pop('n_estimators', 1000)
        
        # Add objective and evaluation metric (don't override random_state if provided)
        base_params = {
            'objective': 'regression',
            'metric': self.eval_metric,
            'verbose': -1,  # Suppress training output
            'num_threads': self.max_cpu_cores  # Limit CPU usage to 75%
        }
        # Only add random_state if not already provided in params
        if 'random_state' not in params:
            base_params['random_state'] = 42
        
        params.update(base_params)
        
        # Train model with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
            lgb.log_evaluation(period=0)  # Suppress evaluation output
        ]
        
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            num_boost_round=n_estimators,
            callbacks=callbacks
        )
        
        # Log training completion
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score['test'][self.eval_metric]
        
        logger.info(f"Training completed. Best iteration: {best_iteration}, Best score: {best_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Ensure feature names match
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Feature names don't match training data")
        
        # Make predictions
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'split') -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Args:
            importance_type: Type of importance ('split', 'gain')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        # Get importance scores
        importance_scores = self.model.feature_importance(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(importance_scores))],
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
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
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.15, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, step=0.1),
                'min_child_weight': trial.suggest_float('min_child_weight', 1, 30, step=0.1),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 500),
                'num_leaves': trial.suggest_int('num_leaves', 10, 500),
                'num_threads': self.max_cpu_cores,  # Limit CPU usage to 75%
            }
            
            try:
                # Create model with trial parameters using the _create_model method
                trial_model = self._create_model(
                    params=params,
                    model_name=f"lightgbm_trial_{trial.number}"
                )
                
                # Disable MLflow for trial models to avoid clutter
                trial_model.disable_mlflow = True
                
                trial_model.fit(X_train, y_train, X_test, y_test, params=params)
                
                # Extract current prices for test sets
                test_current_prices = X_test['close'].values
                
                # Run threshold optimization for this trial
                logger.info(f"Running threshold optimization for trial {trial.number}")
                
                threshold_results = trial_model.optimize_prediction_threshold(
                    X_test=X_test,
                    y_test=y_test,
                    current_prices_test=test_current_prices,
                    confidence_method='leaf_depth', 
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
                        self.confidence_method = 'leaf_depth'
                    
                    logger.info(f"🎯 NEW BEST TRIAL {trial.number}: Profit Per Investment = {optimized_profit_score:.3f}")
                    if threshold_info['optimal_threshold'] is not None:
                        logger.info(f"   Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                        logger.info(f"   Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                        logger.info(f"   Investment success rate: {threshold_info['investment_success_rate']:.3f}")
                        logger.info(f"   Custom accuracy: {threshold_info['custom_accuracy']:.3f}")
                    
                    self.previous_best = optimized_profit_score
                else:
                    logger.info(f"Trial {trial.number}: Profit Per Investment = {optimized_profit_score:.3f} (Best: {self.best_investment_success_rate:.3f})")
                
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
                    "confidence_method": getattr(self, 'confidence_method', 'leaf_depth')
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
                    self.confidence_method = getattr(self, 'confidence_method', 'leaf_depth')
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        
        if self.model is not None:
            info.update({
                'n_features': self.model.num_feature(),
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score,
                'categorical_features': self.categorical_features,
                'early_stopping_rounds': self.early_stopping_rounds,
                'eval_metric': self.eval_metric
            })
        
        return info
    
    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any], 
                    X_eval: pd.DataFrame, experiment_name: str = None) -> str:
        """
        Save the trained LightGBM model to MLflow using the universal logging function
        
        Args:
            metrics: Model evaluation metrics to log
            params: Model parameters to log
            X_eval: Evaluation features for signature generation
            experiment_name: Experiment name (uses default if None)
            
        Returns:
            MLflow run ID where the model was saved
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")
        
        # Use the universal logging function via the class method
        run_id = self.log_model_to_mlflow(
            metrics=metrics,
            params=params,
            X_eval=X_eval,
            experiment_name=experiment_name
        )
        
        logger.info(f"✅ LightGBM model saved using universal logging function. Run ID: {run_id}")
        return run_id
    
    def load_model(self, run_id: str) -> None:
        """
        Load a saved LightGBM model from MLflow using run ID
        
        Args:
            run_id: MLflow run ID to load model from
        """
        try:
            logger.info(f"Loading LightGBM model from MLflow run: {run_id}")
            
            # Get the model artifact path from the run
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(run_id)
            
            # List artifacts to find the model
            artifacts = client.list_artifacts(run_id)
            model_artifact_path = None
            
            # First, try to find the exact "model" directory (most common case)
            for artifact in artifacts:
                if artifact.is_dir and artifact.path == "model":
                    model_artifact_path = artifact.path
                    break
            
            # If not found, look for any directory with 'model' in the name
            if model_artifact_path is None:
                for artifact in artifacts:
                    if artifact.is_dir and 'model' in artifact.path.lower():
                        model_artifact_path = artifact.path
                        break
            
            # Default to "model" (the standard MLflow artifact path)
            if model_artifact_path is None:
                model_artifact_path = "model"
                logger.warning("⚠️ No model artifact found, defaulting to 'model' path")
            
            # Log available artifacts for debugging
            logger.info(f"Available artifacts in run {run_id}:")
            for artifact in artifacts:
                logger.info(f"  - {artifact.path} ({'dir' if artifact.is_dir else 'file'})")
            
            # Construct model URI and load
            model_uri = f"runs:/{run_id}/{model_artifact_path}"
            logger.info(f"Loading model from URI: {model_uri}")
            
            # Load the model
            loaded_model = mlflow.lightgbm.load_model(model_uri)
            self.model = loaded_model
            
            # Load additional metadata from run
            self._load_metadata_from_run(run_info)
            
            logger.info(f"✅ LightGBM model loaded successfully from MLflow run: {run_id}")
                
        except Exception as e:
            logger.error(f"❌ Error loading LightGBM model from MLflow run {run_id}: {str(e)}")
            raise
    
    def _load_metadata_from_run(self, run_info) -> None:
        """
        Load model metadata from MLflow run info
        
        Args:
            run_info: MLflow run info object
        """
        try:
            params = run_info.data.params
            
            # Restore model configuration
            self.model_name = params.get('model_model_name', self.model_name)
            self.prediction_horizon = int(params.get('model_prediction_horizon', self.prediction_horizon))
            self.early_stopping_rounds = int(params.get('model_early_stopping_rounds', self.early_stopping_rounds))
            self.eval_metric = params.get('model_eval_metric', self.eval_metric)
            
            # Extract feature names from MLflow model signature
            try:
                # Direct approach: get signature from model URI
                model_uri = f"runs:/{run_info.info.run_id}/model"
                logger.info(f"Attempting to load model signature from: {model_uri}")
                
                # Load model info to get signature
                from mlflow.models import get_model_info
                model_info = get_model_info(model_uri)
                logger.info(f"Model info loaded: {model_info is not None}")
                
                if model_info and model_info.signature:
                    logger.info(f"Model signature found: {model_info.signature is not None}")
                    
                    if model_info.signature.inputs:
                        logger.info(f"Signature inputs found: {len(model_info.signature.inputs.inputs) if hasattr(model_info.signature.inputs, 'inputs') else 'No inputs attr'}")
                        
                        # Extract feature names from signature inputs
                        feature_names = []
                        
                        # Handle different signature input formats
                        if hasattr(model_info.signature.inputs, 'inputs'):
                            # Schema format
                            for input_spec in model_info.signature.inputs.inputs:
                                if hasattr(input_spec, 'name') and input_spec.name:
                                    feature_names.append(input_spec.name)
                        elif hasattr(model_info.signature.inputs, 'schema'):
                            # Alternative schema format
                            if hasattr(model_info.signature.inputs.schema, 'input_names'):
                                feature_names = model_info.signature.inputs.schema.input_names
                        else:
                            logger.info(f"Signature inputs type: {type(model_info.signature.inputs)}")
                            logger.info(f"Signature inputs attributes: {dir(model_info.signature.inputs)}")
                        
                        if feature_names:
                            self.feature_names = feature_names
                            logger.info(f"✅ Loaded {len(self.feature_names)} feature names from MLflow signature")
                        else:
                            logger.warning("⚠️ No feature names found in model signature inputs")
                    else:
                        logger.warning("⚠️ Model signature has no inputs")
                else:
                    logger.warning("⚠️ No model signature found in model info")
                        
            except Exception as signature_error:
                logger.warning(f"⚠️ Could not extract feature names from signature: {str(signature_error)}")
                logger.info(f"Signature error details: {type(signature_error).__name__}: {signature_error}")
                
            logger.info("Model metadata loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠ Could not load all metadata: {str(e)}")
    
    @classmethod
    def load_from_mlflow(cls, run_id: str) -> 'LightGBMModel':
        """
        Class method to create a new LightGBMModel instance and load from MLflow
        
        Args:
            run_id: MLflow run ID to load model from
            
        Returns:
            New LightGBMModel instance with loaded model
        """
        # Create new instance
        lgb_model = cls()
        
        # Load model from MLflow
        lgb_model.load_model(run_id=run_id)
        
        return lgb_model
    
    def log_model_to_mlflow(self, metrics: Dict[str, float], params: Dict[str, Any], 
                            X_eval: pd.DataFrame, experiment_name: str = None) -> str:
        """
        Log the trained LightGBM model to MLflow using the universal logging function
        
        Args:
            metrics: Model evaluation metrics
            params: Model parameters
            X_eval: Evaluation features for signature generation
            experiment_name: Experiment name (uses default if None)
            
        Returns:
            MLflow run ID
        """
        if self.model is None:
            raise RuntimeError("No trained model to log")
        
        if experiment_name is None:
            experiment_name = f"{self.model_name}_experiment"
        
        # Use the universal logging function
        run_id = log_to_mlflow_lightgbm(
            model=self.model,
            metrics=metrics,
            params=params,
            experiment_name=experiment_name,
            X_eval=X_eval
        )
        
        logger.info(f"✅ Model logged to MLflow using universal function. Run ID: {run_id}")
        return run_id
    
    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'leaf_depth') -> np.ndarray:
        """
        Calculate confidence scores for predictions using various methods
        
        Args:
            X: Feature matrix
            method: Confidence calculation method ('leaf_depth', 'margin', 'variance', 'simple')
            
        Returns:
            Array of confidence scores (higher = more confident)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before calculating confidence")
        
        if method == 'leaf_depth':
            leaf_indices = self.model.predict(X, pred_leaf=True, num_iteration=self.model.best_iteration)
            confidence_scores = np.mean(leaf_indices, axis=1)
            confidence_scores = np.power(confidence_scores, 2)
            
        elif method == 'margin':
            predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
            confidence_scores = np.abs(predictions)
            
        elif method == 'variance':
            predictions = []
            n_trees = self.model.best_iteration or 100
            
            # Get predictions from different subsets of trees
            for i in range(10, min(n_trees, 100), 10):
                pred = self.model.predict(X, num_iteration=i)
                predictions.append(pred)
            
            # Calculate variance across different tree counts
            predictions_array = np.array(predictions)
            variance = np.var(predictions_array, axis=0)
            confidence_scores = 1.0 / (1.0 + variance)  # Inverse variance as confidence
            
        elif method == 'simple':
            # Simple confidence based on prediction distance from current price
            predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
            
            # Extract current prices from features (assuming 'close' column exists)
            current_prices = X['close'].values if 'close' in X.columns else np.ones_like(predictions)
            
            # Confidence = prediction change magnitude (normalized)
            relative_change = np.abs(predictions - current_prices) / current_prices
            confidence_scores = np.clip(relative_change, 0, 1)  # Clip to [0, 1]
            
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        # Normalize confidence scores to [0, 1] range
        min_conf, max_conf = confidence_scores.min(), confidence_scores.max()
        if max_conf > min_conf:
            confidence_scores = (confidence_scores - min_conf) / (max_conf - min_conf)
        else:
            # All confidence scores are identical - this is problematic
            logger.warning(f"All confidence scores are identical ({min_conf:.4f}) - using uniform distribution")
            confidence_scores = np.full_like(confidence_scores, 0.5)
        
        logger.info(f"Final confidence - Range: [{confidence_scores.min():.4f}, {confidence_scores.max():.4f}]")
        logger.info(f"Final confidence - Mean: {confidence_scores.mean():.4f}, std: {confidence_scores.std():.4f}")
        
        return confidence_scores
    
    # Removed: duplicate optimize_prediction_threshold; now inherited from BaseModel
    
    # Removed: duplicate predict_with_threshold; now inherited from BaseModel

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 50) -> List[str]:
        """
        Selects the best features using a preliminary LightGBM model based on feature importance.
        The 'close' column is always included if present in the original DataFrame.
        Args:
            X: The full feature matrix.
            y: The target series.
            n_features_to_select: The target number of features to select.
        Returns:
            A list of the selected feature names.
        """
        logger.info(f"🚀 Starting feature selection to find the best {n_features_to_select} features...")

        # Use a simple, fast configuration for the preliminary model
        prelim_params = {
            'objective': 'regression', 'metric': 'rmse', 'n_estimators': 500,
            'learning_rate': 0.05, 'verbose': -1, 'num_threads': self.max_cpu_cores, 'seed': 42
        }
        
        categorical_features = self._identify_categorical_features(X)
        train_data = lgb.Dataset(X, label=y, feature_name=list(X.columns), categorical_feature=categorical_features)
        
        prelim_model = lgb.train(params=prelim_params, train_set=train_data)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': prelim_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features from preliminary model (by gain):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  - {row['feature']}: {row['importance']:.2f}")

        # Select top N features
        selected_features = list(importance_df['feature'].head(n_features_to_select))

        # Ensure required columns are included
        required_columns = ['close', 'ticker_id']
        for column in required_columns:
            if column in X.columns and column not in selected_features:
                logger.info(f"✅ '{column}' column was not in the top features. Adding it to the list.")
                # Remove the least important feature from the list to make space
                removed_feature = selected_features.pop()
                selected_features.append(column)
                logger.info(f"   Removed '{removed_feature}' to make space for '{column}'.")

        logger.info(f"✅ Feature selection complete. Selected {len(selected_features)} features.")
        return selected_features

def log_to_mlflow_lightgbm(model, metrics, params, experiment_name, X_eval):
    """
    Log trained LightGBM model, metrics, and parameters to MLflow.
    Requires X_eval DataFrame for signature generation.
    Args:
        model: Trained LightGBM model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        X_eval (pd.DataFrame): Evaluation features for signature generation
    Returns:
        str: Run ID
    """
    try:
        # Define pip requirements for the model
        pip_requirements = [
            "lightgbm>=3.3.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "mlflow>=2.0.0"
        ]
        
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
            run_name=f"lightgbm_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            # Log parameters (excluding base params if desired)
            params_to_log = {
                k: v
                for k, v in params.items()
                if k not in ["device", "objective", "verbosity", "seed", "nthread", "verbose"]
            }
            mlflow.log_params(params_to_log)
            mlflow.log_metrics(metrics)

            # Create input example using the DataFrame X_eval
            input_example = X_eval.iloc[:5].copy()

            # Identify and convert integer columns to float64
            if hasattr(input_example, "dtypes"):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == "i":
                        logger.info(f"Converting integer column '{col}' to float64 for signature")
                        input_example[col] = input_example[col].astype("float64")

            # Infer signature - create predictions for LightGBM
            predictions_example = model.predict(input_example, num_iteration=model.best_iteration)
            signature = mlflow.models.infer_signature(input_example, predictions_example)

            # Update model registration with signature
            model_info = mlflow.lightgbm.log_model(
                model,
                "model",  # Keep artifact path as "model"
                pip_requirements=pip_requirements,
                signature=signature,
            )

            # Step 2: Register model separately using the classic API
            try:
                run_uri = f"runs:/{run.info.run_id}/model"
                registered_model_name = f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"
                mlflow.register_model(run_uri, registered_model_name)
                logger.info(f"✅ Model registered as: {registered_model_name}")
            except Exception as reg_error:
                logger.warning(f"⚠️ Model registration failed (this is optional): {reg_error}")
                logger.info("✅ Model logged successfully without registration")

            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def main():
    """
    Standalone LightGBM hypertuning and evaluation using load_all_data
    """
    logger.info("=" * 80)
    logger.info("🎯 STANDALONE LIGHTGBM HYPERTUNING & EVALUATION")
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
        number_of_trials = 20
        n_features_to_select = 80
        
        # OPTION 1: Use the enhanced data preparation function with cleaning (direct import)
        data_result = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=prediction_horizon,
            split_date='2025-02-01',
            ticker=None,  
            clean_features=True, 
        )
        
        # Extract prepared data
        X_train = data_result['X_train']
        X_test = data_result['X_test']
        y_train = data_result['y_train']
        y_test = data_result['y_test']
        target_column = data_result['target_column']
        train_date_range = data_result['train_date_range']
        test_date_range = data_result['test_date_range']
        
        lgb_model = LightGBMModel(
            model_name="lightgbm_feature_selector",
            prediction_horizon=prediction_horizon
        )

        # 2. Perform feature selection
        selected_features = lgb_model.select_features(X_train, y_train, n_features_to_select)
        
        # Create new DataFrames with only the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        logger.info(f"   DataFrames updated with {len(selected_features)} selected features.")
        

        # Remove rows with the highest 15 date_int values
        if 'date_int' in X_test_selected.columns:
            threshold = X_test_selected['date_int'].copy()
            threshold = threshold.drop_duplicates().max()-15
            logger.info(f"📅 Threshold: {threshold}")
            mask = X_test_selected['date_int'] < threshold
            X_test_selected, y_test = X_test_selected[mask], y_test[mask]
            logger.info(f"📅 Removed rows with date_int >= {threshold} (kept {len(X_test_selected)} samples)")
        else:
            logger.warning("⚠️ 'date_int' column not found - skipping date filtering")
        
        lgb_model = LightGBMModel(
            model_name="lightgbm_standalone_hypertuned",
            prediction_horizon=prediction_horizon
        )
        
        # Create objective function using the LightGBM model class method with selected features
        objective_function = lgb_model.objective(X_train_selected, y_train, X_test_selected, y_test)
        sampler = optuna.samplers.TPESampler(seed=42)
        # Run optimization with CPU limit
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective_function, n_trials=number_of_trials, n_jobs=1)
        
        # Get best results from study
        best_params = study.best_params
        best_profit = study.best_value  # This is now threshold-optimized profit per investment
        
        logger.info("🎯 Hyperparameter optimization with threshold optimization completed!")
        logger.info(f"✅ Best Threshold-Optimized Profit per Investment: ${best_profit:.2f}")
        logger.info(f"✅ Best parameters: {best_params}")
        
        # Finalize the best model (ensure lgb_model contains the best performing model)
        lgb_model.finalize_best_model()
        
        # Get best trial info for verification (now includes threshold info)
        best_trial_info = lgb_model.get_best_trial_info()
        logger.info(f"✅ Best trial info: {best_trial_info}")
        
        # lgb_model now contains the best model with optimal threshold, no need to create a new one
        final_model = lgb_model
        
        # Check if the best model has threshold optimization results
        has_threshold_optimization = (hasattr(final_model, 'best_threshold_info') and 
                                    final_model.best_threshold_info is not None and
                                    final_model.best_threshold_info.get('optimal_threshold') is not None)
        
        if has_threshold_optimization:
            # Extract current prices for evaluation
            if 'close' in X_test_selected.columns:
                final_current_prices = X_test_selected['close'].values
            else:
                final_current_prices = y_test.values * 0.95  # Fallback
            
            # Evaluate with the optimal threshold from hyperparameter optimization
            # Use stored optimal values from hyperparameter optimization
            optimal_threshold = getattr(final_model, 'optimal_threshold', 0.5)
            confidence_method = getattr(final_model, 'confidence_method', 'leaf_depth')
            
            threshold_performance = final_model.threshold_evaluator.evaluate_threshold_performance(
                model=final_model,
                X_test=X_test_selected,
                y_test=y_test,
                current_prices_test=final_current_prices,
                threshold=optimal_threshold,
                confidence_method=confidence_method
            )
            
            # Also get unfiltered baseline for comparison
            baseline_predictions = final_model.predict(X_test_selected)
            baseline_profit = final_model.threshold_evaluator.calculate_profit_score(
                y_test.values, baseline_predictions, final_current_prices
            )
            baseline_profit_per_investment = baseline_profit / len(y_test)
            
            logger.info("📊 Final Results Comparison:")
            logger.info(f"   Baseline (unfiltered) profit per investment: ${baseline_profit_per_investment:.2f}")
            logger.info(f"   Threshold-optimized profit per investment: ${threshold_performance['profit_per_investment']:.2f}")
            logger.info(f"   Improvement ratio: {threshold_performance['profit_per_investment'] / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0:.2f}x")
            logger.info(f"   Samples kept: {threshold_performance['samples_evaluated']}/{len(X_test_selected)} ({threshold_performance['samples_kept_ratio']:.1%})")
            logger.info(f"   Investment success rate: {threshold_performance['investment_success_rate']:.3f}")
            
            # Use threshold-optimized metrics for final evaluation
            final_profit_per_investment = threshold_performance['profit_per_investment']
            final_total_profit = threshold_performance['total_profit']
            final_investment_success_rate = threshold_performance['investment_success_rate']
            final_samples_kept = threshold_performance['samples_evaluated']
            
            # Traditional metrics on filtered data
            final_mse = threshold_performance['mse']
            final_mae = threshold_performance['mae']
            final_r2 = threshold_performance['r2_score']
            
            # Store threshold results for MLflow logging
            threshold_metrics = {
                'final_optimal_threshold': final_model.best_threshold_info['optimal_threshold'],
                'final_samples_kept_ratio': threshold_performance['samples_kept_ratio'],
                'final_investment_success_rate': final_investment_success_rate,
                'final_baseline_profit_per_investment': baseline_profit_per_investment,
                'final_improvement_ratio': threshold_performance['profit_per_investment'] / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0
            }
        
        logger.info("📊 Final Optimized Results:")
        logger.info(f"   Total Profit: ${final_total_profit:.2f}")
        logger.info(f"   Profit per Investment: ${final_profit_per_investment:.2f}")
        if has_threshold_optimization:
            logger.info(f"   Samples Used: {final_samples_kept}/{len(X_test_selected)} (threshold-filtered)")
        else:
            logger.info(f"   Samples Used: {final_samples_kept}/{len(X_test_selected)} (all samples)")
        logger.info(f"   Traditional MSE: {final_mse:.4f}")
        logger.info(f"   Traditional MAE: {final_mae:.4f}")
        logger.info(f"   Traditional R²: {final_r2:.4f}")
        
        feature_importance = final_model.get_feature_importance('gain')
        
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
            "features_selected": len(selected_features)
        }
        
        # Add threshold optimization metrics if successful
        final_metrics.update(threshold_metrics)
        
        # Add top feature importance to metrics
        top_features = feature_importance.head(10)
        for i, (_, row) in enumerate(top_features.iterrows()):
            final_metrics[f"feature_importance_{i+1}_{row['feature']}"] = row['importance']
        
        # Prepare comprehensive parameters for logging
        final_params = best_params.copy()
        final_params.update({
            "prediction_horizon": prediction_horizon,
            "early_stopping_rounds": final_model.early_stopping_rounds,
            "eval_metric": final_model.eval_metric,
            "hypertuning_trials": number_of_trials,
            "hypertuning_direction": "maximize",
            "hypertuning_metric": "profit_score",
            "target_column": target_column,
            "split_date": data_result['split_date'],
            "feature_count": data_result['feature_count'],
            "train_samples": data_result['train_samples'],
            "test_samples": data_result['test_samples'],
            "threshold_optimization_enabled": has_threshold_optimization
        })
        
        # Add threshold parameters if optimization was successful during hyperparameter optimization
        if has_threshold_optimization:
            final_params.update({
                "threshold_method": final_model.confidence_method,
                "threshold_optimization_during_hypertuning": True,
                "optimal_threshold_from_hypertuning": final_model.best_threshold_info['optimal_threshold']
            })
        
        # Use the universal logging function via save_model method
        saved_run_id = final_model.save_model(
            metrics=final_metrics,
            params=final_params,
            X_eval=X_test_selected,
            experiment_name=experiment_name
        )
        
        logger.info(f"✅ Model saved using updated save_model method. Run ID: {saved_run_id}")
        logger.info("✅ Model registered with timestamp-based name via universal function")
        
        logger.info("=" * 80)
        logger.info("🎉 STANDALONE LIGHTGBM HYPERTUNING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset: {data_result['train_samples'] + data_result['test_samples']:,} samples, {data_result['feature_count']} features")
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📅 Train period: {train_date_range}")
        logger.info(f"📅 Test period: {test_date_range}")
        logger.info(f"🔧 Hypertuning: {number_of_trials} trials completed (optimizing for profit)")
        logger.info(f"📈 Final Total Profit: ${final_total_profit:.2f}")
        logger.info(f"📈 Average Profit per Investment: ${final_profit_per_investment:.2f}")
        logger.info(f"📈 Traditional MSE: {final_mse:.4f}")
        logger.info(f"💾 Model saved to MLflow run: {saved_run_id}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in standalone LightGBM hypertuning: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 