"""
Base Model Class

This module provides an abstract base class for all machine learning models
in the stock prediction system, with built-in MLflow tracking, persistence,
and evaluation capabilities.
"""

import os
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
import mlflow.sklearn

from src.utils.logger import get_logger
from src.utils.mlflow_integration import MLflowIntegration

logger = get_logger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all ML models with MLflow integration
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model for tracking
            config: Model configuration parameters
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
        self.disable_mlflow = False  # Flag to disable MLflow logging
        
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
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
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
        
        if not hasattr(self.model, 'predict_proba'):
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
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                 metrics_calculator: Any) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features for evaluation
            y: True targets
            metrics_calculator: CustomMetrics instance
            
        Returns:
            Dictionary of metric scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics = {}
        
        # Custom accuracy metric for financial data
        if hasattr(metrics_calculator, 'custom_accuracy'):
            metrics['custom_accuracy'] = metrics_calculator.custom_accuracy(y, predictions)
        
        # Directional accuracy
        if hasattr(metrics_calculator, 'directional_accuracy'):
            metrics['directional_accuracy'] = metrics_calculator.directional_accuracy(y, predictions)
        
        # Traditional ML metrics
        if hasattr(metrics_calculator, 'calculate_regression_metrics'):
            reg_metrics = metrics_calculator.calculate_regression_metrics(y, predictions)
            metrics.update(reg_metrics)
        
        return metrics
    
    def start_mlflow_run(self, experiment_name: Optional[str] = None) -> None:
        """
        Start MLflow tracking run
        
        Args:
            experiment_name: Name of the experiment (optional)
        """
        if self.disable_mlflow:
            return
        
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
        if self.disable_mlflow:
            return
        
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
        if self.disable_mlflow:
            return
        
        try:
            self.mlflow_integration.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"MLflow metric logging failed: {e}")
    
    def log_model(self, flavor: str = "sklearn") -> None:
        """
        Log model to MLflow
        
        Args:
            flavor: MLflow model flavor (sklearn, xgboost, catboost, etc.)
        """
        try:
            if self.is_trained:
                self.mlflow_integration.log_model(
                    self.model,
                    artifact_path=self.model_name,
                    flavor=flavor,
                    registered_model_name=f"{self.model_name}_stock_prediction"
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
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'config': self.config,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'run_id': self.run_id,
            'training_history': self.training_history
        } 