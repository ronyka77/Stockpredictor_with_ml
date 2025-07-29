"""
MLP Predictor Module

This module contains the MLPPredictor class with core training and prediction methods.
Handles model creation, basic training, and prediction functionality.
"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from src.models.time_series.base_pytorch_model import PyTorchBasePredictor
from src.models.evaluation import ThresholdEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPPredictor(PyTorchBasePredictor):
    """
    Predictor class for the MLP model with advanced training features.
    
    This class handles the creation, training, and prediction of the MLPModule,
    leveraging the common logic from PyTorchBasePredictor with enhanced features.
    """
    
    def __init__(self, model_name: str = "MLP", config: Optional[Dict[str, Any]] = None, 
                 threshold_evaluator: Optional[ThresholdEvaluator] = None):
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        
        # Set default MLP-specific config if not provided
        if config is None:
            config = {}
        
        # MLP-specific defaults
        self.config.setdefault('layer_sizes', [128, 64, 32])
        self.config.setdefault('activation', 'relu')
        self.config.setdefault('dropout', 0.2)
        self.config.setdefault('batch_norm', False)
        self.config.setdefault('residual', False)
        self.config.setdefault('task', 'regression')
        self.config.setdefault('learning_rate', 1e-3)
        self.config.setdefault('epochs', 50)
        self.config.setdefault('batch_size', 32)
        self.config.setdefault('optimizer', 'adam')
        self.config.setdefault('weight_decay', 0.0)
        self.config.setdefault('gradient_clip', None)
        
        # Advanced training features
        self.config.setdefault('early_stopping_patience', 10)
        self.config.setdefault('early_stopping_min_delta', 1e-4)
        self.config.setdefault('lr_scheduler', None)  # 'cosine', 'step', 'plateau'
        self.config.setdefault('lr_scheduler_params', {})
        self.config.setdefault('checkpoint_dir', './checkpoints')
        self.config.setdefault('save_best_model', True)
        self.config.setdefault('save_checkpoint_frequency', 5)  # Save every N epochs
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'epoch': []
        }

    def fit(self, train_data, val_data=None, feature_names=None, resume_from_checkpoint: str = None):
        """
        Train the MLP model with DataFrame inputs (converts to DataLoaders internally)
        
        Args:
            train_data: Either DataLoader for training data or (X_train, y_train) tuple
            val_data: Either DataLoader for validation data or (X_val, y_val) tuple (optional)
            feature_names: List of feature names (optional)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            
        Returns:
            Self for method chaining
        """
        # Store feature names
        if feature_names is None and isinstance(train_data, tuple):
            feature_names = list(train_data[0].columns)
        self.feature_names = feature_names
        
        # Convert DataFrames to DataLoaders if needed
        if isinstance(train_data, tuple) and len(train_data) == 2:
            # train_data is (X_train, y_train) tuple
            X_train, y_train = train_data
            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            
            if val_data is not None and isinstance(val_data, tuple) and len(val_data) == 2:
                # val_data is (X_val, y_val) tuple
                X_val, y_val = val_data
                val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
            else:
                val_loader = val_data  # val_data is already a DataLoader or None
        else:
            # train_data is already a DataLoader
            train_loader = train_data
            val_loader = val_data
        
        # Call the parent fit method with DataLoaders
        return super().fit(train_loader, val_loader, feature_names=feature_names, resume_from_checkpoint=resume_from_checkpoint)

    def _create_dataloader(self, X: pd.DataFrame, y: pd.Series, shuffle: bool = True):
        """
        Create a DataLoader from DataFrame and Series
        
        Args:
            X: Features DataFrame
            y: Targets Series
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
        
        batch_size = self.config.get('batch_size', 32)
        return MLPDataUtils.create_dataloader_from_dataframe(X, y, batch_size, shuffle)

    def _create_model(self) -> nn.Module:
        """
        Creates the MLPModule instance based on the model's configuration.
        """
        from src.models.time_series.mlp.mlp_architecture import MLPModelFactory
        
        return MLPModelFactory.create_mlp_module_from_config(self.config)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Convert DataFrame to tensor
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
        else:
            X_tensor = torch.FloatTensor(X)
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'variance') -> np.ndarray:
        """
        Get prediction confidence scores.
        
        Args:
            X: Input features DataFrame
            method: Confidence calculation method ('variance', 'simple', 'margin')
            
        Returns:
            Confidence scores as numpy array
        """
        if self.model is None:
            raise ValueError("Model must be trained before calculating confidence scores")
        
        self.model.eval()
        
        # Convert DataFrame to tensor
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
        else:
            X_tensor = torch.FloatTensor(X)
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions_np = predictions.cpu().numpy()
        
        if method == 'variance':
            # For MLP, we can use prediction magnitude as confidence
            # Higher absolute predictions indicate higher confidence
            confidence = np.abs(predictions_np).flatten()
            # Normalize to [0, 1] range
            if confidence.max() > 0:
                confidence = confidence / confidence.max()
            return confidence
            
        elif method == 'simple':
            # Simple confidence based on prediction magnitude
            confidence = np.abs(predictions_np).flatten()
            # Sigmoid-like normalization
            confidence = 1 / (1 + np.exp(-confidence))
            return confidence
            
        elif method == 'margin':
            # Margin-based confidence (distance from zero)
            confidence = np.abs(predictions_np).flatten()
            # Normalize using tanh for smooth [0, 1] mapping
            confidence = np.tanh(confidence)
            return confidence
            
        else:
            raise ValueError(f"Unsupported confidence method: {method}. "
                          f"Supported methods: 'variance', 'simple', 'margin'")

    def get_feature_importance(self, importance_type: str = 'gradient') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance calculation
            
        Returns:
            DataFrame with feature importance scores
        """
        # This will be implemented in the feature importance task
        if self.feature_names is None:
            return pd.DataFrame()
        
        # Placeholder implementation
        importance_scores = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history including loss and learning rate curves.
        
        Returns:
            Dictionary containing training history
        """
        return self.training_history.copy()

    def optimize_prediction_threshold(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    current_prices_test: np.ndarray,
                                    confidence_method: str = 'variance',
                                    threshold_range: Tuple[float, float] = (0.1, 0.9),
                                    n_thresholds: int = 80) -> Dict[str, Any]:
        """
        Optimize prediction threshold using ThresholdEvaluator.
        
        Args:
            X_test: Test features
            y_test: Test targets
            current_prices_test: Current prices for test data
            confidence_method: Method for confidence calculation
            threshold_range: Range of thresholds to test
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        if self.threshold_evaluator is None:
            raise ValueError("ThresholdEvaluator must be provided for threshold optimization")
        
        results = self.threshold_evaluator.optimize_prediction_threshold(
            model=self,
            X_test=X_test,
            y_test=y_test,
            current_prices_test=current_prices_test,
            confidence_method=confidence_method,
            threshold_range=threshold_range,
            n_thresholds=n_thresholds
        )
        
        # Store optimal threshold and confidence method if optimization was successful
        if results.get('status') == 'success':
            self.optimal_threshold = results.get('optimal_threshold')
            self.confidence_method = results.get('confidence_method')
        
        return results

    def predict_with_threshold(self, X: pd.DataFrame, 
                                return_confidence: bool = False,
                                threshold: Optional[float] = None,
                                confidence_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions with confidence-based filtering
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            threshold: Confidence threshold (uses optimal if None)
            confidence_method: Confidence method (uses stored if None)
            
        Returns:
            Dictionary with predictions, confidence scores, and filtering info
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Use stored optimal values if not provided
        if threshold is None:
            threshold = getattr(self, 'optimal_threshold', 0.5)
        
        if confidence_method is None:
            confidence_method = getattr(self, 'confidence_method', 'variance')
        
        # Use central evaluator for threshold-based predictions
        return self.threshold_evaluator.predict_with_threshold(
            model=self,
            X=X,
            threshold=threshold,
            confidence_method=confidence_method,
            return_confidence=return_confidence
        ) 