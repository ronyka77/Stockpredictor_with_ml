"""
MLP Architecture Module

This module defines the MLP neural network architecture for stock prediction.
Contains the MLPModule class with configurable layers, activations, and training features.
Also includes data creation utilities and model factory functions.
"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPModule(nn.Module):
    """
    Configurable MLP neural network architecture for stock prediction.
    
    Supports flexible layer configurations, multiple activation functions,
    dropout regularization, batch normalization, and residual connections.
    """
    
    def __init__(self, 
                 input_size: int,
                 layer_sizes: List[int],
                 output_size: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 batch_norm: bool = False,
                 residual: bool = False,
                 task: str = 'regression'):
        """
        Initialize MLP module.
        
        Args:
            input_size: Number of input features
            layer_sizes: List of hidden layer sizes
            output_size: Number of output units (default: 1 for regression)
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
            dropout: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            task: Task type ('regression' or 'classification')
        """
        super().__init__()
        
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.task = task
        
        # Validate inputs
        if not layer_sizes:
            raise ValueError("layer_sizes must be a non-empty list")
        
        # Ensure layer_sizes is a list (handle tuple from Optuna)
        if isinstance(layer_sizes, tuple):
            layer_sizes = list(layer_sizes)
        
        # Create layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Build network
        self._create_layers()
        
        # Create output layer based on task
        self._create_output_layer()
        
        # Initialize weights properly to prevent gradient issues
        self._initialize_weights()
        
    def _get_activation(self, activation_name: str) -> nn.Module:
        """
        Get activation function by name.
        
        Args:
            activation_name: Name of activation function
            
        Returns:
            Activation function module
        """
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_name}. "
                           f"Supported: {list(activation_map.keys())}")
        
        return activation_map[activation_name]
    
    def _create_layers(self):
        """Create the hidden layers of the network."""
        layer_sizes = [self.input_size] + self.layer_sizes
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            
            # Batch normalization (if enabled)
            if self.batch_norm:
                bn_layer = nn.BatchNorm1d(layer_sizes[i + 1])
                self.batch_norms.append(bn_layer)
            
            # Dropout layer
            dropout_layer = nn.Dropout(self.dropout)
            self.dropouts.append(dropout_layer)
    
    def _create_output_layer(self):
        """Create the output layer based on task type."""
        if self.task == 'regression':
            # Linear output for regression
            self.output_layer = nn.Linear(self.layer_sizes[-1], self.output_size)
        elif self.task == 'classification':
            # Linear + sigmoid for binary classification
            self.output_layer = nn.Sequential(
                nn.Linear(self.layer_sizes[-1], self.output_size),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}. "
                           f"Supported: 'regression', 'classification'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Store original input for residual connection
        original_input = x
        
        # Process through hidden layers
        for i, layer in enumerate(self.layers):
            # Linear transformation
            x_linear = layer(x)
            
            # Batch normalization (if enabled)
            if self.batch_norm and self.batch_norms is not None:
                x_linear = self.batch_norms[i](x_linear)
            
            # Activation function - FIXED: Use the activation function directly
            if self.activation == 'relu':
                x_activated = torch.relu(x_linear)
            elif self.activation == 'leaky_relu':
                x_activated = torch.nn.functional.leaky_relu(x_linear)
            elif self.activation == 'elu':
                x_activated = torch.nn.functional.elu(x_linear)
            elif self.activation == 'gelu':
                x_activated = torch.nn.functional.gelu(x_linear)
            else:
                # Fallback to the old method
                x_activated = self._get_activation(self.activation)(x_linear)
            
            # Residual connection (if enabled and input/output sizes match)
            if self.residual and i == 0 and original_input.size(-1) == x_activated.size(-1):
                x_activated = x_activated + original_input
            
            # Dropout (only apply to hidden layers, not the last layer)
            if i < len(self.dropouts):
                x = self.dropouts[i](x_activated)
            else:
                x = x_activated
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Get information about the network architecture.
        
        Returns:
            Dictionary containing architecture information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'layer_sizes': self.layer_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'residual': self.residual,
            'task': self.task,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_layers': len(self.layers)
        }
    
    def _initialize_weights(self):
        """Initialize weights properly to prevent gradient vanishing/exploding."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # Initialize batch norm parameters
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class MLPDataUtils:
    """
    Utility class for MLP data creation and processing.
    Centralizes all data-related functionality for MLP models.
    """
    
    @staticmethod
    def validate_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Centralized data validation and cleaning without scaling.
        
        This method handles NaN/Inf values and returns cleaned data.
        Scaling is handled separately via apply_scaling method.
        
        Args:
            data: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ValueError: If data is empty or contains invalid values after cleaning
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Handle NaN values
        # Replace infinite values with NaN first
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with mean (following existing patterns from ML feature loader)
        cleaned_data = cleaned_data.fillna(0)
        
        # Check if any NaN values remain (should not happen after fillna)
        remaining_nans = cleaned_data.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"⚠️ {remaining_nans} NaN values remain after cleaning - filling with 0")
            cleaned_data = cleaned_data.fillna(0)
        
        # Check for infinite values after cleaning
        inf_count = np.isinf(cleaned_data.values).sum()
        if inf_count > 0:
            logger.warning(f"⚠️ {inf_count} infinite values found after cleaning - replacing with 0")
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0)
        
        # Final validation
        if cleaned_data.empty:
            raise ValueError("Data became empty after cleaning")
        
        # Check for any remaining invalid values
        final_nans = cleaned_data.isnull().sum().sum()
        final_infs = np.isinf(cleaned_data.values).sum()
        
        if final_nans > 0 or final_infs > 0:
            logger.error(f"❌ Data still contains invalid values after cleaning: {final_nans} NaNs, {final_infs} Infs")
            raise ValueError(f"Data validation failed: {final_nans} NaNs, {final_infs} Infs remain")
        
        logger.info(f"✅ Data validation and cleaning completed successfully: {len(cleaned_data)} rows, {len(cleaned_data.columns)} columns")
        
        return cleaned_data
    
    @staticmethod
    def scale_data(
        data: pd.DataFrame, 
        scaler: Optional[StandardScaler] = None, 
        fit_scaler: bool = False
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Apply StandardScaler to data for consistent normalization.
        
        Supports both training (fitting new scaler) and prediction (using pre-fitted scaler) scenarios.
        
        Args:
            data: Input DataFrame to scale
            scaler: Pre-fitted StandardScaler instance (for prediction)
            fit_scaler: Whether to fit a new scaler (True for training, False for prediction)
            
        Returns:
            Tuple of (scaled_data, fitted_scaler)
            
        Raises:
            ValueError: If scaler is None when fit_scaler=False
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Apply StandardScaler
        if fit_scaler:
            # Fit new scaler (for training)
            if scaler is not None:
                logger.warning("⚠️ fit_scaler=True but scaler provided - will fit new scaler")
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            logger.info("✅ Fitted new StandardScaler on training data")
        else:
            # Use existing scaler (for prediction)
            if scaler is None:
                raise ValueError("scaler must be provided when fit_scaler=False")
            
            scaled_data = scaler.transform(data)
            logger.info("✅ Applied existing StandardScaler to prediction data")
        
        # Convert back to DataFrame with original column names
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        
        return scaled_data, scaler
    
    @staticmethod
    def create_dataloader_from_dataframe(
        X: pd.DataFrame, 
        y: pd.Series, 
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader from DataFrame and Series with performance optimizations.
        
        Args:
            X: Features DataFrame
            y: Targets Series
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading (0 = main process)
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            DataLoader instance
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Create DataLoader with performance optimizations
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )
    
    @staticmethod
    def create_train_val_dataloaders(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation DataLoaders with performance optimizations.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for both loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_loader = MLPDataUtils.create_dataloader_from_dataframe(
            X_train, y_train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = MLPDataUtils.create_dataloader_from_dataframe(
            X_val, y_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        
        return train_loader, val_loader

class MLPModelWrapper(nn.Module):
    """
    Wrapper class that contains the MLP model along with all necessary components for prediction.
    This wrapper can be saved to and loaded from MLflow with all required functionality.
    """
    
    def __init__(self, 
                model: MLPModule,
                scaler: Optional[StandardScaler] = None,
                feature_names: Optional[List[str]] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the wrapper with model and prediction components.
        
        Args:
            model: Trained MLPModule instance
            scaler: Fitted StandardScaler for preprocessing
            feature_names: List of feature names (can be pandas Index)
            config: Model configuration dictionary
        """
        super().__init__()
        self.model = model
        self.scaler = scaler
        
        # Handle feature_names - convert pandas Index to list if needed
        if feature_names is not None:
            if hasattr(feature_names, 'tolist'):
                # Convert pandas Index to list
                self.feature_names = feature_names.tolist()
            elif isinstance(feature_names, (list, tuple)):
                # Already a list or tuple
                self.feature_names = list(feature_names)
            else:
                # Convert to list
                self.feature_names = list(feature_names)
        else:
            self.feature_names = []
            
        self.config = config or {}
        
        # Store model architecture info
        if model is not None:
            self.input_size = model.input_size
            self.layer_sizes = model.layer_sizes
            self.activation = model.activation
            self.dropout = model.dropout
            self.batch_norm = model.batch_norm
            self.residual = model.residual
            self.task = model.task
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        return self.model(x)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with proper preprocessing.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("No model available for prediction")
        
        self.model.eval()
        
        # Apply preprocessing
        X_scaled, _ = MLPDataUtils.scale_data(X, self.scaler, False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled.values)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions_np = predictions.cpu().numpy()
        
        # Ensure 1D output
        if predictions_np.ndim > 1:
            predictions_np = predictions_np.flatten()
        
        return predictions_np
    
    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'variance') -> np.ndarray:
        """
        Get prediction confidence scores.
        
        Args:
            X: Input features DataFrame
            method: Confidence calculation method
            
        Returns:
            Confidence scores as numpy array
        """
        predictions = self.predict(X)
        
        if method == 'variance':
            confidence = np.abs(predictions)
            if confidence.max() > 0:
                confidence = confidence / confidence.max()
            return confidence
        elif method == 'simple':
            confidence = np.abs(predictions)
            confidence = 1 / (1 + np.exp(-confidence))
            return confidence
        elif method == 'margin':
            confidence = np.abs(predictions)
            confidence = np.tanh(confidence)
            return confidence
        else:
            raise ValueError(f"Unsupported confidence method: {method}")
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the wrapped model architecture."""
        if self.model is not None:
            return self.model.get_architecture_info()
        return {}
    
    def to_device(self, device: str):
        """Move the wrapped model to specified device."""
        if self.model is not None:
            self.model = self.model.to(device)
        return self
