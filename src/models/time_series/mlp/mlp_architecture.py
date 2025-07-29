"""
MLP Architecture Module

This module defines the MLP neural network architecture for stock prediction.
Contains the MLPModule class with configurable layers, activations, and training features.
Also includes data creation utilities and model factory functions.
"""

import torch.nn as nn
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader, TensorDataset

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
        
        # Create layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Build network
        self._create_layers()
        
        # Create output layer based on task
        self._create_output_layer()
        
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
        
        # Process through hidden layers
        for i, layer in enumerate(self.layers):
            # Linear transformation
            x_linear = layer(x)
            
            # Batch normalization (if enabled)
            if self.batch_norm and self.batch_norms is not None:
                x_linear = self.batch_norms[i](x_linear)
            
            # Activation function
            x_activated = self._get_activation(self.activation)(x_linear)
            
            # Residual connection (if enabled and input/output sizes match)
            if self.residual and i == 0 and x.size(-1) == x_activated.size(-1):
                x_activated = x_activated + x
            
            # Dropout
            x = self.dropouts[i](x_activated)
        
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


class MLPDataUtils:
    """
    Utility class for MLP data creation and processing.
    Centralizes all data-related functionality for MLP models.
    """
    
    @staticmethod
    def create_dataloader_from_dataframe(
        X: pd.DataFrame, 
        y: pd.Series, 
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a DataLoader from DataFrame and Series.
        
        Args:
            X: Features DataFrame
            y: Targets Series
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Create DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def create_train_val_dataloaders(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation DataLoaders.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for both loaders
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_loader = MLPDataUtils.create_dataloader_from_dataframe(
            X_train, y_train, batch_size, shuffle=True
        )
        val_loader = MLPDataUtils.create_dataloader_from_dataframe(
            X_val, y_val, batch_size, shuffle=False
        )
        
        return train_loader, val_loader


class MLPModelFactory:
    """
    Factory class for creating MLP models with consistent patterns.
    Centralizes model creation logic to eliminate duplication.
    """
    
    @staticmethod
    def create_mlp_module(
        input_size: int,
        layer_sizes: List[int],
        output_size: int = 1,
        activation: str = 'relu',
        dropout: float = 0.2,
        batch_norm: bool = False,
        residual: bool = False,
        task: str = 'regression'
    ) -> MLPModule:
        """
        Create MLPModule with consistent parameter handling.
        
        Args:
            input_size: Number of input features
            layer_sizes: List of hidden layer sizes
            output_size: Number of output units
            activation: Activation function
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            task: Task type
            
        Returns:
            Configured MLPModule instance
        """
        logger.info(f"Creating MLP with architecture: input_size={input_size}, "
                   f"layer_sizes={layer_sizes}, activation={activation}, "
                   f"dropout={dropout}, batch_norm={batch_norm}, residual={residual}")
        
        return MLPModule(
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            task=task
        )
    
    @staticmethod
    def create_mlp_module_from_config(config: Dict[str, Any]) -> MLPModule:
        """
        Create MLPModule from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing MLP parameters
            
        Returns:
            Configured MLPModule instance
        """
        input_size = config.get("input_size")
        if input_size is None:
            raise ValueError("Config must include 'input_size' parameter")
        
        return MLPModelFactory.create_mlp_module(
            input_size=input_size,
            layer_sizes=config.get("layer_sizes", [128, 64, 32]),
            output_size=config.get("output_size", 1),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.2),
            batch_norm=config.get("batch_norm", False),
            residual=config.get("residual", False),
            task=config.get("task", "regression")
        ) 