"""
PyTorch Base Model Class

This module provides an abstract base class for all PyTorch-based models,
handling the common logic for training, prediction, device management,
and persistence.
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from abc import abstractmethod
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PyTorchBasePredictor(BaseModel):
    """
    Abstract base class for PyTorch models, handling training loops, 
    device management, and model persistence.
    """

    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        # The actual model instance (nn.Module) will be created in the subclass
        self.model = None

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """
        Create the underlying PyTorch model instance (nn.Module).
        This must be implemented by subclasses.
        
        Returns:
            A PyTorch model instance.
        """
        pass

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None, **kwargs) -> 'PyTorchBasePredictor':
        """
        Train the PyTorch model.
        
        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set (optional).
            **kwargs: Additional training parameters can be passed via config.
            
        Returns:
            Self for method chaining.
        """
        self.model = self._create_model()
        self.model.to(self.device)
        self.feature_names = kwargs.get('feature_names', self.feature_names)

        epochs = self.config.get('epochs', 20)
        learning_rate = self.config.get('learning_rate', 1e-3)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

            if val_loader:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                        val_outputs = self.model(batch_X_val)
                        val_loss = criterion(val_outputs.squeeze(), batch_y_val)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

        self.is_trained = True
        logger.info("Training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the PyTorch model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        self.model.to(self.device)

        X_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.config.get('batch_size', 32), shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).squeeze()