"""
PyTorch Base Model Class

This module provides an abstract base class for all PyTorch-based models,
handling the common logic for training, prediction, device management,
and persistence.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from abc import abstractmethod
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional

from src.models.base_model import BaseModel
from src.models.evaluation.threshold_evaluator import ThresholdEvaluator
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


class PyTorchBasePredictor(BaseModel):
    """
    Abstract base class for PyTorch models, handling training loops,
    device management, and model persistence.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = None,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for PyTorchBasePredictor models, but torch.cuda.is_available() is False"
            )
        self.device = torch.device("cuda")
        logger.info(f"Using device: {self.device}")
        self.model = None

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create the underlying PyTorch model instance (nn.Module)."""
        pass

    def _make_loader(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None, shuffle: bool = False
    ) -> DataLoader:
        """Helper to create DataLoader from features (and optional targets)."""
        x_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32)
        dataset = (
            TensorDataset(x_tensor)
            if y is None
            else TensorDataset(x_tensor, torch.tensor(y, dtype=torch.float32))
        )
        return DataLoader(
            dataset, batch_size=self.config.get("batch_size", 32), shuffle=shuffle, num_workers=0
        )

    def _run_epoch(
        self, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer = None
    ) -> float:
        """Run one epoch (train if optimizer provided, else eval)."""
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()
        total_loss = 0

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                batch_x, batch_y = (batch[0].to(self.device), batch[1].to(self.device))
                if is_train:
                    optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                if is_train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        feature_names: List[str] = None,
        **kwargs,
    ) -> "PyTorchBasePredictor":
        """Train the PyTorch model."""
        self.model = self._create_model().to(self.device)
        self.feature_names = feature_names

        epochs = self.config.get("epochs", 20)
        loss_name = self.config.get("loss", "mse").lower()
        criterion = (
            nn.HuberLoss(delta=float(self.config.get("huber_delta", 0.1)))
            if loss_name == "huber"
            else nn.MSELoss()
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.get("learning_rate", 1e-3), weight_decay=1e-4
        )

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, criterion, optimizer)
            log_msg = f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}"

            if val_loader:
                val_loss = self._run_epoch(val_loader, criterion)
                log_msg += f", Val Loss: {val_loss:.6f}"

            logger.info(log_msg)

        self.is_trained = True
        logger.info("Training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the PyTorch model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        loader = self._make_loader(X)

        predictions = []
        with torch.no_grad():
            for (batch_x,) in loader:
                outputs = self.model(batch_x.to(self.device))
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions).squeeze()

    def _get_predictions_batch(self, X: pd.DataFrame, enable_dropout: bool = False) -> np.ndarray:
        """Internal helper for getting predictions with optional dropout."""
        original_training = self.model.training
        if enable_dropout:
            self.model.train()
        try:
            return self.predict(X)
        finally:
            if not original_training:
                self.model.eval()

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "leaf_depth") -> np.ndarray:
        """Calculate confidence scores for predictions using various methods."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating confidence")

        if method == "variance":
            predictions = np.array(
                [self._get_predictions_batch(X, enable_dropout=True) for _ in range(5)]
            )
            variance = np.var(predictions, axis=0)
            confidence_scores = 1.0 / (1.0 + variance)

        elif method == "simple":
            confidence_scores = np.abs(self.predict(X))

        elif method == "margin":
            predictions = self.predict(X)
            confidence_scores = np.abs(predictions - np.mean(predictions))

        elif method == "leaf_depth":
            predictions = self.predict(X)
            x_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32)
            x_perturbed = x_tensor + torch.randn_like(x_tensor) * 0.01

            perturbed_df = X.copy()
            perturbed_df[self.feature_names] = x_perturbed.numpy()
            perturbed_predictions = self.predict(perturbed_df)

            prediction_change = np.abs(predictions - perturbed_predictions)
            confidence_scores = 1.0 / (1.0 + prediction_change)

        else:
            raise ValueError(f"Unknown confidence method: {method}")

        # Normalize to [0, 1]
        min_conf, max_conf = confidence_scores.min(), confidence_scores.max()
        if max_conf > min_conf:
            confidence_scores = (confidence_scores - min_conf) / (max_conf - min_conf)
        else:
            logger.warning(
                f"All confidence scores are identical ({min_conf:.4f}) - using uniform distribution"
            )
            confidence_scores = np.full_like(confidence_scores, 0.5)

        logger.info(
            f"Final confidence - Range: [{confidence_scores.min():.4f}, {confidence_scores.max():.4f}]"
        )
        logger.info(
            f"Final confidence - Mean: {confidence_scores.mean():.4f}, std: {confidence_scores.std():.4f}"
        )

        return confidence_scores
