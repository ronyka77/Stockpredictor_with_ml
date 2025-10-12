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
from src.utils.logger import get_logger

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        feature_names: List[str] = None,
        **kwargs,
    ) -> "PyTorchBasePredictor":
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
        self.feature_names = feature_names

        epochs = self.config.get("epochs", 20)
        learning_rate = self.config.get("learning_rate", 1e-3)

        # Loss toggle (MSE or Huber)
        loss_name = self.config.get("loss", "mse").lower()
        if loss_name == "huber":
            delta = float(self.config.get("huber_delta", 0.1))
            criterion = nn.HuberLoss(delta=delta)
        else:
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

            if val_loader:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val, batch_y_val = (
                            batch_X_val.to(self.device),
                            batch_y_val.to(self.device),
                        )
                        val_outputs = self.model(batch_X_val)
                        val_loss = criterion(val_outputs.squeeze(), batch_y_val)
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}"
                )

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
        loader = DataLoader(
            dataset, batch_size=self.config.get("batch_size", 32), shuffle=False, num_workers=0
        )

        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions).squeeze()

    def get_prediction_confidence(
        self, X: pd.DataFrame, method: str = "leaf_depth"
    ) -> np.ndarray:
        """
        Calculate confidence scores for predictions using various methods

        Args:
            X: Feature matrix
            method: Confidence calculation method ('variance', 'simple', 'margin')

        Returns:
            Array of confidence scores (higher = more confident)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating confidence")

        if method == "variance":
            # Use prediction variance across multiple forward passes with dropout enabled
            # Enable dropout temporarily for uncertainty quantification (not training)
            original_training = self.model.training
            self.model.train()  # Enable dropout for variance calculation

            try:
                predictions = []
                n_passes = 5  # Reduced from 10 for efficiency

                X_tensor = torch.tensor(
                    X[self.feature_names].values, dtype=torch.float32
                )
                dataset = TensorDataset(X_tensor)
                loader = DataLoader(
                    dataset, batch_size=self.config.get("batch_size", 32), shuffle=False, num_workers=0
                )

                for _ in range(n_passes):
                    pass_predictions = []
                    with torch.no_grad():
                        for (batch_X,) in loader:
                            batch_X = batch_X.to(self.device)
                            outputs = self.model(batch_X)
                            pass_predictions.append(outputs.cpu().numpy())
                    predictions.append(np.concatenate(pass_predictions).squeeze())

                # Calculate variance across passes
                predictions_array = np.array(predictions)
                variance = np.var(predictions_array, axis=0)
                confidence_scores = 1.0 / (
                    1.0 + variance
                )  # Inverse variance as confidence

            finally:
                # Always restore original training state
                if not original_training:
                    self.model.eval()

        elif method == "simple":
            # Simple confidence based on prediction magnitude (no dropout needed)
            self.model.eval()
            predictions = self.predict(X)
            # Use absolute prediction values as confidence
            confidence_scores = np.abs(predictions)

        elif method == "margin":
            # Use prediction margin (distance from zero for regression) (no dropout needed)
            self.model.eval()
            predictions = self.predict(X)
            # For regression, margin is the distance from the mean prediction
            # Higher distance from mean = more confident (outlier predictions)
            mean_pred = np.mean(predictions)
            confidence_scores = np.abs(predictions - mean_pred)

        elif method == "leaf_depth":
            # For PyTorch models, use prediction stability as proxy for leaf_depth (no dropout needed)
            self.model.eval()
            # Get predictions with small input perturbations to measure stability
            predictions = self.predict(X)

            # Create slightly perturbed inputs
            X_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32)
            noise_scale = 0.01  # Small perturbation
            noise = torch.randn_like(X_tensor) * noise_scale
            X_perturbed = X_tensor + noise

            # Get predictions on perturbed data
            dataset = TensorDataset(X_perturbed)
            loader = DataLoader(
                dataset, batch_size=self.config.get("batch_size", 32), shuffle=False, num_workers=0
            )

            perturbed_predictions = []
            with torch.no_grad():
                for (batch_X,) in loader:
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X)
                    perturbed_predictions.append(outputs.cpu().numpy())

            perturbed_predictions = np.concatenate(perturbed_predictions).squeeze()

            # Stability = inverse of prediction change (more stable = higher confidence)
            prediction_change = np.abs(predictions - perturbed_predictions)
            confidence_scores = 1.0 / (1.0 + prediction_change)

        else:
            raise ValueError(f"Unknown confidence method: {method}")

        # Normalize confidence scores to [0, 1] range
        min_conf, max_conf = confidence_scores.min(), confidence_scores.max()
        if max_conf > min_conf:
            confidence_scores = (confidence_scores - min_conf) / (max_conf - min_conf)
        else:
            # All confidence scores are identical - this is problematic
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
