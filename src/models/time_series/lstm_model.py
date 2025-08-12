"""
LSTM Model Module

This module defines the LSTM model architecture and the predictor class
that handles its training and evaluation lifecycle.
"""
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
import mlflow
import optuna

from src.models.time_series.base_pytorch_model import PyTorchBasePredictor
from src.models.evaluation import ThresholdEvaluator
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLFlowManager
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.data_utils.sequential_data_loader import TimeSeriesDataset

logger = get_logger(__name__)

class LSTMModule(nn.Module):
    """
    Core LSTM model architecture.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Important for compatibility with DataLoader
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMPredictor(PyTorchBasePredictor):
    """
    Predictor class for the LSTM model.
    
    This class handles the creation, training, and prediction of the LSTMModule,
    leveraging the common logic from PyTorchBasePredictor.
    """
    def __init__(self, model_name: str = "LSTM", config: Optional[Dict[str, Any]] = None, threshold_evaluator: Optional[ThresholdEvaluator] = None):
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        self.sequence_length = config.get("sequence_length", 20) if config else 20

    def _create_model(self) -> nn.Module:
        """
        Creates the LSTMModule instance based on the model's configuration.
        """
        # Default config values if not provided
        input_size = self.config.get("input_size")
        if input_size is None:
            raise ValueError("LSTMPredictor config must include 'input_size'.")
            
        hidden_size = self.config.get("hidden_size", 128)
        num_layers = self.config.get("num_layers", 2)
        output_size = self.config.get("output_size", 1)
        dropout = self.config.get("dropout", 0.2)
        
        return LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )

    def fit(self, train_loader, val_loader=None, feature_names=None):
        """
        Train the LSTM model with mixed precision support (if CUDA available).
        Supports optimizer selection, weight decay, and gradient clipping.
        """
        self.model = self._create_model()
        self.model.to(self.device)
        if feature_names is not None:
            self.feature_names = feature_names

        # Training config
        epochs = self.config.get('epochs', 20)
        learning_rate = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        gradient_clip = self.config.get('gradient_clip', None)

        logger.info(f"Using optimizer: {optimizer_name}")
        criterion = nn.MSELoss()
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', defaulting to Adam.")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        use_amp = torch.cuda.is_available()
        if use_amp:
            scaler = torch.amp.GradScaler(device='cuda')
            logger.info("Using mixed precision training (torch.amp.GradScaler)")
        else:
            logger.info("Using standard float32 training (CPU or non-AMP GPU)")

        logger.info(f"Starting LSTM training for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                    # NaN/Inf loss check
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN/Inf loss detected! Batch stats:")
                        print("batch_X min/max:", batch_X.min().item(), batch_X.max().item())
                        print("batch_y min/max:", batch_y.min().item(), batch_y.max().item())
                        print("loss:", loss.item())
                        break
                    scaler.scale(loss).backward()
                    if gradient_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    # NaN/Inf loss check
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN/Inf loss detected! Batch stats:")
                        print("batch_X min/max:", batch_X.min().item(), batch_X.max().item())
                        print("batch_y min/max:", batch_y.min().item(), batch_y.max().item())
                        print("loss:", loss.item())
                        break
                    loss.backward()
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                        if use_amp:
                            with torch.amp.autocast(device_type='cuda'):
                                val_outputs = self.model(batch_X_val)
                                val_loss = criterion(val_outputs.squeeze(), batch_y_val)
                        else:
                            val_outputs = self.model(batch_X_val)
                            val_loss = criterion(val_outputs.squeeze(), batch_y_val)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / len(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.6f}")

        self.is_trained = True
        return self

    def set_scaler(self, scaler):
        """
        Set the feature scaler for prediction scaling.
        
        Args:
            scaler: Fitted StandardScaler instance
        """
        self.scaler = scaler
        logger.info("Feature scaler set for prediction scaling")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        This method handles both sequential data (for training) and regular 2D data (for evaluation).
        For regular 2D data, it creates overlapping sequences to maintain the LSTM's sequential nature.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        self.model.to(self.device)

        # Check if input is already 3D (sequential data)
        if len(X.shape) == 3:
            # Input is already sequential data, use base implementation
            return super().predict(X)
        
        # Input is 2D (regular feature matrix), need to create sequences
        # Apply scaling if scaler is available
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_scaled = self.scaler.transform(X[self.feature_names])
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            logger.info("Applied feature scaling for prediction")
        else:
            X_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32)
            logger.info("No scaler available, using raw features for prediction")
        
        # Create overlapping sequences for prediction
        # For prediction, we'll use the last sequence_length samples for each prediction
        # This maintains the sequential nature while allowing prediction on regular 2D data
        predictions = []
        
        with torch.no_grad():
            for i in range(len(X_tensor)):
                # For each sample, create a sequence using the last sequence_length samples
                # If we don't have enough history, pad with zeros
                if i < self.sequence_length - 1:
                    # Pad with zeros for early samples
                    padding_size = self.sequence_length - 1 - i
                    padding = torch.zeros(padding_size, X_tensor.shape[1], dtype=torch.float32)
                    sequence = torch.cat([padding, X_tensor[:i+1]], dim=0)
                else:
                    # Use the last sequence_length samples
                    sequence = X_tensor[i-self.sequence_length+1:i+1]
                
                # Add batch dimension and move to device
                sequence = sequence.unsqueeze(0).to(self.device)  # Shape: (1, sequence_length, features)
                
                # Make prediction
                output = self.model(sequence)
                predictions.append(output.cpu().numpy())
        
        return np.concatenate(predictions).squeeze()

    def predict_proba(self, X: pd.DataFrame, method: str = 'variance', 
                    n_passes: int = 10, return_confidence: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate probability-like outputs for LSTM predictions using uncertainty quantification.
        
        This method provides probability-like scores that indicate the model's confidence
        in its predictions. For regression tasks like stock prediction, this translates
        to confidence intervals and uncertainty estimates.
        
        Args:
            X: Feature matrix (pd.DataFrame)
            method: Method for uncertainty quantification:
                - 'variance': Use prediction variance across multiple forward passes
                - 'lstm_hidden': Use hidden state variance as uncertainty proxy
                - 'monte_carlo': Monte Carlo dropout for uncertainty estimation
                - 'ensemble': Use multiple model predictions (if available)
                - 'bootstrap': Bootstrap-style uncertainty estimation
            n_passes: Number of forward passes for uncertainty estimation
            return_confidence: Whether to return confidence scores alongside probabilities
            
        Returns:
            Dictionary containing:
                - 'predictions': Mean predictions
                - 'probabilities': Probability-like confidence scores [0, 1]
                - 'uncertainty': Uncertainty estimates (standard deviation)
                - 'confidence': Confidence scores (if return_confidence=True)
                - 'lower_bound': Lower confidence bound
                - 'upper_bound': Upper confidence bound
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making probability predictions")
        
        self.model.to(self.device)
        
        # Helper function to create sequences for 2D input
        def create_sequences_for_2d_input(X_tensor):
            """Create sequences for 2D input data"""
            sequences = []
            for i in range(len(X_tensor)):
                if i < self.sequence_length - 1:
                    # Pad with zeros for early samples
                    padding_size = self.sequence_length - 1 - i
                    padding = torch.zeros(padding_size, X_tensor.shape[1], dtype=torch.float32)
                    sequence = torch.cat([padding, X_tensor[:i+1]], dim=0)
                else:
                    # Use the last sequence_length samples
                    sequence = X_tensor[i-self.sequence_length+1:i+1]
                sequences.append(sequence)
            return torch.stack(sequences)
        
        # Convert input to tensor
        # Apply scaling if scaler is available
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_scaled = self.scaler.transform(X[self.feature_names])
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            logger.info("Applied feature scaling for predict_proba")
        else:
            X_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32)
            logger.info("No scaler available, using raw features for predict_proba")
        
        # Create sequences for 2D input
        if len(X_tensor.shape) == 2:
            X_tensor = create_sequences_for_2d_input(X_tensor)
        
        # Create DataLoader for batch processing
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.config.get('batch_size', 32), shuffle=False)
        
        if method == 'variance':
            # Method 1: Prediction variance across multiple forward passes
            original_training = self.model.training
            self.model.train()  # Enable dropout for variance calculation
            
            try:
                all_predictions = []
                
                for _ in range(n_passes):
                    pass_predictions = []
                    with torch.no_grad():
                        for batch_X, in loader:
                            batch_X = batch_X.to(self.device)
                            outputs = self.model(batch_X)
                            pass_predictions.append(outputs.cpu().numpy())
                    all_predictions.append(np.concatenate(pass_predictions).squeeze())
                
                # Calculate statistics across passes
                predictions_array = np.array(all_predictions)
                mean_predictions = np.mean(predictions_array, axis=0)
                std_predictions = np.std(predictions_array, axis=0)
                
                # Convert to probability-like scores (inverse of uncertainty)
                probabilities = 1.0 / (1.0 + std_predictions)
                
                # Calculate confidence bounds (95% confidence interval)
                confidence_level = 0.95
                z_score = 1.96  # 95% confidence interval
                lower_bound = mean_predictions - z_score * std_predictions
                upper_bound = mean_predictions + z_score * std_predictions
                
            finally:
                # Always restore original training state
                if not original_training:
                    self.model.eval()
        
        elif method == 'lstm_hidden':
            # Method 2: Hidden state variance as uncertainty proxy
            original_training = self.model.training
            self.model.train()  # Enable dropout for variance calculation
            
            try:
                all_predictions = []
                all_hidden_states = []
                
                for _ in range(n_passes):
                    pass_predictions = []
                    pass_hidden = []
                    
                    with torch.no_grad():
                        for batch_X, in loader:
                            batch_X = batch_X.to(self.device)
                            # Get LSTM output and hidden states
                            lstm_out, (hidden, cell) = self.model.lstm(batch_X)
                            outputs = self.model.fc(lstm_out[:, -1, :])
                            
                            pass_predictions.append(outputs.cpu().numpy())
                            # Use the last hidden state from the last layer
                            last_hidden = hidden[-1]  # Shape: (batch_size, hidden_size)
                            pass_hidden.append(last_hidden.cpu().numpy())
                    
                    all_predictions.append(np.concatenate(pass_predictions).squeeze())
                    all_hidden_states.append(np.concatenate(pass_hidden, axis=0))
                
                # Calculate prediction statistics
                predictions_array = np.array(all_predictions)
                mean_predictions = np.mean(predictions_array, axis=0)
                std_predictions = np.std(predictions_array, axis=0)
                
                # Calculate hidden state variance as additional uncertainty measure
                hidden_array = np.array(all_hidden_states)
                hidden_variance = np.var(hidden_array, axis=0)
                avg_hidden_variance = np.mean(hidden_variance, axis=1)
                
                # Combine prediction uncertainty with hidden state uncertainty
                combined_uncertainty = std_predictions + 0.1 * avg_hidden_variance
                probabilities = 1.0 / (1.0 + combined_uncertainty)
                
                # Calculate confidence bounds
                confidence_level = 0.95
                z_score = 1.96
                lower_bound = mean_predictions - z_score * combined_uncertainty
                upper_bound = mean_predictions + z_score * combined_uncertainty
                
            finally:
                if not original_training:
                    self.model.eval()
        
        elif method == 'monte_carlo':
            # Method 3: Monte Carlo dropout for uncertainty estimation
            original_training = self.model.training
            self.model.train()  # Enable dropout
            
            try:
                all_predictions = []
                
                for _ in range(n_passes):
                    pass_predictions = []
                    with torch.no_grad():
                        for batch_X, in loader:
                            batch_X = batch_X.to(self.device)
                            outputs = self.model(batch_X)
                            pass_predictions.append(outputs.cpu().numpy())
                    all_predictions.append(np.concatenate(pass_predictions).squeeze())
                
                # Calculate statistics
                predictions_array = np.array(all_predictions)
                mean_predictions = np.mean(predictions_array, axis=0)
                std_predictions = np.std(predictions_array, axis=0)
                
                # Monte Carlo dropout probabilities
                probabilities = 1.0 / (1.0 + std_predictions)
                
                # Confidence bounds
                confidence_level = 0.95
                z_score = 1.96
                lower_bound = mean_predictions - z_score * std_predictions
                upper_bound = mean_predictions + z_score * std_predictions
                
            finally:
                if not original_training:
                    self.model.eval()
        
        elif method == 'bootstrap':
            # Method 4: Bootstrap-style uncertainty estimation
            # This simulates bootstrap by adding noise to inputs
            self.model.eval()
            
            all_predictions = []
            noise_scale = 0.01  # Small perturbation scale
            
            for _ in range(n_passes):
                pass_predictions = []
                
                # Add noise to input
                noise = torch.randn_like(X_tensor) * noise_scale
                X_noisy = X_tensor + noise
                
                with torch.no_grad():
                    for batch_X, in loader:
                        batch_X = X_noisy[loader.dataset.indices] if hasattr(loader.dataset, 'indices') else X_noisy
                        batch_X = batch_X.to(self.device)
                        outputs = self.model(batch_X)
                        pass_predictions.append(outputs.cpu().numpy())
                
                all_predictions.append(np.concatenate(pass_predictions).squeeze())
            
            # Calculate statistics
            predictions_array = np.array(all_predictions)
            mean_predictions = np.mean(predictions_array, axis=0)
            std_predictions = np.std(predictions_array, axis=0)
            
            # Bootstrap probabilities
            probabilities = 1.0 / (1.0 + std_predictions)
            
            # Confidence bounds
            confidence_level = 0.95
            z_score = 1.96
            lower_bound = mean_predictions - z_score * std_predictions
            upper_bound = mean_predictions + z_score * std_predictions
        
        else:
            # Fallback to simple variance method
            logger.warning(f"Unknown method '{method}', falling back to 'variance'")
            return self.predict_proba(X, method='variance', n_passes=n_passes, return_confidence=return_confidence)
        
        # Normalize probabilities to [0, 1] range
        min_prob, max_prob = probabilities.min(), probabilities.max()
        if max_prob > min_prob:
            probabilities = (probabilities - min_prob) / (max_prob - min_prob)
        else:
            logger.warning(f"All probabilities are identical ({min_prob:.4f}) - using uniform distribution")
            probabilities = np.full_like(probabilities, 0.5)
        
        # Prepare return dictionary
        result = {
            'predictions': mean_predictions,
            'probabilities': probabilities,
            'uncertainty': std_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': method,
            'n_passes': n_passes,
            'confidence_level': confidence_level
        }
        
        # Add confidence scores if requested
        if return_confidence:
            confidence_scores = self.get_prediction_confidence(X, method='lstm_hidden')
            result['confidence'] = confidence_scores
        
        logger.info(f"LSTM predict_proba - Method: {method}")
        logger.info(f"  Predictions range: [{mean_predictions.min():.4f}, {mean_predictions.max():.4f}]")
        logger.info(f"  Probabilities range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
        logger.info(f"  Uncertainty range: [{std_predictions.min():.4f}, {std_predictions.max():.4f}]")
        
        return result

    def predict_proba_simple(self, X: pd.DataFrame, method: str = 'variance', n_passes: int = 10) -> np.ndarray:
        """
        Simplified predict_proba that returns only probability-like scores.
        
        This method provides a scikit-learn-like interface for getting probability scores.
        
        Args:
            X: Feature matrix (pd.DataFrame)
            method: Method for uncertainty quantification
            n_passes: Number of forward passes for uncertainty estimation
            
        Returns:
            Array of probability-like scores [0, 1] where higher values indicate higher confidence
        """
        result = self.predict_proba(X, method=method, n_passes=n_passes, return_confidence=False)
        return result['probabilities']

    def get_prediction_intervals(self, X: pd.DataFrame, confidence_level: float = 0.95, 
                                method: str = 'variance', n_passes: int = 10) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals with confidence bounds.
        
        This method provides confidence intervals for predictions, which is useful for
        risk management and uncertainty quantification in stock prediction.
        
        Args:
            X: Feature matrix (pd.DataFrame)
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            method: Method for uncertainty quantification
            n_passes: Number of forward passes for uncertainty estimation
            
        Returns:
            Dictionary containing:
                - 'predictions': Mean predictions
                - 'lower_bound': Lower confidence bound
                - 'upper_bound': Upper confidence bound
                - 'confidence_level': The confidence level used
                - 'method': The method used for uncertainty estimation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting prediction intervals")
        
        # Get comprehensive predict_proba results
        proba_result = self.predict_proba(X, method=method, n_passes=n_passes, return_confidence=False)
        
        # Extract components
        predictions = proba_result['predictions']
        lower_bound = proba_result['lower_bound']
        upper_bound = proba_result['upper_bound']
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'method': method,
            'n_passes': n_passes
        }

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'variance') -> np.ndarray:
        """
        Calculate confidence scores for LSTM predictions using predict_proba methods.
        This function now leverages the predict_proba implementation to always return
        probability-like confidence scores.
        
        Args:
            X: Feature matrix
            method: Confidence calculation method - uses predict_proba methods:
                - 'variance': Use prediction variance across multiple forward passes
                - 'lstm_hidden': Use hidden state variance as uncertainty proxy
                - 'monte_carlo': Monte Carlo dropout for uncertainty estimation
                - 'bootstrap': Bootstrap-style uncertainty estimation
            
        Returns:
            Array of probability-like confidence scores [0, 1] (higher = more confident)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating confidence")
        
        # Map method names to predict_proba compatible methods
        method_mapping = {
            'variance': 'variance',
            'lstm_hidden': 'lstm_hidden', 
            'monte_carlo': 'monte_carlo',
            'bootstrap': 'bootstrap',
            'simple': 'variance',  # Map simple to variance method
            'margin': 'variance',  # Map margin to variance method
            'leaf_depth': 'bootstrap'  # Map leaf_depth to bootstrap method
        }
        
        # Get the corresponding predict_proba method
        proba_method = method_mapping.get(method, 'variance')
        
        # Use predict_proba to get probability-like confidence scores
        try:
            # Use predict_proba_simple for efficiency (returns only probabilities)
            confidence_scores = self.predict_proba_simple(
                X, 
                method=proba_method, 
                n_passes=5  # Reduced for efficiency in confidence calculation
            )
            
            logger.info(f"LSTM confidence using {method} -> {proba_method} method")
            logger.info(f"LSTM confidence - Range: [{confidence_scores.min():.4f}, {confidence_scores.max():.4f}]")
            logger.info(f"LSTM confidence - Mean: {confidence_scores.mean():.4f}, std: {confidence_scores.std():.4f}")
            
            return confidence_scores
            
        except Exception as e:
            logger.warning(f"Error using predict_proba method '{proba_method}', falling back to simple variance: {str(e)}")
            
            # Fallback: use simple variance method
            try:
                confidence_scores = self.predict_proba_simple(
                    X, 
                    method='variance', 
                    n_passes=3  # Minimal passes for fallback
                )
                return confidence_scores
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {str(fallback_error)}")
                # Last resort: return uniform confidence
                return np.full(len(X), 0.5)

    def optimize_prediction_threshold(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    current_prices_test: np.ndarray,
                                    confidence_method: str = 'lstm_hidden',
                                    threshold_range: Tuple[float, float] = (0.1, 0.9),
                                    n_thresholds: int = 80) -> Dict[str, Any]:
        """
        Optimize prediction threshold based on confidence scores to maximize profit on test data
        
        Args:
            X_test: Test features (unseen data)
            y_test: Test targets
            current_prices_test: Current prices for test set
            confidence_method: Method for calculating confidence scores
            threshold_range: Range of thresholds to test (min, max)
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with optimization results based on test data only
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before optimizing thresholds")
        
        # Use central evaluator for threshold optimization
        results = self.threshold_evaluator.optimize_prediction_threshold_lstm(
            model=self,
            X_test=X_test,
            y_test=y_test,
            current_prices_test=current_prices_test,
            confidence_method=confidence_method,
            threshold_range=threshold_range,
            n_thresholds=n_thresholds
        )
        
        # Store optimal threshold if optimization was successful
        if results['status'] == 'success':
            self.optimal_threshold = results['optimal_threshold']
            self.confidence_method = results['confidence_method']
        
        return results

    def _create_model_for_tuning(self, params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None) -> 'LSTMPredictor':
        """
        Create a new LSTM model instance with specified parameters for hyperparameter tuning
        
        Args:
            params: Dictionary of LSTM parameters
            model_name: Optional custom name for the new model
            
        Returns:
            New LSTMPredictor instance configured with the provided parameters
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
        
        new_model = LSTMPredictor(
            model_name=model_name,
            config=config,
            threshold_evaluator=self.threshold_evaluator
        )
        
        logger.info(f"Created new LSTM model '{model_name}' with parameters: {params}")
        
        return new_model

    def _prepare_data_for_tuning(self, sequence_length: int = 20, batch_size: int = 1024):
        """
        Prepare data for LSTM hyperparameter tuning
        
        Args:
            sequence_length: Length of sequences for LSTM
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing prepared data and loaders
        """
        logger.info("STEP 1: Preparing data with stationarity transformation...")
        data_dict = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=getattr(self, 'prediction_horizon', 10),
            apply_stationarity_transform=True,
        )
        
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        
        # STEP 2: Feature scaling to prevent NaN loss
        logger.info("STEP 2: Applying feature scaling...")
        from sklearn.preprocessing import StandardScaler
        
        # Initialize scaler and fit on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Log scaling results
        logger.info("Feature scaling completed:")
        logger.info(f"  X_train before scaling - min: {X_train.min().min():.2f}, max: {X_train.max().max():.2f}")
        logger.info(f"  X_train after scaling - min: {X_train_scaled.min().min():.2f}, max: {X_train_scaled.max().max():.2f}")
        logger.info(f"  X_train after scaling - mean: {X_train_scaled.mean().mean():.2f}, std: {X_train_scaled.std().mean():.2f}")
        
        # Create datasets with scaled features
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train, sequence_length=sequence_length)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test, sequence_length=sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        return {
            'data_dict': data_dict,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'X_train': X_train_scaled,  # Return scaled features
            'X_test': X_test_scaled,    # Return scaled features
            'y_train': y_train,
            'y_test': y_test,
            'current_prices': X_test['close'].values,  # Keep original close prices for evaluation
            'scaler': scaler  # Store scaler for later use
        }

    def create_hyperparameter_objective(self, data_prep: Dict[str, Any], objective_column: str = 'test_profit_per_investment') -> callable:
        """
        Create Optuna objective function for hyperparameter optimization with threshold optimization
        
        Args:
            data_prep: Dictionary containing prepared data and loaders
            objective_column: Column to optimize for
            
        Returns:
            Objective function for Optuna optimization with threshold optimization
        """
        # Initialize tracking variables for best model
        self.best_investment_success_rate = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None
        
        def objective(trial):
            """Objective function for Optuna optimization with threshold optimization for each trial"""
            
            # Suggest hyperparameters for LSTM
            params = {
                'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512]),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 0.0001, 0.01, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
                'epochs': trial.suggest_int('epochs', 5, 30),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                'sequence_length': trial.suggest_int('sequence_length', 10, 60),
                'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 5.0),
            }
            
            # Add LSTM-specific parameters
            params.update({
                'input_size': data_prep['X_train'].shape[1],
                'output_size': 1,
            })
            
            try:
                # Create model with trial parameters
                trial_model = self._create_model_for_tuning(
                    params=params,
                    model_name=f"lstm_trial_{trial.number}"
                )
                
                # Set the scaler for prediction scaling
                if 'scaler' in data_prep:
                    trial_model.set_scaler(data_prep['scaler'])
                
                # Disable MLflow for trial models to avoid clutter
                trial_model.disable_mlflow = True
                
                # Prepare data with trial sequence length
                sequence_length = params['sequence_length']
                batch_size = params['batch_size']
                
                # Use scaled data from data_prep (already scaled in _prepare_data_for_tuning)
                train_dataset = TimeSeriesDataset(data_prep['X_train'], data_prep['y_train'], sequence_length=sequence_length)
                test_dataset = TimeSeriesDataset(data_prep['X_test'], data_prep['y_test'], sequence_length=sequence_length)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                
                # Train the model
                trial_model.fit(
                    train_loader, 
                    val_loader=test_loader, 
                    feature_names=data_prep['X_train'].columns.tolist()
                )
                
                # Run threshold optimization for this trial
                logger.info(f"Running threshold optimization for trial {trial.number}")
                
                threshold_results = trial_model.optimize_prediction_threshold(
                    X_test=data_prep['X_test'],
                    y_test=data_prep['y_test'],
                    current_prices_test=data_prep['current_prices'],
                    confidence_method='lstm_hidden',
                    threshold_range=(0.1, 0.9),
                    n_thresholds=80
                )
                
                # Use threshold-optimized investment success rate
                optimized_profit_score = threshold_results['best_result'][objective_column]
                
                # Store additional threshold info for logging
                threshold_info = {
                    'optimal_threshold': threshold_results['optimal_threshold'],
                    'samples_kept_ratio': threshold_results['best_result']['test_samples_ratio'],
                    'investment_success_rate': threshold_results['best_result']['investment_success_rate'],
                    'custom_accuracy': threshold_results['best_result']['test_custom_accuracy'],
                    'total_threshold_profit': threshold_results['best_result']['test_profit'],
                    'profitable_investments': threshold_results['best_result']['profitable_investments'],
                    'profit_per_investment': threshold_results['best_result']['test_profit_per_investment']
                }
                
                # Log threshold optimization results for this trial
                logger.info(f"Trial {trial.number} threshold optimization:")
                logger.info(f"  Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                logger.info(f"  Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                logger.info(f"  Optimized {objective_column}: {optimized_profit_score:.3f}")
                
                # Check if this is the best trial so far
                if optimized_profit_score > self.best_investment_success_rate:
                    self.best_investment_success_rate = optimized_profit_score
                    self.best_trial_model = trial_model
                    self.best_trial_params = params.copy()
                    self.best_threshold_info = threshold_info.copy()
                    
                    logger.info(f"🎯 NEW BEST TRIAL {trial.number}: {objective_column} = {optimized_profit_score:.3f}")
                    if threshold_info['optimal_threshold'] is not None:
                        logger.info(f"   Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                        logger.info(f"   Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                        logger.info(f"   Custom accuracy: {threshold_info['custom_accuracy']:.3f}")
                    
                    self.previous_best = optimized_profit_score
                else:
                    logger.info(f"Trial {trial.number}: Investment Success Rate = {optimized_profit_score:.3f} (Best: {self.best_investment_success_rate:.3f})")
                
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
            "model_updated": self.best_trial_model is not None
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
                    "confidence_method": 'lstm_hidden'
                }
            })
        else:
            base_info["threshold_optimization"] = None
        
        return base_info

    def finalize_best_model(self) -> 'LSTMPredictor':
        """
        Finalize the best model after hyperparameter optimization with threshold optimization
        This ensures the main model instance contains the best performing model and threshold info
        
        Returns:
            The best LSTM model with optimal parameters
        """
        if hasattr(self, 'best_trial_model') and self.best_trial_model is not None:
            # Copy the best model's state to this instance
            best_model = self.best_trial_model
            
            # Copy threshold optimization information if available
            if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                if self.best_threshold_info.get('optimal_threshold') is not None:
                    best_model.optimal_threshold = self.best_threshold_info['optimal_threshold']
                    best_model.confidence_method = 'lstm_hidden'
            
            # Set the scaler if available (for prediction scaling)
            if hasattr(self, 'data_prep_scaler') and self.data_prep_scaler is not None:
                best_model.set_scaler(self.data_prep_scaler)
                logger.info("✅ Feature scaler set on best model for prediction scaling")
            
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
            
            return best_model
        else:
            logger.warning("⚠ No best model found to finalize")
            return None

    def run_hyperparameter_optimization(self, n_trials: int = 20, prediction_horizon: int = 10, 
                                        experiment_name: str = "lstm_stock_predictor") -> Dict[str, Any]:
        """
        Run complete hyperparameter optimization pipeline
        
        Args:
            n_trials: Number of Optuna trials to run
            prediction_horizon: Prediction horizon in days
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("=" * 80)
        logger.info("🎯 LSTM HYPERPARAMETER OPTIMIZATION PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Store prediction horizon for data preparation
            self.prediction_horizon = prediction_horizon
            
            # 0. Setup MLflow experiment tracking
            logger.info("0. Setting up MLflow experiment tracking...")
            mlflow_manager = MLFlowManager()
            experiment_id = mlflow_manager.setup_experiment(experiment_name)
            
            logger.info(f"✅ MLflow experiment setup completed: {experiment_id}")
            logger.info(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
            
            # 1. Prepare data
            logger.info("1. Preparing data...")
            data_prep = self._prepare_data_for_tuning()
            
            # Store scaler for later use in finalize_best_model
            if 'scaler' in data_prep:
                self.data_prep_scaler = data_prep['scaler']
                logger.info("✅ Feature scaler stored for best model finalization")
            
            logger.info("✅ Data preparation completed")
            logger.info(f"   Train samples: {len(data_prep['y_train'])}")
            logger.info(f"   Test samples: {len(data_prep['y_test'])}")
            logger.info(f"   Features: {data_prep['X_train'].shape[1]}")
            
            # 2. Hyperparameter optimization
            logger.info("2. Starting hyperparameter optimization...")
            
            # Create objective function
            objective_function = self.create_hyperparameter_objective(data_prep)
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_function, n_trials=n_trials, n_jobs=1)
            
            # Get best results from study
            best_params = study.best_params
            best_profit = study.best_value  # This is now threshold-optimized profit per investment
            
            logger.info("🎯 Hyperparameter optimization with threshold optimization completed!")
            logger.info(f"✅ Best Threshold-Optimized Profit per Investment: ${best_profit:.2f}")
            logger.info(f"✅ Best parameters: {best_params}")
            
            # 3. Finalize the best model
            logger.info("3. Finalizing best model...")
            final_model = self.finalize_best_model()
            
            if final_model is None:
                raise RuntimeError("Failed to finalize best model")
            
            # Get best trial info for verification
            best_trial_info = self.get_best_trial_info()
            logger.info(f"✅ Best trial info: {best_trial_info}")
            
            # 4. Final evaluation with the best model
            logger.info("4. Making final evaluation with best model...")
            
            # Check if the best model has threshold optimization results
            has_threshold_optimization = (hasattr(self, 'best_threshold_info') and 
                                        self.best_threshold_info is not None and
                                        self.best_threshold_info.get('optimal_threshold') is not None)
            
            if has_threshold_optimization:
                logger.info("✅ Best model includes threshold optimization results")
                
                # Evaluate with the optimal threshold from hyperparameter optimization
                optimal_threshold = getattr(final_model, 'optimal_threshold', 0.5)
                confidence_method = getattr(final_model, 'confidence_method', 'lstm_hidden')
                
                threshold_performance = final_model.threshold_evaluator.evaluate_threshold_performance(
                    model=final_model,
                    X_test=data_prep['X_test'],
                    y_test=data_prep['y_test'],
                    current_prices_test=data_prep['current_prices'],
                    threshold=optimal_threshold,
                    confidence_method=confidence_method
                )
                
                # Also get unfiltered baseline for comparison
                baseline_predictions = final_model.predict(data_prep['X_test'])
                baseline_profit = final_model.threshold_evaluator.calculate_profit_score(
                    data_prep['y_test'].values, baseline_predictions, data_prep['current_prices']
                )
                baseline_profit_per_investment = baseline_profit / len(data_prep['y_test'])
                # Use threshold-optimized metrics for final evaluation
                final_profit_per_investment = threshold_performance['test_profit_per_investment']
                final_total_profit = threshold_performance['total_profit']
                final_investment_success_rate = threshold_performance['investment_success_rate']
                final_samples_kept = threshold_performance['samples_evaluated']

                logger.info("📊 Final Results Comparison:")
                logger.info(f"   Baseline (unfiltered) profit per investment: ${baseline_profit_per_investment:.2f}")
                logger.info(f"   Threshold-optimized profit per investment: ${final_profit_per_investment:.2f}")
                logger.info(f"   Improvement ratio: {final_profit_per_investment / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0:.2f}x")
                logger.info(f"   Samples kept: {final_samples_kept}/{len(data_prep['X_test'])} ({threshold_performance['samples_kept_ratio']:.1%})")
                logger.info(f"   Investment success rate: {final_investment_success_rate:.3f}")
                
                # Traditional metrics on filtered data
                final_mse = threshold_performance['mse']
                final_mae = threshold_performance['mae']
                final_r2 = threshold_performance['r2_score']
            
            # 5. Log model to MLflow
            logger.info("5. Logging model to MLflow...")
            
            # Prepare comprehensive metrics for logging
            final_metrics = {
                "final_total_profit": final_total_profit,
                "final_profit_per_investment": final_profit_per_investment,
                "final_investment_success_rate": final_investment_success_rate,
                "final_mse": final_mse,
                "final_mae": final_mae,
                "final_r2": final_r2,
                "final_test_samples": len(data_prep['y_test']),
                "hypertuning_best_profit": best_profit,
                "hypertuning_trials_completed": n_trials
            }
            
            # Prepare comprehensive parameters for logging
            final_params = best_params.copy()
            final_params.update({
                "prediction_horizon": prediction_horizon,
                "hypertuning_trials": n_trials,
                "hypertuning_direction": "maximize",
                "hypertuning_metric": "profit_score",
                "feature_count": data_prep['X_train'].shape[1],
                "train_samples": len(data_prep['y_train']),
                "test_samples": len(data_prep['y_test']),
                "threshold_optimization_enabled": has_threshold_optimization
            })
            
            # Add threshold parameters if optimization was successful
            if has_threshold_optimization:
                final_params.update({
                    "threshold_method": final_model.confidence_method,
                    "threshold_optimization_during_hypertuning": True,
                    "optimal_threshold_from_hypertuning": self.best_threshold_info['optimal_threshold']
                })
            
            # Save model to MLflow
            saved_run_id = final_model.save_model(experiment_name, final_metrics=final_metrics)
            
            logger.info("=" * 80)
            logger.info("🎉 LSTM HYPERPARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"📊 Dataset: {len(data_prep['y_train']) + len(data_prep['y_test']):,} samples, {data_prep['X_train'].shape[1]} features")
            logger.info(f"🎯 Target: {prediction_horizon}-day horizon")
            logger.info(f"🔧 Hypertuning: {n_trials} trials completed (optimizing for profit)")
            logger.info(f"📈 Final Total Profit: ${final_total_profit:.2f}")
            logger.info(f"📈 Average Profit per Investment: ${final_profit_per_investment:.2f}")
            logger.info(f"📈 Traditional MSE: {final_mse:.4f}")
            logger.info("🎯 Threshold optimization included for prediction filtering")
            logger.info(f"💾 Model saved to MLflow run: {saved_run_id}")
            logger.info("=" * 80)
            
            return {
                'best_model': final_model,
                'best_params': best_params,
                'best_profit': best_profit,
                'final_metrics': final_metrics,
                'saved_run_id': saved_run_id,
                'has_threshold_optimization': has_threshold_optimization
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in LSTM hyperparameter optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_model(self, run_id: str) -> None:
        """
        Load a saved LSTM model from MLflow using run ID
        
        Args:
            run_id: MLflow run ID to load model from
        """
        try:
            logger.info(f"Loading LSTM model from MLflow run: {run_id}")
            
            # Get the model artifact path from the run
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(run_id)
            
            # List artifacts to find the model
            artifacts = client.list_artifacts(run_id)
            model_artifact_path = None
            
            # Look for PyTorch model artifact
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
            
            # Load the PyTorch model
            loaded_model = mlflow.pytorch.load_model(model_uri)
            self.model = loaded_model
            
            # Load scaler if available
            scaler_artifact_path = None
            for artifact in artifacts:
                if artifact.path == "scaler/scaler.pkl" or artifact.path.endswith("scaler.pkl"):
                    scaler_artifact_path = artifact.path
                    break
            
            if scaler_artifact_path:
                try:
                    import pickle
                    
                    # Download scaler artifact
                    scaler_local_path = client.download_artifacts(run_id, scaler_artifact_path)
                    
                    with open(scaler_local_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    
                    logger.info("✅ Feature scaler loaded from MLflow artifacts")
                    
                    # Clean up downloaded file
                    import os
                    os.unlink(scaler_local_path)
                    
                except Exception as scaler_error:
                    logger.warning(f"⚠️ Could not load scaler: {str(scaler_error)}")
            else:
                logger.info("ℹ️ No scaler artifact found - model will use raw features for prediction")
            
            # Load additional metadata from run
            self._load_metadata_from_run(run_info)
            
            logger.info(f"✅ LSTM model loaded successfully from MLflow run: {run_id}")
                
        except Exception as e:
            logger.error(f"❌ Error loading LSTM model from MLflow run {run_id}: {str(e)}")
            raise

    def load_performance_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Load performance metrics from MLflow run
        
        Args:
            run_id: MLflow run ID to load metrics from
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(run_id)
            
            metrics = {}
            
            # Load metrics from run data
            if run_info.data.metrics:
                for metric_name, metric_value in run_info.data.metrics.items():
                    metrics[metric_name] = metric_value
            
            # Load parameters that might be metrics (with 'final_' prefix)
            if run_info.data.params:
                for param_name, param_value in run_info.data.params.items():
                    if param_name.startswith(('final_')):
                        # Try to convert string parameters back to numeric if possible
                        try:
                            if '.' in param_value:
                                metrics[param_name] = float(param_value)
                            else:
                                metrics[param_name] = int(param_value)
                        except ValueError:
                            metrics[param_name] = param_value
            
            logger.info(f"✅ Loaded {len(metrics)} performance metrics from MLflow run: {run_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error loading performance metrics from MLflow run {run_id}: {str(e)}")
            return {}

    def get_model_performance_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary for a saved model
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary containing performance summary
        """
        try:
            # Load the model first if not already loaded
            if not self.is_trained:
                self.load_model(run_id)
            
            # Load performance metrics
            metrics = self.load_performance_metrics(run_id)
            
            # Create performance summary
            summary = {
                'run_id': run_id,
                'model_name': self.model_name,
                'is_trained': self.is_trained,
                'has_scaler': hasattr(self, 'scaler') and self.scaler is not None,
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'metrics': metrics
            }
            
            # Extract key performance indicators
            key_metrics = {}
            for metric_name, value in metrics.items():
                if any(keyword in metric_name.lower() for keyword in ['profit', 'accuracy', 'mse', 'mae', 'r2', 'threshold']):
                    key_metrics[metric_name] = value
            
            summary['key_metrics'] = key_metrics
            
            logger.info(f"✅ Performance summary created for run: {run_id}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error creating performance summary for run {run_id}: {str(e)}")
            return {'error': str(e)}

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
            self.sequence_length = int(params.get('model_sequence_length', self.sequence_length))
            
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
                        # Extract feature names from signature
                        feature_names = []
                        for input_spec in model_info.signature.inputs:
                            if hasattr(input_spec, 'name'):
                                feature_names.append(input_spec.name)
                            else:
                                # Handle different signature formats
                                feature_names.append(str(input_spec))
                        
                        if feature_names:
                            self.feature_names = feature_names
                            logger.info(f"✅ Extracted {len(feature_names)} feature names from model signature")
                        else:
                            logger.warning("⚠️ No feature names found in model signature")
                    else:
                        logger.warning("⚠️ Model signature has no inputs")
                else:
                    logger.warning("⚠️ No model signature found")
                        
            except Exception as signature_error:
                logger.warning(f"⚠️ Could not extract feature names from signature: {str(signature_error)}")
                logger.info(f"Signature error details: {type(signature_error).__name__}: {signature_error}")
                
            logger.info("Model metadata loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load all metadata: {str(e)}")

    @classmethod
    def load_from_mlflow(cls, run_id: str) -> 'LSTMPredictor':
        """
        Class method to create a new LSTMPredictor instance and load from MLflow
        
        Args:
            run_id: MLflow run ID to load model from
            
        Returns:
            New LSTMPredictor instance with loaded model
        """
        # Create new instance
        lstm_model = cls()
        
        # Load model from MLflow
        lstm_model.load_model(run_id=run_id)
        
        return lstm_model

    def save_model(self, experiment_name: str, final_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Save LSTM model to MLflow with performance metrics
        
        Args:
            experiment_name: MLflow experiment name
            final_metrics: Dictionary containing final performance metrics
            
        Returns:
            MLflow run ID
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            import mlflow
            import mlflow.pytorch
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
            # Start a new run
            with mlflow.start_run() as run:
                self.run_id = run.info.run_id
                
                # Log model parameters
                params = {
                    'model_model_name': self.model_name,
                    'model_sequence_length': str(self.sequence_length),
                    'model_input_size': str(self.config.get('input_size', 'Unknown')),
                    'model_hidden_size': str(self.config.get('hidden_size', 'Unknown')),
                    'model_num_layers': str(self.config.get('num_layers', 'Unknown')),
                    'model_output_size': str(self.config.get('output_size', 'Unknown')),
                    'model_dropout': str(self.config.get('dropout', 'Unknown')),
                    'model_epochs': str(self.config.get('epochs', 'Unknown')),
                    'model_learning_rate': str(self.config.get('learning_rate', 'Unknown')),
                    'model_batch_size': str(self.config.get('batch_size', 'Unknown')),
                    'model_weight_decay': str(self.config.get('weight_decay', 'Unknown')),
                    'model_optimizer': str(self.config.get('optimizer', 'Unknown')),
                    'model_gradient_clip': str(self.config.get('gradient_clip', 'Unknown')),
                    'feature_count': str(len(self.feature_names) if self.feature_names else 0),
                    'is_trained': str(self.is_trained)
                }
                
                if self.feature_names:
                    params['feature_names'] = str(self.feature_names)
                
                mlflow.log_params(params)
                
                # Log final performance metrics if available
                if final_metrics:
                    logger.info("📊 Logging final performance metrics to MLflow...")
                    for metric_name, metric_value in final_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)
                            logger.info(f"  Logged metric: {metric_name} = {metric_value}")
                        else:
                            # Convert non-numeric metrics to string parameters
                            mlflow.log_param(f"final_{metric_name}", str(metric_value))
                            logger.info(f"  Logged parameter: final_{metric_name} = {metric_value}")
                
                # Log the PyTorch model
                mlflow.pytorch.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name=f"{self.model_name}_stock_prediction"
                )
                
                # Save the scaler if available
                if hasattr(self, 'scaler') and self.scaler is not None:
                    import pickle
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
                        pickle.dump(self.scaler, tmp_file)
                        tmp_file_path = tmp_file.name
                    
                    mlflow.log_artifact(tmp_file_path, artifact_path="scaler")
                    import os
                    os.unlink(tmp_file_path)  # Clean up temp file
                    logger.info("✅ Feature scaler saved to MLflow artifacts")
                
                logger.info(f"LSTM model saved to MLflow with run ID: {self.run_id}")
                return self.run_id
                
        except Exception as e:
            logger.error(f"❌ Error saving LSTM model to MLflow: {str(e)}")
            raise

def main():
    """
    Example usage of LSTM model with hyperparameter optimization
    """
    logger.info("=" * 80)
    logger.info("🚀 LSTM MODEL WITH HYPERPARAMETER OPTIMIZATION EXAMPLE")
    logger.info("=" * 80)
    
    try:
        # Initialize LSTM model
        lstm_model = LSTMPredictor(
            model_name="lstm_optimized_example",
            config={
                'input_size': 100,  # Will be set automatically during optimization
                'hidden_size': 128,
                'num_layers': 2,
                'output_size': 1,
                'dropout': 0.2,
                'epochs': 10,
                'learning_rate': 0.001,
                'batch_size': 512,
                'sequence_length': 20
            }
        )
        
        # Run hyperparameter optimization
        results = lstm_model.run_hyperparameter_optimization(
            n_trials=4,  # Reduced for example
            prediction_horizon=10,
            experiment_name="lstm_model"
        )
        
        logger.info("🎉 Example completed successfully!")
        logger.info(f"Best model saved to MLflow run: {results['saved_run_id']}")
        
        # Demonstrate loading performance metrics
        if results['saved_run_id']:
            logger.info("📊 Loading performance metrics from saved model...")
            performance_summary = lstm_model.get_model_performance_summary(results['saved_run_id'])
            
            if 'key_metrics' in performance_summary:
                logger.info("🎯 Key Performance Metrics:")
                for metric_name, value in performance_summary['key_metrics'].items():
                    logger.info(f"  {metric_name}: {value}")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 