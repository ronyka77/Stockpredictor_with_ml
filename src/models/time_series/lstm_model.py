"""
LSTM Model Module

This module defines the LSTM model architecture and the predictor class
that handles its training and evaluation lifecycle.
"""
import torch.nn as nn
from typing import Dict, Any, Optional

from src.models.time_series.base_pytorch_model import PyTorchBasePredictor

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
    def __init__(self, model_name: str = "LSTM", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)

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