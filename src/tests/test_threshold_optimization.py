#!/usr/bin/env python3
"""
Test script to verify ThresholdEvaluator initialization during hyperparameter optimization
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.time_series.lstm_model import LSTMPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_hyperparameter_optimization():
    """Test hyperparameter optimization to verify ThresholdEvaluator initialization"""
    print("Starting hyperparameter optimization test...")
    
    # Initialize LSTM model
    lstm_model = LSTMPredictor(
        model_name="test_lstm_optimization",
        config={
            'input_size': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2,
            'epochs': 2,  # Reduced for testing
            'learning_rate': 0.001,
            'batch_size': 32,
            'sequence_length': 20
        }
    )
    
    print("LSTM model initialized. Starting hyperparameter optimization...")
    
    # Run hyperparameter optimization with reduced trials
    try:
        results = lstm_model.run_hyperparameter_optimization(
            n_trials=3,  # Reduced for testing
            prediction_horizon=10,
            experiment_name="test_threshold_optimization"
        )
        
        print("✅ Hyperparameter optimization completed successfully!")
        print(f"Best model saved to MLflow run: {results.get('saved_run_id', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error during hyperparameter optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hyperparameter_optimization() 