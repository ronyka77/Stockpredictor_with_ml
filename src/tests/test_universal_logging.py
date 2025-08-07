"""
Test script to demonstrate the universal log_to_mlflow function
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gradient_boosting.xgboost_model import XGBoostModel, log_to_mlflow
from utils.logger import get_logger

logger = get_logger(__name__)

def test_universal_logging():
    """Test the universal log_to_mlflow function"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate sample features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add a price column (required for XGBoost model)
    X['close'] = np.random.uniform(50, 200, n_samples)
    
    # Generate target (future price)
    y = pd.Series(X['close'] * (1 + np.random.normal(0.05, 0.1, n_samples)))
    
    logger.info("Creating and training XGBoost model...")
    
    # Create and train model
    model = XGBoostModel(
        model_name="test_universal_logging",
        prediction_horizon=10
    )
    
    # Train the model
    model.fit(X, y, validation_split=0.2)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate some metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        "mse": mean_squared_error(y, predictions),
        "mae": mean_absolute_error(y, predictions),
        "r2": r2_score(y, predictions),
        "samples": len(y)
    }
    
    # Get model parameters
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "prediction_horizon": 10
    }
    
    logger.info("Testing universal log_to_mlflow function...")
    
    # Test the universal logging function
    run_id = log_to_mlflow(
        model=model.model,
        metrics=metrics,
        params=params,
        experiment_name="test_universal_logging_experiment",
        X_eval=X
    )
    
    logger.info(f"✅ Universal logging completed! Run ID: {run_id}")
    
    # Test the class method version
    logger.info("Testing class method log_model_to_mlflow...")
    
    class_run_id = model.log_model_to_mlflow(
        metrics=metrics,
        params=params,
        X_eval=X,
        experiment_name="test_class_method_logging"
    )
    
    logger.info(f"✅ Class method logging completed! Run ID: {class_run_id}")
    
    # Test the updated save_model method
    logger.info("Testing updated save_model method...")
    
    save_method_run_id = model.save_model(
        metrics=metrics,
        params=params,
        X_eval=X,
        experiment_name="test_save_method_logging"
    )
    
    logger.info(f"✅ Save method logging completed! Run ID: {save_method_run_id}")
    
    return run_id, class_run_id, save_method_run_id

if __name__ == "__main__":
    logger.info("Starting universal logging test...")
    run_id, class_run_id, save_method_run_id = test_universal_logging()
    logger.info("Test completed! Run IDs:")
    logger.info(f"  Universal function: {run_id}")
    logger.info(f"  Class method: {class_run_id}")
    logger.info(f"  Save method: {save_method_run_id}") 