"""
Custom Metrics for Financial Model Evaluation

This module provides specialized metrics for evaluating stock prediction models,
focusing on custom accuracy metrics that reward conservative predictions.
"""

import numpy as np
import pandas as pd
from typing import Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

class CustomMetrics:
    """
    Custom metrics calculator for financial model evaluation
    """
    
    def __init__(self):
        """
        Initialize custom metrics calculator
        """
    
    def custom_accuracy(self, y_true: Union[pd.Series, np.ndarray], 
                        y_pred: Union[pd.Series, np.ndarray]) -> float:
        """
        Custom accuracy metric for upward percentage return predictions.

        This metric focuses exclusively on upward predictions (where `y_pred > 0`).
        
        A prediction is considered "successful" or "accurate" if:
        1. The actual return was positive (`y_true > 0`), meaning the investment was profitable.
        2. The prediction was conservative (`y_true <= y_pred`), meaning the actual gain did not exceed the predicted gain.

        Downward or zero-change predictions are ignored in this metric.
        
        Args:
            y_true: Actual percentage returns (e.g., 0.05 for 5% gain)
            y_pred: Predicted percentage returns (e.g., 0.03 for 3% gain)
            y_previous: Not used for percentage returns (kept for compatibility)
            
        Returns:
            Custom accuracy score (0 to 1) calculated only on upward predictions.
            Returns 0.0 if no upward predictions were made.
        """
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Ensure y_true and y_pred have same length
        if len(y_true) != len(y_pred):
            logger.warning(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
        
        if len(y_true) == 0:
            logger.warning("Empty arrays provided, returning 0.0")
            return 0.0
        
        # Only consider upward predictions
        upward_mask = y_pred > 0
        
        if np.sum(upward_mask) == 0:
            logger.info("No upward predictions made, returning 0.0 accuracy.")
            return 0.0
            
        y_true_upward = y_true[upward_mask]
        y_pred_upward = y_pred[upward_mask]

        successful_predictions = (y_true_upward > 0) & (y_true_upward <= y_pred_upward)
        
        accuracy = np.mean(successful_predictions)
        
        logger.info(f"Custom upward accuracy: {accuracy:.4f} ({np.sum(successful_predictions)}/{len(y_true_upward)})")
        return accuracy 