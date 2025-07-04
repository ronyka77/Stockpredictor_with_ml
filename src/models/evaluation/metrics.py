"""
Custom Metrics for Financial Model Evaluation

This module provides specialized metrics for evaluating stock prediction models,
including custom accuracy metrics that reward conservative predictions and
traditional ML metrics adapted for financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List, Tuple
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

class CustomMetrics:
    """
    Custom metrics calculator for financial model evaluation
    """
    
    def __init__(self, tolerance: float = 0.10):
        """
        Initialize custom metrics calculator
        
        Args:
            tolerance: Tolerance for custom accuracy metric (default 10%)
        """
        self.tolerance = tolerance
        logger.info(f"CustomMetrics initialized with {tolerance*100}% tolerance")
    
    def custom_accuracy(self, y_true: Union[pd.Series, np.ndarray], 
                        y_pred: Union[pd.Series, np.ndarray],
                        y_previous: Union[pd.Series, np.ndarray, None] = None) -> float:
        """
        Custom accuracy metric for financial predictions
        
        Business Formula: (abs(P_actual - P_predicted) / P_actual <= tolerance) OR (correct directional prediction)
        
        This metric rewards:
        1. Predictions within tolerance (10% by default)
        2. Correct directional predictions (both predicted and actual move in same direction from previous price)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_previous: Previous values (for directional accuracy calculation)
            
        Returns:
            Custom accuracy score (0 to 1)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Calculate relative error
        relative_error = np.abs(y_true_filtered - y_pred_filtered) / np.abs(y_true_filtered)
        
        # Condition 1: Within tolerance
        within_tolerance = relative_error <= self.tolerance
        
        # Condition 2: Correct directional prediction
        if y_previous is not None:
            y_previous = np.asarray(y_previous)[mask]
            if len(y_previous) == len(y_true_filtered):
                # Calculate directions from previous price
                actual_direction = y_true_filtered > y_previous
                predicted_direction = y_pred_filtered > y_previous
                correct_direction = actual_direction == predicted_direction
            else:
                logger.warning("y_previous length mismatch, using conservative prediction fallback")
                correct_direction = y_pred_filtered < y_true_filtered  # Conservative fallback
        else:
            # Fallback: Conservative prediction when y_previous not available
            logger.debug("y_previous not provided, using conservative prediction fallback")
            correct_direction = y_pred_filtered < y_true_filtered
        
        # Custom accuracy: either within tolerance OR correct direction
        accurate = within_tolerance | correct_direction
        
        accuracy = np.mean(accurate)
        
        logger.debug(f"Custom accuracy: {accuracy:.4f} (tolerance: {self.tolerance})")
        return accuracy
    
    def directional_accuracy(self, y_true: Union[pd.Series, np.ndarray], 
                            y_pred: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate directional accuracy (predicting correct direction of change)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy score (0 to 1)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_true) < 2:
            return 0.0
        
        # Calculate direction of change
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct_direction = true_direction == pred_direction
        accuracy = np.mean(correct_direction)
        
        return accuracy
    
    def hit_rate(self, y_true: Union[pd.Series, np.ndarray], 
                    y_pred: Union[pd.Series, np.ndarray],
                    threshold: float = 0.05) -> float:
        """
        Calculate hit rate (percentage of predictions within threshold)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            threshold: Threshold for hit rate calculation
            
        Returns:
            Hit rate score (0 to 1)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        
        relative_error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
        hit_rate = np.mean(relative_error <= threshold)
        
        return hit_rate
    
    def volatility_adjusted_accuracy(self, y_true: Union[pd.Series, np.ndarray], 
                                    y_pred: Union[pd.Series, np.ndarray],
                                    window: int = 20) -> float:
        """
        Calculate volatility-adjusted accuracy
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            window: Window for volatility calculation
            
        Returns:
            Volatility-adjusted accuracy score
        """
        y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
        y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
        
        # Calculate returns
        true_returns = y_true.pct_change().dropna()
        pred_returns = y_pred.pct_change().dropna()
        
        if len(true_returns) < window:
            return 0.0
        
        # Calculate rolling volatility
        rolling_vol = true_returns.rolling(window=window).std()
        
        # Calculate errors
        errors = np.abs(true_returns - pred_returns)
        
        # Adjust errors by volatility
        adjusted_errors = errors / (rolling_vol + 1e-8)  # Add small constant to avoid division by zero
        
        # Calculate accuracy (lower adjusted error is better)
        accuracy = 1.0 / (1.0 + adjusted_errors.mean())
        
        return accuracy
    
    def directional_precision_recall(self, y_true: Union[pd.Series, np.ndarray], 
                                    y_pred: Union[pd.Series, np.ndarray],
                                    y_previous: Union[pd.Series, np.ndarray, None] = None) -> Dict[str, float]:
        """
        Calculate precision and recall for directional predictions (up/down movements)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_previous: Previous values (for directional calculation)
            
        Returns:
            Dictionary with precision and recall metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_previous is None:
            # Use previous values from the series itself
            if len(y_true) < 2:
                return {'directional_precision': 0.0, 'directional_recall': 0.0, 'directional_f1': 0.0}
            
            # Calculate direction changes
            true_direction = np.diff(y_true) > 0  # True for upward movements
            pred_direction = np.diff(y_pred) > 0  # True for predicted upward movements
        else:
            y_previous = np.asarray(y_previous)
            if len(y_previous) != len(y_true):
                logger.warning("y_previous length mismatch, using fallback calculation")
                return self.directional_precision_recall(y_true, y_pred, None)
            
            # Calculate directions from previous prices
            true_direction = y_true > y_previous  # True for upward movements
            pred_direction = y_pred > y_previous  # True for predicted upward movements
        
        # Calculate precision, recall, and F1 for upward movements
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(true_direction, pred_direction, zero_division=0.0)
            recall = recall_score(true_direction, pred_direction, zero_division=0.0)
            f1 = f1_score(true_direction, pred_direction, zero_division=0.0)
            
            return {
                'directional_precision': precision,
                'directional_recall': recall,
                'directional_f1': f1
            }
            
        except Exception as e:
            logger.warning(f"Error calculating directional precision/recall: {e}")
            return {'directional_precision': 0.0, 'directional_recall': 0.0, 'directional_f1': 0.0}
    
    def threshold_precision_recall(self, y_true: Union[pd.Series, np.ndarray], 
                                 y_pred: Union[pd.Series, np.ndarray],
                                 threshold: float = 0.05) -> Dict[str, float]:
        """
        Calculate precision and recall for threshold-based predictions
        (whether prediction is within acceptable error threshold)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            threshold: Relative error threshold for "correct" predictions
            
        Returns:
            Dictionary with threshold-based precision and recall metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return {'threshold_precision': 0.0, 'threshold_recall': 0.0, 'threshold_f1': 0.0}
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Calculate relative error
        relative_error = np.abs(y_true_filtered - y_pred_filtered) / np.abs(y_true_filtered)
        
        # Binary classification: within threshold (1) or not (0)
        actual_within_threshold = np.ones(len(y_true_filtered))  # All actual values are "correct"
        predicted_within_threshold = (relative_error <= threshold).astype(int)
        
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(actual_within_threshold, predicted_within_threshold, zero_division=0.0)
            recall = recall_score(actual_within_threshold, predicted_within_threshold, zero_division=0.0)
            f1 = f1_score(actual_within_threshold, predicted_within_threshold, zero_division=0.0)
            
            return {
                'threshold_precision': precision,
                'threshold_recall': recall,
                'threshold_f1': f1
            }
            
        except Exception as e:
            logger.warning(f"Error calculating threshold precision/recall: {e}")
            return {'threshold_precision': 0.0, 'threshold_recall': 0.0, 'threshold_f1': 0.0}
    
    def price_range_precision_recall(self, y_true: Union[pd.Series, np.ndarray], 
                                   y_pred: Union[pd.Series, np.ndarray],
                                   price_ranges: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Calculate precision and recall for price range predictions
        (predicting which price range the actual value falls into)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            price_ranges: List of (min, max) tuples defining price ranges
                         If None, will create ranges based on quartiles
            
        Returns:
            Dictionary with price range precision and recall metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if price_ranges is None:
            # Create quartile-based ranges
            quartiles = np.percentile(y_true, [25, 50, 75])
            price_ranges = [
                (np.min(y_true), quartiles[0]),     # Q1
                (quartiles[0], quartiles[1]),       # Q2
                (quartiles[1], quartiles[2]),       # Q3
                (quartiles[2], np.max(y_true))      # Q4
            ]
        
        def assign_range(values, ranges):
            """Assign each value to a price range"""
            range_assignments = np.zeros(len(values), dtype=int)
            for i, value in enumerate(values):
                for j, (min_val, max_val) in enumerate(ranges):
                    if min_val <= value <= max_val:
                        range_assignments[i] = j
                        break
                else:
                    # If no range matches, assign to closest range
                    distances = [min(abs(value - min_val), abs(value - max_val)) 
                               for min_val, max_val in ranges]
                    range_assignments[i] = np.argmin(distances)
            return range_assignments
        
        try:
            # Assign actual and predicted values to ranges
            true_ranges = assign_range(y_true, price_ranges)
            pred_ranges = assign_range(y_pred, price_ranges)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Calculate macro-averaged precision and recall across all ranges
            precision = precision_score(true_ranges, pred_ranges, average='macro', zero_division=0.0)
            recall = recall_score(true_ranges, pred_ranges, average='macro', zero_division=0.0)
            f1 = f1_score(true_ranges, pred_ranges, average='macro', zero_division=0.0)
            
            return {
                'price_range_precision': precision,
                'price_range_recall': recall,
                'price_range_f1': f1
            }
            
        except Exception as e:
            logger.warning(f"Error calculating price range precision/recall: {e}")
            return {'price_range_precision': 0.0, 'price_range_recall': 0.0, 'price_range_f1': 0.0}
    
    def calculate_regression_metrics(self, y_true: Union[pd.Series, np.ndarray], 
                                    y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Calculate standard regression metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        metrics = {}
        
        try:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            
            # MAPE with handling for zero values
            mask = y_true != 0
            if np.any(mask):
                metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
            else:
                metrics['mape'] = np.inf
            
            # Additional financial metrics
            metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            metrics['mean_error'] = np.mean(y_true - y_pred)  # Bias
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            # Return default metrics
            metrics = {
                'mae': np.inf, 'mse': np.inf, 'rmse': np.inf,
                'r2': -np.inf, 'explained_variance': -np.inf,
                'mape': np.inf, 'max_error': np.inf, 'mean_error': 0.0
            }
        
        return metrics
    
    def calculate_all_metrics(self, y_true: Union[pd.Series, np.ndarray], 
                            y_pred: Union[pd.Series, np.ndarray],
                            y_previous: Union[pd.Series, np.ndarray, None] = None) -> Dict[str, float]:
        """
        Calculate comprehensive set of metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_previous: Previous values (for directional metrics)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Custom financial metrics
        metrics['custom_accuracy'] = self.custom_accuracy(y_true, y_pred, y_previous)
        metrics['directional_accuracy'] = self.directional_accuracy(y_true, y_pred)
        metrics['hit_rate_5pct'] = self.hit_rate(y_true, y_pred, threshold=0.05)
        metrics['hit_rate_10pct'] = self.hit_rate(y_true, y_pred, threshold=0.10)
        metrics['volatility_adjusted_accuracy'] = self.volatility_adjusted_accuracy(y_true, y_pred)
        
        # Precision and recall metrics
        directional_pr = self.directional_precision_recall(y_true, y_pred, y_previous)
        metrics.update(directional_pr)
        
        threshold_pr_5pct = self.threshold_precision_recall(y_true, y_pred, threshold=0.05)
        metrics.update({f"{k}_5pct": v for k, v in threshold_pr_5pct.items()})
        
        threshold_pr_10pct = self.threshold_precision_recall(y_true, y_pred, threshold=0.10)
        metrics.update({f"{k}_10pct": v for k, v in threshold_pr_10pct.items()})
        
        price_range_pr = self.price_range_precision_recall(y_true, y_pred)
        metrics.update(price_range_pr)
        
        # Standard regression metrics
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
        metrics.update(regression_metrics)
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """
        Print formatted metrics summary
        
        Args:
            metrics: Dictionary of metrics to display
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        
        # Financial metrics
        print("\nFinancial Metrics:")
        print(f"  Custom Accuracy:        {metrics.get('custom_accuracy', 0):.4f}")
        print(f"  Directional Accuracy:   {metrics.get('directional_accuracy', 0):.4f}")
        print(f"  Hit Rate (5%):          {metrics.get('hit_rate_5pct', 0):.4f}")
        print(f"  Hit Rate (10%):         {metrics.get('hit_rate_10pct', 0):.4f}")
        print(f"  Vol-Adjusted Accuracy:  {metrics.get('volatility_adjusted_accuracy', 0):.4f}")
        
        # Precision and Recall metrics
        print("\nPrecision & Recall Metrics:")
        print(f"  Directional Precision:  {metrics.get('directional_precision', 0):.4f}")
        print(f"  Directional Recall:     {metrics.get('directional_recall', 0):.4f}")
        print(f"  Directional F1:         {metrics.get('directional_f1', 0):.4f}")
        print(f"  Threshold Precision (5%): {metrics.get('threshold_precision_5pct', 0):.4f}")
        print(f"  Threshold Recall (5%):  {metrics.get('threshold_recall_5pct', 0):.4f}")
        print(f"  Threshold F1 (5%):      {metrics.get('threshold_f1_5pct', 0):.4f}")
        print(f"  Price Range Precision:  {metrics.get('price_range_precision', 0):.4f}")
        print(f"  Price Range Recall:     {metrics.get('price_range_recall', 0):.4f}")
        print(f"  Price Range F1:         {metrics.get('price_range_f1', 0):.4f}")
        
        # Standard metrics
        print("\nStandard Metrics:")
        print(f"  MAE:                    {metrics.get('mae', 0):.4f}")
        print(f"  RMSE:                   {metrics.get('rmse', 0):.4f}")
        print(f"  R²:                     {metrics.get('r2', 0):.4f}")
        print(f"  MAPE:                   {metrics.get('mape', 0):.4f}")
        print(f"  Mean Error (Bias):      {metrics.get('mean_error', 0):.4f}")
        
        print("="*50) 