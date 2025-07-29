"""
Threshold-Based Model Evaluation

This module provides centralized threshold optimization and profit-based evaluation
capabilities that can be used by different gradient boosting models (XGBoost, LightGBM, CatBoost).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Protocol, runtime_checkable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.logger import get_logger
from src.models.evaluation.metrics import CustomMetrics

logger = get_logger(__name__)

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for models that support threshold-based evaluation"""
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on features"""
        ...
    
    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'leaf_depth') -> np.ndarray:
        """Get confidence scores for predictions"""
        ...

class ThresholdEvaluator:
    """
    Centralized threshold optimization and profit-based evaluation for gradient boosting models
    
    This class provides model-agnostic evaluation capabilities including:
    - Threshold optimization based on confidence scores
    - Profit-based performance evaluation
    - Custom accuracy calculation using conservative prediction logic
    - Investment decision analysis
    """
    
    def __init__(self, investment_amount: float = 100.0):
        """
        Initialize threshold evaluator
        
        Args:
            investment_amount: Default investment amount per stock
        """
        self.investment_amount = investment_amount
        self.custom_metrics = CustomMetrics()
        # logger.info(f"ThresholdEvaluator initialized with ${investment_amount:.2f} investment amount")
    
    def _vectorized_profit_calculation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        current_prices: np.ndarray, 
                                        investment_amount: Optional[float] = None) -> np.ndarray:
        """
        Vectorized profit calculation for percentage returns
        
        Args:
            y_true: Actual percentage returns (e.g., 0.05 for 5% gain)
            y_pred: Predicted percentage returns (e.g., 0.03 for 3% gain)
            current_prices: Current prices
            investment_amount: Investment amount per stock
            
        Returns:
            Array of profits for each sample
        """
        if investment_amount is None:
            investment_amount = self.investment_amount
        
        # Only invest where model predicts positive return
        invest_mask = y_pred > 0
        
        # Calculate profits vectorized
        profits = np.zeros_like(y_true)
        if invest_mask.sum() > 0:
            # Ensure all arrays are 1D
            y_true_1d = y_true.flatten() if y_true.ndim > 1 else y_true
            y_pred_1d = y_pred.flatten() if y_pred.ndim > 1 else y_pred
            current_prices_1d = current_prices.flatten() if current_prices.ndim > 1 else current_prices
            
            # Recreate invest_mask with 1D arrays
            invest_mask_1d = y_pred_1d > 0
            
            if invest_mask_1d.sum() > 0:
                shares_bought = investment_amount / current_prices_1d[invest_mask_1d]
                # For percentage returns: profit = shares * current_price * actual_return
                profits_1d = np.zeros_like(y_true_1d)
                profits_1d[invest_mask_1d] = shares_bought * current_prices_1d[invest_mask_1d] * y_true_1d[invest_mask_1d]
                
                # Map back to original shape
                if y_true.ndim > 1:
                    profits = profits_1d.reshape(y_true.shape)
                else:
                    profits = profits_1d
        
        return profits
    
    def _vectorized_threshold_testing(self, test_predictions: np.ndarray, 
                                    test_confidence: np.ndarray,
                                    y_test: np.ndarray, 
                                    current_prices_test: np.ndarray,
                                    thresholds: np.ndarray) -> pd.DataFrame:
        """
        Vectorized threshold testing - process all thresholds simultaneously
        
        Args:
            test_predictions: Model predictions
            test_confidence: Confidence scores
            y_test: Test targets
            current_prices_test: Current prices
            thresholds: Array of thresholds to test
            
        Returns:
            DataFrame with results for all thresholds
        """
        
        # Create threshold masks for all thresholds at once using broadcasting
        threshold_masks = test_confidence[:, np.newaxis] >= thresholds[np.newaxis, :]
        
        results = []

        # Relaxed sample constraints: 0.05% to 5% of the dataset (more suitable for LSTM models)
        min_samples = max(1, int(0.0005 * len(test_confidence)))  # 0.05% minimum
        max_samples = int(0.05 * len(test_confidence))  # 5% maximum
        
        logger.debug(f"Sample constraints: min={min_samples}, max={max_samples} (total samples={len(test_confidence)})")
        
        for i, threshold in enumerate(thresholds):
            mask = threshold_masks[:, i]
            samples_kept = mask.sum()
            
            if not (min_samples <= samples_kept <= max_samples):
                continue
            
            filtered_y_true = y_test[mask]
            filtered_y_pred = test_predictions[mask]
            filtered_current_prices = current_prices_test[mask]
            filtered_confidence = test_confidence[mask]
            
            # Ensure we have valid data
            if len(filtered_y_true) == 0:
                continue
                
            filtered_profits = self._vectorized_profit_calculation(
                filtered_y_true, filtered_y_pred, filtered_current_prices
            )
            
            # Calculate metrics
            total_profit = filtered_profits.sum()
            profit_per_investment = total_profit / samples_kept if samples_kept > 0 else 0
            
            custom_accuracy = self.custom_metrics.custom_accuracy(
                filtered_y_true, filtered_y_pred
            )
            
            investment_decisions = filtered_y_pred > 0
            if investment_decisions.sum() > 0:
                successful_investments = ((filtered_y_true[investment_decisions] > 0) & 
                                        (filtered_y_true[investment_decisions] <= filtered_y_pred[investment_decisions])).sum()
                investment_success_rate = successful_investments / investment_decisions.sum()
                profitable_investments = successful_investments
            else:
                investment_success_rate = 0.0
                profitable_investments = 0
            
            result_dict = {
                'threshold': threshold,
                'test_samples_kept': samples_kept,
                'test_samples_ratio': samples_kept / len(test_confidence),
                'test_profit': total_profit,
                'test_custom_accuracy': custom_accuracy,
                'test_profit_per_investment': profit_per_investment,
                'investment_success_rate': investment_success_rate,
                'profitable_investments': profitable_investments,
                'average_confidence': filtered_confidence.mean()
            }
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("⚠️ No valid thresholds found that satisfy the sample constraints (0.05% to 5%)")
            logger.warning(f"   Confidence range: [{test_confidence.min():.4f}, {test_confidence.max():.4f}]")
            logger.warning(f"   Total samples: {len(test_confidence)}")
            return pd.DataFrame()  # Return empty DataFrame instead of dict
        
        # Find best threshold based on profit per investment
        best_idx = results_df['test_profit_per_investment'].idxmax()
        best_result = results_df.loc[best_idx]
        
        logger.info("🎯 Threshold Optimization Results (Test Data Only - Optimized for Profit Per Investment):")
        logger.info(f"   Best threshold: {best_result['threshold']:.3f}")
        logger.info(f"   Test samples kept: {best_result['test_samples_kept']}/{len(test_confidence)} ({best_result['test_samples_ratio']:.1%})")
        logger.info(f"   Investment success rate: {best_result['investment_success_rate']:.3f}")
        logger.info(f"   Test profit per investment: ${best_result['test_profit_per_investment']:.2f}")
        logger.info(f"   Test custom accuracy: {best_result['test_custom_accuracy']:.3f}")
        logger.info(f"   Total test profit: ${best_result['test_profit']:.2f}")
        
        return results_df
    
    def calculate_filtered_profit(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    current_prices: np.ndarray, 
                                    investment_amount: Optional[float] = None) -> float:
        """
        Calculate profit for filtered predictions (percentage returns)
        
        Args:
            y_true: Actual percentage returns (e.g., 0.05 for 5% gain)
            y_pred: Predicted percentage returns (e.g., 0.03 for 3% gain)
            current_prices: Current prices
            investment_amount: Investment amount per stock (uses default if None)
            
        Returns:
            Total profit from filtered predictions
        """
        if investment_amount is None:
            investment_amount = self.investment_amount
            
        if len(y_true) == 0:
            return 0.0
        
        # Only invest where model predicts positive return
        invest_mask = y_pred > 0
        
        if invest_mask.sum() == 0:
            return 0.0
        
        # Calculate profits for selected investments
        selected_actual = y_true[invest_mask]
        selected_current = current_prices[invest_mask]
        
        # Calculate shares bought at current market price
        shares_bought = investment_amount / selected_current
        
        # For percentage returns: profit = shares * current_price * actual_return
        profits = shares_bought * selected_current * selected_actual
        
        return profits.sum()
    
    def calculate_custom_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate custom accuracy for filtered predictions using conservative prediction logic
        
        Args:
            y_true: Actual future prices
            y_pred: Predicted future prices
            current_prices: Current prices
            
        Returns:
            Custom accuracy score (0 to 1)
        """
        if len(y_true) == 0:
            return 0.0
        
        # Custom accuracy using conservative prediction logic
        return self.custom_metrics.custom_accuracy(y_true, y_pred)
    
    def calculate_investment_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive investment decision metrics
        
        Args:
            y_true: Actual future prices
            y_pred: Predicted future prices
            current_prices: Current prices
            
        Returns:
            Dictionary with investment decision metrics
        """
        if len(y_true) == 0:
            return {
                'investments_made': 0,
                'investment_rate': 0.0,
                'investment_success_rate': 0.0,
                'profitable_investments': 0,
                'total_samples': 0
            }
        
        # Investment decisions: only invest where model predicts positive return
        invest_mask = y_pred > 0
        investment_decisions = y_pred > 0
            
        investments_made = invest_mask.sum()
        investment_rate = investments_made / len(y_pred)
        
        # Investment success rate (for investment decision analysis)
        if investment_decisions.sum() > 0:
            # Success = actual future price > current price (profitable investment)
            successful_investments = ((y_true[investment_decisions] > 0) & 
                                        (y_true[investment_decisions] <= y_pred[investment_decisions])).sum()
            investment_success_rate = successful_investments / investment_decisions.sum()
            profitable_investments = successful_investments
        else:
            investment_success_rate = 0.0
            profitable_investments = 0
        
        return {
            'investments_made': investments_made,
            'investment_rate': investment_rate,
            'investment_success_rate': investment_success_rate,
            'profitable_investments': profitable_investments,
            'total_samples': len(y_pred)
        }
    
    def optimize_prediction_threshold(self, model: ModelProtocol,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    current_prices_test: np.ndarray,
                                    confidence_method: str = 'simple',
                                    threshold_range: Tuple[float, float] = (0.1, 0.9),
                                    n_thresholds: int = 80) -> Dict[str, Any]:
        """
        Optimize prediction threshold based on confidence scores to maximize investment success rate on test data
        
        Args:
            model: Model instance with predict and get_prediction_confidence methods
            X_test: Test features (unseen data)
            y_test: Test targets
            current_prices_test: Current prices for test set
            confidence_method: Method for calculating confidence scores
            threshold_range: Range of thresholds to test (min, max)
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with optimization results based on test data only (optimized for investment success rate)
        """
        logger.info(f"🎯 Starting threshold optimization on test data using {confidence_method} confidence method")
        logger.info(f"   Testing {n_thresholds} thresholds from {threshold_range[0]:.2f} to {threshold_range[1]:.2f}")
        
        # Validate model protocol
        if not isinstance(model, ModelProtocol):
            raise ValueError("Model must implement ModelProtocol (predict and get_prediction_confidence methods)")
        
        # Phase 1: Get predictions
        try:
            test_predictions = model.predict(X_test)
            logger.debug(f"Predictions shape: {test_predictions.shape}")
            logger.debug(f"Predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
        except Exception as e:
            logger.error(f"❌ Failed to get predictions: {e}")
            return {'status': 'failed', 'message': f'Prediction failed: {e}'}
        
        # Validate prediction diversity
        unique_predictions = np.unique(test_predictions)
        min_required_diversity = len(test_predictions) / 100
        
        if len(unique_predictions) < min_required_diversity:
            logger.error("🚨 CRITICAL: Model produced insufficient prediction diversity.")
            logger.error(f"   Unique predictions: {len(unique_predictions)}")
            logger.error(f"   Minimum required: {min_required_diversity:.1f}")
            logger.error("   This indicates a failed training process, likely due to constant features.")
            logger.error("   Aborting threshold optimization.")
            return {'status': 'failed', 'message': f'Model produced insufficient prediction diversity: {len(unique_predictions)} unique values (minimum required: {min_required_diversity:.1f}).'}
        
        # Phase 2: Get confidence scores
        try:
            test_confidence = model.get_prediction_confidence(X_test, method=confidence_method)
            logger.debug(f"Confidence shape: {test_confidence.shape}")
            logger.debug(f"Confidence range: [{test_confidence.min():.4f}, {test_confidence.max():.4f}]")
            logger.debug(f"Confidence mean: {test_confidence.mean():.4f}, std: {test_confidence.std():.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to get confidence scores: {e}")
            return {'status': 'failed', 'message': f'Confidence calculation failed: {e}'}
        
        # Validate confidence scores
        if np.isnan(test_confidence).any():
            logger.error("❌ Confidence scores contain NaN values")
            return {'status': 'failed', 'message': 'Confidence scores contain NaN values'}
        
        if np.isinf(test_confidence).any():
            logger.error("❌ Confidence scores contain infinite values")
            return {'status': 'failed', 'message': 'Confidence scores contain infinite values'}
        
        # For percentage returns: invest when predicted return > 0
        # Ensure test_predictions is 1D
        test_predictions_1d = test_predictions.flatten() if test_predictions.ndim > 1 else test_predictions
        invest_mask = test_predictions_1d > 0
        logger.info("🔍 DIAGNOSTIC - Investment Decision Analysis:")
        logger.info(f"   Total investment candidates: {invest_mask.sum()}/{len(invest_mask)} ({invest_mask.sum()/len(invest_mask)*100:.1f}%)")        
        if invest_mask.sum() > 0:
            # Ensure y_test.values and test_predictions are 1D
            y_test_1d = y_test.values.flatten() if y_test.values.ndim > 1 else y_test.values
            test_predictions_1d = test_predictions.flatten() if test_predictions.ndim > 1 else test_predictions
            actual_vs_predicted = y_test_1d[invest_mask] >= test_predictions_1d[invest_mask]
            logger.info(f"   Conservative success rate (all samples): {actual_vs_predicted.sum()}/{invest_mask.sum()} ({actual_vs_predicted.sum()/invest_mask.sum()*100:.1f}%)")
        else:
            logger.info("   No investment candidates found!")
        
        # Phase 3: Vectorized threshold testing
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        # Use vectorized threshold testing
        results_df = self._vectorized_threshold_testing(
            test_predictions, test_confidence, y_test.values, current_prices_test, thresholds
        )
        
        # Phase 4: Analyze results
        if len(results_df) == 0:
            logger.warning("⚠️ No valid thresholds found that satisfy the sample constraints (0.05% to 5%)")
            logger.warning(f"   Confidence range: [{test_confidence.min():.4f}, {test_confidence.max():.4f}]")
            logger.warning(f"   Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
            return {'status': 'failed', 'message': 'No valid thresholds found under the 0.05%-5% sample constraint'}
        
        # Find best threshold based on profit per investment
        best_idx = results_df['test_profit_per_investment'].idxmax()
        best_result = results_df.loc[best_idx]
        
        logger.info("🎯 Threshold Optimization Results (Test Data Only - Optimized for Profit Per Investment):")
        logger.info(f"   Best threshold: {best_result['threshold']:.3f}")
        logger.info(f"   Test samples kept: {best_result['test_samples_kept']}/{len(test_confidence)} ({best_result['test_samples_ratio']:.1%})")
        logger.info(f"   Investment success rate: {best_result['investment_success_rate']:.3f}")
        logger.info(f"   Test profit per investment: ${best_result['test_profit_per_investment']:.2f}")
        logger.info(f"   Test custom accuracy: {best_result['test_custom_accuracy']:.3f}")
        logger.info(f"   Total test profit: ${best_result['test_profit']:.2f}")
        logger.info(f"   Average confidence of selected predictions: {best_result['average_confidence']:.3f}")
        
        return {
            'status': 'success',
            'optimal_threshold': best_result['threshold'],
            'confidence_method': confidence_method,
            'best_result': best_result.to_dict(),
            'all_results': results_df,
            'threshold_range': threshold_range,
            'n_thresholds_tested': len(results_df),
            'total_test_samples': len(test_confidence)
        }
    
    def optimize_prediction_threshold_lstm(self, model: ModelProtocol,
                                        X_test: pd.DataFrame, y_test: pd.Series,
                                        current_prices_test: np.ndarray,
                                        confidence_method: str = 'simple',
                                        threshold_range: Tuple[float, float] = (0.1, 0.9),
                                        n_thresholds: int = 80) -> Dict[str, Any]:
        """
        Optimize prediction threshold specifically for LSTM models
        
        Args:
            model: LSTM model instance
            X_test: Test features
            y_test: Test targets
            current_prices_test: Current prices for test set
            confidence_method: Confidence calculation method (recommended: 'simple', 'margin', 'leaf_depth')
            threshold_range: Range of thresholds to test
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"🧠 LSTM-Specific Threshold Optimization using {confidence_method} confidence method")
        
        # For LSTM models, use more conservative confidence methods that don't require dropout
        if confidence_method in ['variance', 'lstm_hidden']:
            logger.warning(f"⚠️ Confidence method '{confidence_method}' requires dropout activation for LSTM models")
            logger.info("   Consider using 'simple', 'margin', or 'leaf_depth' for more stable results")
        
        # Use wider threshold range for LSTM models
        if threshold_range == (0.1, 0.9):
            threshold_range = (0.05, 0.95)  # Wider range for LSTM
            logger.info(f"   Using LSTM-optimized threshold range: {threshold_range}")
        
        # Call the main optimization method with LSTM-specific parameters
        return self.optimize_prediction_threshold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            current_prices_test=current_prices_test,
            confidence_method=confidence_method,
            threshold_range=threshold_range,
            n_thresholds=n_thresholds
        )
    
    def predict_with_threshold(self, model: ModelProtocol, X: pd.DataFrame, 
                                threshold: float, confidence_method: str = 'leaf_depth',
                                return_confidence: bool = False) -> Dict[str, Any]:
        """
        Make predictions with confidence-based filtering
        
        Args:
            model: Model instance with predict and get_prediction_confidence methods
            X: Feature matrix
            threshold: Confidence threshold
            confidence_method: Confidence calculation method
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions, confidence scores, and filtering info
        """
        # Validate model protocol
        if not isinstance(model, ModelProtocol):
            raise ValueError("Model must implement ModelProtocol (predict and get_prediction_confidence methods)")
        
        # Get all predictions and confidence scores
        all_predictions = model.predict(X)
        all_confidence = model.get_prediction_confidence(X, method=confidence_method)
        
        # Apply confidence threshold
        high_confidence_mask = all_confidence >= threshold
        
        # Filter predictions
        filtered_predictions = all_predictions[high_confidence_mask]
        filtered_confidence = all_confidence[high_confidence_mask]
        filtered_indices = np.where(high_confidence_mask)[0]
        
        result = {
            'all_predictions': all_predictions,
            'filtered_predictions': filtered_predictions,
            'filtered_indices': filtered_indices,
            'confidence_threshold': threshold,
            'confidence_method': confidence_method,
            'total_samples': len(X),
            'filtered_samples': len(filtered_predictions),
            'samples_kept_ratio': len(filtered_predictions) / len(X),
            'high_confidence_mask': high_confidence_mask
        }
        
        if return_confidence:
            result['all_confidence'] = all_confidence
            result['filtered_confidence'] = filtered_confidence
        
        logger.info("🎯 Threshold Filtering Results:")
        logger.info(f"   Confidence threshold: {threshold:.3f}")
        logger.info(f"   Samples kept: {len(filtered_predictions)}/{len(X)} ({result['samples_kept_ratio']:.1%})")
        logger.info(f"   Average confidence of kept samples: {filtered_confidence.mean():.3f}")
        
        return result
    
    def evaluate_threshold_performance(self, model: ModelProtocol,
                                        X_test: pd.DataFrame, y_test: pd.Series,
                                        current_prices_test: np.ndarray,
                                        threshold: float,
                                        confidence_method: str = 'leaf_depth') -> Dict[str, Any]:
        """
        Evaluate model performance with threshold filtering
        
        Args:
            model: Model instance with predict and get_prediction_confidence methods
            X_test: Test features
            y_test: Test targets
            current_prices_test: Current prices for test set
            threshold: Confidence threshold
            confidence_method: Confidence method
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        # Get filtered predictions
        prediction_result = self.predict_with_threshold(
            model, X_test, threshold, confidence_method, return_confidence=True
        )
        
        # Extract filtered data
        filtered_indices = prediction_result['filtered_indices']
        filtered_predictions = prediction_result['filtered_predictions']
        
        if len(filtered_indices) == 0:
            logger.warning("⚠️ No predictions passed the confidence threshold")
            return {'status': 'failed', 'message': 'No predictions passed threshold'}
        
        # Get corresponding actual values and current prices
        filtered_actual = y_test.iloc[filtered_indices].values
        filtered_current_prices = current_prices_test[filtered_indices]
        
        # Calculate comprehensive metrics
        mse = mean_squared_error(filtered_actual, filtered_predictions)
        mae = mean_absolute_error(filtered_actual, filtered_predictions)
        r2 = r2_score(filtered_actual, filtered_predictions)
        
        # Profit-based metrics
        total_profit = self.calculate_filtered_profit(
            filtered_actual, filtered_predictions, filtered_current_prices
        )
        profit_per_investment = total_profit / len(filtered_predictions)
        
        # Custom accuracy calculation
        custom_accuracy = self.calculate_custom_accuracy(
            filtered_actual, filtered_predictions
        )
        
        # Investment decision metrics
        investment_metrics = self.calculate_investment_metrics(
            filtered_actual, filtered_predictions
        )
        
        results = {
            'status': 'success',
            'threshold_used': prediction_result['confidence_threshold'],
            'confidence_method': prediction_result['confidence_method'],
            'samples_evaluated': len(filtered_predictions),
            'samples_kept_ratio': prediction_result['samples_kept_ratio'],
            'average_confidence': prediction_result['filtered_confidence'].mean(),
            
            # Traditional metrics
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            
            # Profit metrics
            'total_profit': total_profit,
            'profit_per_investment': profit_per_investment,
            
            # Custom accuracy
            'custom_accuracy': custom_accuracy,
            
            # Investment metrics
            'investments_made': investment_metrics['investments_made'],
            'investment_rate': investment_metrics['investment_rate'],
            'investment_success_rate': investment_metrics['investment_success_rate'],
            'profitable_investments': investment_metrics['profitable_investments'],
            
            # Detailed breakdown
            'total_test_samples': len(X_test),
            'filtered_samples': len(filtered_predictions),
            'avg_confidence_all': prediction_result['all_confidence'].mean(),
            'avg_confidence_filtered': prediction_result['filtered_confidence'].mean()
        }
        
        logger.info("📊 Threshold Performance Evaluation:")
        logger.info(f"   Samples evaluated: {results['samples_evaluated']}/{results['total_test_samples']} ({results['samples_kept_ratio']:.1%})")
        logger.info(f"   Total profit: ${results['total_profit']:.2f}")
        logger.info(f"   Profit per investment: ${results['profit_per_investment']:.2f}")
        logger.info(f"   Custom accuracy: {results['custom_accuracy']:.3f}")
        logger.info(f"   Investment success rate: {results['investment_success_rate']:.3f}")
        logger.info(f"   Traditional R²: {results['r2_score']:.4f}")
        
        return results
    
    def calculate_profit_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                current_prices: np.ndarray, 
                                investment_amount: Optional[float] = None) -> float:
        """
        Calculate profit score based on percentage return investments with custom accuracy and precision metrics
        
        Args:
            y_true: Actual percentage returns (e.g., 0.05 for 5% gain)
            y_pred: Predicted percentage returns (e.g., 0.03 for 3% gain)
            current_prices: Current stock prices (to calculate shares bought)
            investment_amount: Amount to invest in each stock
            
        Returns:
            Total profit/loss from investing only in stocks where model predicts positive return
        """
        if investment_amount is None:
            investment_amount = self.investment_amount
        
        # Calculate custom accuracy using conservative prediction logic
        custom_accuracy = self.custom_metrics.custom_accuracy(y_true, y_pred)
        
        # Only invest in stocks where model predicts positive return
        invest_mask = y_pred > 0
        
        # Calculate prediction precision metrics
        total_predictions = len(y_pred)
        positive_predictions = invest_mask.sum()
        
        if positive_predictions == 0:
            logger.info("📊 Prediction Metrics: No positive predictions made")
            logger.info(f"📊 Custom Accuracy: {custom_accuracy:.3f} (conservative prediction logic)")
            return 0.0
        
        # Calculate actual profitable investments among positive predictions
        selected_actual_values = y_true[invest_mask]
        selected_predicted_values = y_pred[invest_mask]
        selected_current_prices = current_prices[invest_mask]
        
        # Check which positive predictions were conservative (actual >= predicted)
        conservative_predictions = selected_actual_values >= selected_predicted_values
        correct_conservative_predictions = conservative_predictions.sum()
        
        # Calculate percentage of total predictions that were positive
        positive_prediction_rate = (positive_predictions / total_predictions) * 100
        
        # Calculate percentage of positive predictions that were conservative
        conservative_rate = (correct_conservative_predictions / positive_predictions) * 100 if positive_predictions > 0 else 0.0
        
        # Log accuracy and precision metrics
        logger.info("📊 Prediction Accuracy & Conservative Investment Metrics:")
        logger.info(f"   Custom Accuracy: {custom_accuracy:.3f} (conservative prediction logic)")
        logger.info(f"   Total predictions: {total_predictions}")
        logger.info(f"   Positive predictions: {positive_predictions} ({positive_prediction_rate:.1f}%)")
        logger.info(f"   Conservative predictions (actual >= predicted): {correct_conservative_predictions}")
        logger.info(f"   Conservative success rate: {conservative_rate:.1f}%")
        
        # Calculate number of shares we can buy with investment amount for selected stocks
        shares_bought = investment_amount / selected_current_prices
        
        # Calculate actual profit/loss for each selected stock
        # For percentage returns: profit = shares * current_price * actual_return
        profits = shares_bought * selected_current_prices * selected_actual_values
        
        # Calculate additional profit metrics
        profitable_investments = profits > 0
        total_profitable_count = profitable_investments.sum()
        average_profit_per_investment = profits.mean()
        
        logger.info("📊 Investment Performance Metrics:")
        logger.info(f"   Total investments made: {len(profits)}")
        logger.info(f"   Profitable investments: {total_profitable_count} ({(total_profitable_count/len(profits)*100):.1f}%)")
        logger.info(f"   Average profit per investment: ${average_profit_per_investment:.2f}")
        
        # Return total profit from selected investments
        total_profit = profits.sum()
        return total_profit
