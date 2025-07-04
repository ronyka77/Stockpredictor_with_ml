"""
Threshold-Based Model Evaluation

This module provides centralized threshold optimization and profit-based evaluation
capabilities that can be used by different gradient boosting models (XGBoost, LightGBM, CatBoost).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Protocol, runtime_checkable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from src.utils.logger import get_logger

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
    - Comprehensive metrics calculation
    - Investment decision analysis
    """
    
    def __init__(self, investment_amount: float = 100.0):
        """
        Initialize threshold evaluator
        
        Args:
            investment_amount: Default investment amount per stock
        """
        self.investment_amount = investment_amount
        logger.info(f"ThresholdEvaluator initialized with ${investment_amount:.2f} investment amount")
    
    def calculate_filtered_profit(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 current_prices: np.ndarray, 
                                 investment_amount: Optional[float] = None) -> float:
        """
        Calculate profit for filtered predictions
        
        Args:
            y_true: Actual future prices
            y_pred: Predicted future prices
            current_prices: Current prices
            investment_amount: Investment amount per stock (uses default if None)
            
        Returns:
            Total profit from filtered predictions
        """
        if investment_amount is None:
            investment_amount = self.investment_amount
            
        if len(y_true) == 0:
            return 0.0
        
        # Only invest where model predicts price increase
        invest_mask = y_pred > current_prices
        
        if invest_mask.sum() == 0:
            return 0.0
        
        # Calculate profits for selected investments
        selected_actual = y_true[invest_mask]
        selected_current = current_prices[invest_mask]
        
        # Calculate shares bought and profits
        shares_bought = investment_amount / selected_current
        profits = shares_bought * (selected_actual - selected_current)
        
        return profits.sum()
    
    def calculate_precision_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   current_prices: np.ndarray) -> float:
        """
        Calculate precision for filtered predictions
        
        Args:
            y_true: Actual future prices
            y_pred: Predicted future prices
            current_prices: Current prices
            
        Returns:
            Precision score (correct positive predictions / total positive predictions)
        """
        if len(y_true) == 0:
            return 0.0
        
        # Predictions: invest where model predicts price increase
        predicted_positive = y_pred > current_prices
        
        # Actual: stocks that actually increased in price
        actual_positive = y_true > current_prices
        
        if predicted_positive.sum() == 0:
            return 0.0
        
        # Calculate precision
        correct_positive = (predicted_positive & actual_positive).sum()
        precision = correct_positive / predicted_positive.sum()
        
        return precision
    
    def calculate_investment_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   current_prices: np.ndarray) -> Dict[str, Any]:
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
        
        # Investment decisions
        invest_mask = y_pred > current_prices
        investments_made = invest_mask.sum()
        investment_rate = investments_made / len(y_pred)
        
        # Actual profitability of investments made
        if investments_made > 0:
            actual_profitable = (y_true[invest_mask] > current_prices[invest_mask]).sum()
            investment_success_rate = actual_profitable / investments_made
        else:
            investment_success_rate = 0.0
            actual_profitable = 0
        
        return {
            'investments_made': investments_made,
            'investment_rate': investment_rate,
            'investment_success_rate': investment_success_rate,
            'profitable_investments': actual_profitable,
            'total_samples': len(y_pred)
        }
    
    def optimize_prediction_threshold(self, model: ModelProtocol,
                                    X_train: pd.DataFrame, y_train: pd.Series, 
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    current_prices_train: np.ndarray, 
                                    current_prices_test: np.ndarray,
                                    confidence_method: str = 'leaf_depth',
                                    threshold_range: Tuple[float, float] = (0.1, 0.9),
                                    n_thresholds: int = 20) -> Dict[str, Any]:
        """
        Optimize prediction threshold based on confidence scores to maximize profit
        
        Args:
            model: Model instance with predict and get_prediction_confidence methods
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            current_prices_train: Current prices for training set
            current_prices_test: Current prices for test set
            confidence_method: Method for calculating confidence scores
            threshold_range: Range of thresholds to test (min, max)
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"🎯 Starting threshold optimization using {confidence_method} confidence method")
        
        # Validate model protocol
        if not isinstance(model, ModelProtocol):
            raise ValueError("Model must implement ModelProtocol (predict and get_prediction_confidence methods)")
        
        # Get predictions and confidence scores for both sets
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_confidence = model.get_prediction_confidence(X_train, method=confidence_method)
        test_confidence = model.get_prediction_confidence(X_test, method=confidence_method)
        
        # Test different confidence thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        results = []
        
        for threshold in thresholds:
            # Filter predictions based on confidence threshold
            train_mask = train_confidence >= threshold
            test_mask = test_confidence >= threshold
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            
            # Calculate metrics for filtered predictions
            train_filtered_profit = self.calculate_filtered_profit(
                y_train.values[train_mask], 
                train_predictions[train_mask], 
                current_prices_train[train_mask]
            )
            
            test_filtered_profit = self.calculate_filtered_profit(
                y_test.values[test_mask], 
                test_predictions[test_mask], 
                current_prices_test[test_mask]
            )
            
            # Calculate precision metrics for filtered predictions
            train_precision = self.calculate_precision_metrics(
                y_train.values[train_mask], 
                train_predictions[train_mask], 
                current_prices_train[train_mask]
            )
            
            test_precision = self.calculate_precision_metrics(
                y_test.values[test_mask], 
                test_predictions[test_mask], 
                current_prices_test[test_mask]
            )
            
            results.append({
                'threshold': threshold,
                'train_samples_kept': train_mask.sum(),
                'test_samples_kept': test_mask.sum(),
                'train_samples_ratio': train_mask.sum() / len(train_mask),
                'test_samples_ratio': test_mask.sum() / len(test_mask),
                'train_profit': train_filtered_profit,
                'test_profit': test_filtered_profit,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_profit_per_investment': train_filtered_profit / train_mask.sum() if train_mask.sum() > 0 else 0,
                'test_profit_per_investment': test_filtered_profit / test_mask.sum() if test_mask.sum() > 0 else 0
            })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("⚠️ No valid thresholds found")
            return {'status': 'failed', 'message': 'No valid thresholds found'}
        
        # Find best threshold based on test profit per investment
        best_idx = results_df['test_profit_per_investment'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        logger.info("🎯 Threshold Optimization Results:")
        logger.info(f"   Best threshold: {best_result['threshold']:.3f}")
        logger.info(f"   Test samples kept: {best_result['test_samples_kept']}/{len(test_mask)} ({best_result['test_samples_ratio']:.1%})")
        logger.info(f"   Test profit per investment: ${best_result['test_profit_per_investment']:.2f}")
        logger.info(f"   Test precision: {best_result['test_precision']:.3f}")
        logger.info(f"   Total test profit: ${best_result['test_profit']:.2f}")
        
        return {
            'status': 'success',
            'optimal_threshold': best_result['threshold'],
            'confidence_method': confidence_method,
            'best_result': best_result.to_dict(),
            'all_results': results_df,
            'threshold_range': threshold_range,
            'n_thresholds_tested': len(results_df)
        }
    
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
        # Traditional regression metrics
        mse = mean_squared_error(filtered_actual, filtered_predictions)
        mae = mean_absolute_error(filtered_actual, filtered_predictions)
        r2 = r2_score(filtered_actual, filtered_predictions)
        
        # Profit-based metrics
        total_profit = self.calculate_filtered_profit(
            filtered_actual, filtered_predictions, filtered_current_prices
        )
        profit_per_investment = total_profit / len(filtered_predictions)
        
        # Precision metrics
        precision = self.calculate_precision_metrics(
            filtered_actual, filtered_predictions, filtered_current_prices
        )
        
        # Investment decision metrics
        investment_metrics = self.calculate_investment_metrics(
            filtered_actual, filtered_predictions, filtered_current_prices
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
            'precision': precision,
            
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
        logger.info(f"   Precision: {results['precision']:.3f}")
        logger.info(f"   Investment success rate: {results['investment_success_rate']:.3f}")
        logger.info(f"   Traditional R²: {results['r2_score']:.4f}")
        
        return results
    
    def calculate_profit_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              current_prices: np.ndarray, 
                              investment_amount: Optional[float] = None) -> float:
        """
        Calculate profit score based on actual investment returns with precision metrics
        
        Args:
            y_true: Actual future prices (target values)
            y_pred: Predicted future prices
            current_prices: Current stock prices (to calculate shares bought)
            investment_amount: Amount to invest in each stock
            
        Returns:
            Total profit/loss from investing only in stocks where model predicts price increase
        """
        if investment_amount is None:
            investment_amount = self.investment_amount
            
        # Only invest in stocks where model predicts price will increase
        invest_mask = y_pred > current_prices
        
        # Calculate prediction precision metrics
        total_predictions = len(y_pred)
        positive_predictions = invest_mask.sum()
        
        if positive_predictions == 0:
            logger.info("📊 Precision Metrics: No positive predictions made")
            return 0.0
        
        # Calculate actual profitable investments among positive predictions
        selected_actual_prices = y_true[invest_mask]
        selected_current_prices = current_prices[invest_mask]
        
        # Check which positive predictions were actually profitable
        actual_profitable = selected_actual_prices > selected_current_prices
        correct_positive_predictions = actual_profitable.sum()
        
        # Calculate precision: correct positive predictions / total positive predictions
        precision = correct_positive_predictions / positive_predictions if positive_predictions > 0 else 0.0
        
        # Calculate percentage of total predictions that were positive
        positive_prediction_rate = (positive_predictions / total_predictions) * 100
        
        # Calculate percentage of positive predictions that were profitable
        profitable_rate = (correct_positive_predictions / positive_predictions) * 100 if positive_predictions > 0 else 0.0
        
        # Log precision metrics
        logger.info("📊 Prediction Precision Metrics:")
        logger.info(f"   Total predictions: {total_predictions}")
        logger.info(f"   Positive predictions: {positive_predictions} ({positive_prediction_rate:.1f}%)")
        logger.info(f"   Correct positive predictions: {correct_positive_predictions}")
        logger.info(f"   Precision (profitable/predicted positive): {precision:.3f} ({profitable_rate:.1f}%)")
        
        # Calculate number of shares we can buy with investment amount for selected stocks
        shares_bought = investment_amount / selected_current_prices
        
        # Calculate actual profit/loss for each selected stock
        # Profit = shares * (actual_future_price - current_price)
        profits = shares_bought * (selected_actual_prices - selected_current_prices)
        
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
    
    def compare_threshold_strategies(self, model: ModelProtocol,
                                   X_test: pd.DataFrame, y_test: pd.Series,
                                   current_prices_test: np.ndarray,
                                   thresholds: List[float],
                                   confidence_method: str = 'leaf_depth') -> pd.DataFrame:
        """
        Compare performance across multiple threshold strategies
        
        Args:
            model: Model instance with predict and get_prediction_confidence methods
            X_test: Test features
            y_test: Test targets
            current_prices_test: Current prices for test set
            thresholds: List of thresholds to compare
            confidence_method: Confidence calculation method
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for threshold in thresholds:
            try:
                performance = self.evaluate_threshold_performance(
                    model, X_test, y_test, current_prices_test, threshold, confidence_method
                )
                
                if performance['status'] == 'success':
                    results.append({
                        'threshold': threshold,
                        'samples_kept': performance['samples_evaluated'],
                        'samples_kept_ratio': performance['samples_kept_ratio'],
                        'profit_per_investment': performance['profit_per_investment'],
                        'total_profit': performance['total_profit'],
                        'precision': performance['precision'],
                        'investment_success_rate': performance['investment_success_rate'],
                        'r2_score': performance['r2_score'],
                        'mse': performance['mse'],
                        'mae': performance['mae']
                    })
                else:
                    results.append({
                        'threshold': threshold,
                        'samples_kept': 0,
                        'samples_kept_ratio': 0.0,
                        'profit_per_investment': 0.0,
                        'total_profit': 0.0,
                        'precision': 0.0,
                        'investment_success_rate': 0.0,
                        'r2_score': 0.0,
                        'mse': np.inf,
                        'mae': np.inf
                    })
                    
            except Exception as e:
                logger.warning(f"Error evaluating threshold {threshold}: {e}")
                results.append({
                    'threshold': threshold,
                    'samples_kept': 0,
                    'samples_kept_ratio': 0.0,
                    'profit_per_investment': 0.0,
                    'total_profit': 0.0,
                    'precision': 0.0,
                    'investment_success_rate': 0.0,
                    'r2_score': 0.0,
                    'mse': np.inf,
                    'mae': np.inf
                })
        
        comparison_df = pd.DataFrame(results)
        
        logger.info("📊 Threshold Strategy Comparison:")
        logger.info(f"   Evaluated {len(thresholds)} thresholds")
        logger.info(f"   Best profit per investment: ${comparison_df['profit_per_investment'].max():.2f}")
        logger.info(f"   Best threshold: {comparison_df.loc[comparison_df['profit_per_investment'].idxmax(), 'threshold']:.3f}")
        
        return comparison_df 