"""
Threshold-Based Model Evaluation

This module provides centralized threshold optimization and profit-based evaluation
capabilities that can be used by different gradient boosting models (XGBoost, LightGBM).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Protocol, runtime_checkable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.core.logger import get_logger
from src.models.evaluation.metrics import CustomMetrics

logger = get_logger(__name__)


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for models that support threshold-based evaluation"""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def get_prediction_confidence(
        self, X: pd.DataFrame, method: str = "leaf_depth"
    ) -> np.ndarray: ...


class ThresholdEvaluator:
    """Centralized threshold optimization and profit-based evaluation for gradient boosting models"""

    def __init__(self, investment_amount: float = 100.0):
        self.investment_amount = investment_amount
        self.custom_metrics = CustomMetrics()
        self.min_samples_percentage = 0.002
        self.max_samples_percentage = 0.30

    def _flatten_1d(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is 1D"""
        return arr.flatten() if arr.ndim > 1 else arr

    def _compute_profits(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        investment_amount: float,
    ) -> np.ndarray:
        """Unified profit calculation for percentage returns"""
        y_true, y_pred, current_prices = map(self._flatten_1d, [y_true, y_pred, current_prices])
        invest_mask = y_pred > 0
        profits = np.zeros_like(y_true)
        if invest_mask.sum() > 0:
            shares = investment_amount / current_prices[invest_mask]
            profits[invest_mask] = shares * current_prices[invest_mask] * y_true[invest_mask]
        return profits

    def _calculate_investment_success(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, int]:
        """Calculate investment success rate and count"""
        decisions = y_pred > 0
        if decisions.sum() == 0:
            return 0.0, 0
        successful = ((y_true[decisions] > 0) & (y_true[decisions] <= y_pred[decisions])).sum()
        return successful / decisions.sum(), successful

    def _vectorized_threshold_testing(
        self,
        test_predictions: np.ndarray,
        test_confidence: np.ndarray,
        y_test: np.ndarray,
        current_prices_test: np.ndarray,
        thresholds: np.ndarray,
    ) -> pd.DataFrame:
        """Vectorized threshold testing - process all thresholds simultaneously"""
        threshold_masks = test_confidence[:, np.newaxis] >= thresholds[np.newaxis, :]
        min_samples = max(1, int(self.min_samples_percentage * len(test_confidence)))
        max_samples = int(self.max_samples_percentage * len(test_confidence))

        logger.info(
            f"Sample constraints: min={min_samples}, max={max_samples} (total={len(test_confidence)})"
        )

        results = []
        for i, threshold in enumerate(thresholds):
            mask = threshold_masks[:, i]
            samples_kept = mask.sum()
            if not (min_samples <= samples_kept <= max_samples) or samples_kept == 0:
                continue

            filtered = {
                k: v[mask]
                for k, v in [
                    ("y_true", y_test),
                    ("y_pred", test_predictions),
                    ("prices", current_prices_test),
                    ("conf", test_confidence),
                ]
            }

            profits = self._compute_profits(
                filtered["y_true"], filtered["y_pred"], filtered["prices"], self.investment_amount
            )
            success_rate, successful_count = self._calculate_investment_success(
                filtered["y_true"], filtered["y_pred"]
            )

            results.append(
                {
                    "threshold": threshold,
                    "test_samples_kept": samples_kept,
                    "test_samples_ratio": samples_kept / len(test_confidence),
                    "test_profit": profits.sum(),
                    "test_custom_accuracy": self.custom_metrics.custom_accuracy(
                        filtered["y_true"], filtered["y_pred"]
                    ),
                    "test_profit_per_investment": profits.sum() / samples_kept,
                    "investment_success_rate": success_rate,
                    "profitable_investments": successful_count,
                    "average_confidence": filtered["conf"].mean(),
                }
            )

        results_df = pd.DataFrame(results)
        if len(results_df) == 0:
            logger.warning(
                f"No valid thresholds ({self.min_samples_percentage * 100}%-"
                f"{self.max_samples_percentage * 100}%) | conf range: "
                f"[{test_confidence.min():.4f}, {test_confidence.max():.4f}]"
            )
            return pd.DataFrame()

        return results_df

    def calculate_filtered_profit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        investment_amount: Optional[float] = None,
    ) -> float:
        """Calculate total profit for filtered predictions"""
        if len(y_true) == 0:
            return 0.0
        amt = investment_amount if investment_amount is not None else self.investment_amount
        return self._compute_profits(y_true, y_pred, current_prices, amt).sum()

    def calculate_custom_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate custom accuracy using conservative prediction logic"""
        return 0.0 if len(y_true) == 0 else self.custom_metrics.custom_accuracy(y_true, y_pred)

    def calculate_investment_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive investment decision metrics"""
        if len(y_true) == 0:
            return {
                "investments_made": 0,
                "investment_rate": 0.0,
                "investment_success_rate": 0.0,
                "profitable_investments": 0,
                "total_samples": 0,
            }

        invest_mask = y_pred > 0
        success_rate, successful = self._calculate_investment_success(y_true, y_pred)

        return {
            "investments_made": invest_mask.sum(),
            "investment_rate": invest_mask.sum() / len(y_pred),
            "investment_success_rate": success_rate,
            "profitable_investments": successful,
            "total_samples": len(y_pred),
        }

    def optimize_prediction_threshold(
        self,
        model: ModelProtocol,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        current_prices_test: np.ndarray,
        confidence_method: str = "simple",
        threshold_range: Tuple[float, float] = (0.01, 0.99),
        n_thresholds: int = 90,
    ) -> Dict[str, Any]:
        """Optimize prediction threshold to maximize investment success rate on test data"""
        logger.info(
            f"Threshold optimization: {confidence_method} method, "
            f"{n_thresholds} thresholds [{threshold_range[0]:.2f}-{threshold_range[1]:.2f}]"
        )

        try:
            test_predictions = model.predict(x_test)
            logger.info(
                f"Predictions: shape={test_predictions.shape}, "
                f"range=[{test_predictions.min():.4f}, {test_predictions.max():.4f}]"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"status": "failed", "message": f"Prediction failed: {e}"}

        unique_preds = np.unique(test_predictions)
        min_diversity = len(test_predictions) / 100
        if len(unique_preds) < min_diversity:
            logger.error(
                f"Insufficient prediction diversity: {len(unique_preds)} < {min_diversity:.1f}"
            )
            return {
                "status": "failed",
                "message": f"Insufficient prediction diversity: {len(unique_preds)} unique values",
            }

        try:
            test_confidence = model.get_prediction_confidence(x_test, method=confidence_method)
            logger.info(
                f"Confidence: shape={test_confidence.shape}, "
                f"range=[{test_confidence.min():.4f}, {test_confidence.max():.4f}], "
                f"mean={test_confidence.mean():.4f}, std={test_confidence.std():.4f}"
            )
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return {"status": "failed", "message": f"Confidence calculation failed: {e}"}

        if np.isnan(test_confidence).any() or np.isinf(test_confidence).any():
            logger.error("Confidence scores contain NaN/Inf values")
            return {"status": "failed", "message": "Invalid confidence scores (NaN/Inf)"}

        test_pred_1d = self._flatten_1d(test_predictions)
        invest_mask = test_pred_1d > 0
        logger.info("Investment Decision Analysis:")
        if invest_mask.sum() > 0:
            y_test_1d = self._flatten_1d(y_test.values)
            conservative_correct = (y_test_1d[invest_mask] >= test_pred_1d[invest_mask]).sum()
            logger.info(
                f"Conservative success: {conservative_correct}/{invest_mask.sum()} "
                f"({conservative_correct / invest_mask.sum() * 100:.1f}%)"
            )
        else:
            logger.info("No investment candidates!")

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        results_df = self._vectorized_threshold_testing(
            test_predictions, test_confidence, y_test.values, current_prices_test, thresholds
        )

        if len(results_df) == 0:
            stats = (
                f"conf=[{test_confidence.min():.4f}-{test_confidence.max():.4f}], "
                f"pred=[{test_predictions.min():.4f}-{test_predictions.max():.4f}]"
            )
            logger.warning(f"No valid thresholds | {stats}")
            return {
                "status": "failed",
                "message": f"No valid thresholds ({self.min_samples_percentage * 100}%-{self.max_samples_percentage * 100}%)",
            }

        best_result = results_df.loc[results_df["test_profit_per_investment"].idxmax()]
        logger.info(
            f"Best threshold: {best_result['threshold']:.3f} | "
            f"samples: {best_result['test_samples_kept']}/{len(test_confidence)} "
            f"({best_result['test_samples_ratio']:.1%}) | "
            f"success_rate: {best_result['investment_success_rate']:.3f} | "
            f"profit_per_inv: ${best_result['test_profit_per_investment']:.2f} | "
            f"accuracy: {best_result['test_custom_accuracy']:.3f}"
        )

        return {
            "status": "success",
            "optimal_threshold": best_result["threshold"],
            "confidence_method": confidence_method,
            "best_result": best_result.to_dict(),
            "all_results": results_df,
            "threshold_range": threshold_range,
            "n_thresholds_tested": len(results_df),
            "total_test_samples": len(test_confidence),
        }

    def predict_with_threshold(
        self,
        model: ModelProtocol,
        X: pd.DataFrame,
        threshold: float,
        confidence_method: str = "leaf_depth",
        return_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Make predictions with confidence-based filtering"""
        if not isinstance(model, ModelProtocol):
            raise ValueError("Model must implement ModelProtocol")

        all_predictions = model.predict(X)
        all_confidence = model.get_prediction_confidence(X, method=confidence_method)

        from src.models.evaluation.threshold_policy import (
            ThresholdPolicy,
            ThresholdConfig,
            ThresholdResult,
        )
        from typing import cast

        policy = ThresholdPolicy()
        policy_result = policy.compute_mask(
            all_confidence, X, ThresholdConfig(method="ge", value=float(threshold))
        )
        if policy_result is None:
            raise ValueError("ThresholdPolicy.compute_mask returned None")
        policy_result = cast(ThresholdResult, policy_result)

        filtered_preds = all_predictions[policy_result.mask]
        filtered_conf = all_confidence[policy_result.mask]

        result = {
            "all_predictions": all_predictions,
            "filtered_predictions": filtered_preds,
            "filtered_indices": policy_result.indices,
            "confidence_threshold": threshold,
            "confidence_method": confidence_method,
            "total_samples": len(X),
            "filtered_samples": len(filtered_preds),
            "samples_kept_ratio": len(filtered_preds) / len(X),
            "high_confidence_mask": policy_result.mask,
        }

        if return_confidence:
            result.update({"all_confidence": all_confidence, "filtered_confidence": filtered_conf})

        logger.info(
            f"Threshold filtering: {threshold:.3f} | kept: {len(filtered_preds)}/{len(X)} "
            f"({result['samples_kept_ratio']:.1%}) | avg_conf: {filtered_conf.mean():.3f}"
        )

        return result

    def evaluate_threshold_performance(
        self,
        model: ModelProtocol,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        current_prices_test: np.ndarray,
        threshold: float,
        confidence_method: str = "leaf_depth",
    ) -> Dict[str, Any]:
        """Evaluate model performance with threshold filtering"""
        pred_result = self.predict_with_threshold(
            model, x_test, threshold, confidence_method, return_confidence=True
        )

        filtered_idx = pred_result["filtered_indices"]
        if len(filtered_idx) == 0:
            logger.warning("No predictions passed threshold")
            return {"status": "failed", "message": "No predictions passed threshold"}

        filtered_actual = y_test.iloc[filtered_idx].values
        filtered_preds = pred_result["filtered_predictions"]
        filtered_prices = current_prices_test[filtered_idx]

        metrics = {
            k: v(filtered_actual, filtered_preds)
            for k, v in [
                ("mse", mean_squared_error),
                ("mae", mean_absolute_error),
                ("r2_score", r2_score),
            ]
        }

        total_profit = self.calculate_filtered_profit(
            filtered_actual, filtered_preds, filtered_prices
        )
        investment_metrics = self.calculate_investment_metrics(filtered_actual, filtered_preds)

        results = {
            "status": "success",
            "threshold_used": pred_result["confidence_threshold"],
            "confidence_method": pred_result["confidence_method"],
            "samples_evaluated": len(filtered_preds),
            "samples_kept_ratio": pred_result["samples_kept_ratio"],
            "average_confidence": pred_result["filtered_confidence"].mean(),
            **metrics,
            "total_profit": total_profit,
            "profit_per_investment": total_profit / len(filtered_preds),
            "custom_accuracy": self.calculate_custom_accuracy(filtered_actual, filtered_preds),
            **investment_metrics,
            "total_test_samples": len(x_test),
            "filtered_samples": len(filtered_preds),
            "avg_confidence_all": pred_result["all_confidence"].mean(),
            "avg_confidence_filtered": pred_result["filtered_confidence"].mean(),
        }

        logger.info(
            f"Performance: samples={results['samples_evaluated']}/{results['total_test_samples']} "
            f"({results['samples_kept_ratio']:.1%}) | profit=${results['total_profit']:.2f} "
            f"(${results['profit_per_investment']:.2f}/inv) | accuracy={results['custom_accuracy']:.3f} | "
            f"success_rate={results['investment_success_rate']:.3f} | R2={results['r2_score']:.4f}"
        )

        return results

    def calculate_profit_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        investment_amount: Optional[float] = None,
    ) -> float:
        """Calculate profit score with custom accuracy and precision metrics"""
        amt = investment_amount if investment_amount is not None else self.investment_amount
        custom_acc = self.custom_metrics.custom_accuracy(y_true, y_pred)
        invest_mask = y_pred > 0

        total_preds = len(y_pred)
        positive_preds = invest_mask.sum()

        if positive_preds == 0:
            logger.info(f"No positive predictions | Custom Accuracy: {custom_acc:.3f}")
            return 0.0

        selected = {
            k: v[invest_mask]
            for k, v in [("actual", y_true), ("predicted", y_pred), ("prices", current_prices)]
        }

        conservative_correct = (selected["actual"] >= selected["predicted"]).sum()
        pos_rate = positive_preds / total_preds * 100
        cons_rate = conservative_correct / positive_preds * 100

        logger.info(
            f"Accuracy: {custom_acc:.3f} | Total: {total_preds} | "
            f"Positive: {positive_preds} ({pos_rate:.1f}%) | "
            f"Conservative: {conservative_correct} ({cons_rate:.1f}%)"
        )

        shares = amt / selected["prices"]
        profits = shares * selected["prices"] * selected["actual"]
        profitable_count = (profits > 0).sum()

        logger.info(
            f"Investments: {len(profits)} | Profitable: {profitable_count} "
            f"({profitable_count / len(profits) * 100:.1f}%) | "
            f"Avg profit: ${profits.mean():.2f}"
        )

        return profits.sum()
