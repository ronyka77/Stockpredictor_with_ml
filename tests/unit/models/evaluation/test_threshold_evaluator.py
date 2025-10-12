import numpy as np
import pandas as pd

from src.models.evaluation.threshold_evaluator import ThresholdEvaluator


def test_vectorized_profit_calculation_basic():
    """Compute vectorized profits for basic positive/negative prediction scenarios."""
    te = ThresholdEvaluator(investment_amount=100.0)

    # y_true: actual percentage returns; y_pred: predicted percentage returns
    y_true = np.array([0.1, -0.05, 0.2, 0.0])
    y_pred = np.array([0.05, -0.02, 0.03, 0.0])
    current_prices = np.array([10.0, 20.0, 5.0, 8.0])

    profits = te._vectorized_profit_calculation(y_true, y_pred, current_prices)

    # Only positions 0 and 2 have y_pred > 0
    # position 0: shares = 100/10 = 10 -> profit = 10 * 10 * 0.1 = 10
    # position 2: shares = 100/5 = 20 -> profit = 20 * 5 * 0.2 = 20
    if not np.isclose(profits[0], 10.0):
        raise AssertionError("Profit calculation for position 0 incorrect")
    if profits[1] != 0:
        raise AssertionError("Profit for non-invested position should be 0")
    if not np.isclose(profits[2], 20.0):
        raise AssertionError("Profit calculation for position 2 incorrect")
    if profits[3] != 0:
        raise AssertionError("Profit for non-invested position should be 0")


def test_calculate_filtered_profit_edge_cases():
    """Calculate filtered profit handling no-positive-prediction and single-positive cases."""
    te = ThresholdEvaluator(investment_amount=50.0)

    # No positive predictions -> zero profit
    y_true = np.array([0.1, 0.2])
    y_pred = np.array([-0.01, -0.02])
    prices = np.array([10.0, 20.0])
    if te.calculate_filtered_profit(y_true, y_pred, prices) != 0.0:
        raise AssertionError("Filtered profit should be 0 when no positive predictions")

    # Some positive predictions
    y_pred2 = np.array([0.05, -0.01])
    profit = te.calculate_filtered_profit(y_true, y_pred2, prices)
    # invest in first: shares = 50 / 10 =5 -> profit = 5*10*0.1 =5
    if not np.isclose(profit, 5.0):
        raise AssertionError("Filtered profit calculation incorrect")


class MockModel:
    def __init__(self, preds: np.ndarray, conf: np.ndarray):
        self._preds = preds
        self._conf = conf

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._preds

    def get_prediction_confidence(
        self, X: pd.DataFrame, method: str = "simple"
    ) -> np.ndarray:
        return self._conf


def test_optimize_prediction_threshold_success():
    """Optimize prediction threshold over candidate thresholds and report success and optimal value."""
    te = ThresholdEvaluator(investment_amount=100.0)

    n = 100
    # predictions: first 5 positive (0.05), rest negative
    preds = np.array([0.05] * 5 + [-0.01] * (n - 5))
    # confidence: first 5 high, others low
    conf = np.array([0.95] * 5 + [0.01] * (n - 5))
    # y_test: actual returns - make first 5 profitable
    y_test = pd.Series([0.06] * 5 + [0.0] * (n - 5))
    X_test = pd.DataFrame({"f": range(n)})
    current_prices = np.array([10.0] * n)

    model = MockModel(preds, conf)

    res = te.optimize_prediction_threshold(
        model,
        X_test,
        y_test,
        current_prices,
        confidence_method="simple",
        threshold_range=(0.1, 0.99),
        n_thresholds=10,
    )

    if res.get("status") != "success":
        raise AssertionError("Optimization did not report success")
    if "optimal_threshold" not in res:
        raise AssertionError("optimal_threshold missing from optimization result")
    if res.get("n_thresholds_tested", 0) < 1:
        raise AssertionError("No thresholds tested during optimization")


def test_evaluate_threshold_performance_perfect_predictions():
    """Test threshold performance evaluation with high confidence"""
    te = ThresholdEvaluator(investment_amount=100.0)

    # Create mock model with perfect predictions
    class MockModel:
        def predict(self, X):
            return np.array([0.1, -0.05, 0.2])

        def get_prediction_confidence(self, X, method="leaf_depth"):
            return np.array([0.95, 0.90, 0.98])  # High confidence

    model = MockModel()
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([0.12, -0.03, 0.18])  # Actual values
    current_prices = np.array([10.0, 20.0, 5.0])

    results = te.evaluate_threshold_performance(
        model, X, y, current_prices, threshold=0.8
    )

    # Should return success status
    assert results["status"] == "success"
    assert "total_profit" in results
    assert "custom_accuracy" in results
    assert results["samples_evaluated"] > 0


def test_calculate_profit_score_edge_cases():
    """Test profit score calculation with edge cases"""
    te = ThresholdEvaluator(investment_amount=100.0)

    # Test with mixed predictions
    predictions = np.array([-0.1, 0.05, -0.2])
    actual = np.array([0.1, -0.03, 0.18])
    current_prices = np.array([10.0, 20.0, 5.0])

    profit_score = te.calculate_profit_score(predictions, actual, current_prices)

    # Should return some profit/loss based on investments made
    assert isinstance(profit_score, float)

    # Test with positive predictions that lose money
    predictions_loss = np.array([0.1, -0.05, 0.2])
    actual_loss = np.array([-0.1, -0.03, -0.18])  # All lose money
    current_prices_loss = np.array([10.0, 20.0, 5.0])

    profit_score_loss = te.calculate_profit_score(
        predictions_loss, actual_loss, current_prices_loss
    )

    # Should return some profit/loss value
    assert isinstance(profit_score_loss, float)


def test_predict_with_threshold_returns_dict():
    """Test threshold-based prediction returns proper dictionary"""
    te = ThresholdEvaluator(investment_amount=100.0)

    # Create mock model
    class MockModel:
        def predict(self, X):
            return np.array([0.1, -0.05, 0.2])

        def get_prediction_confidence(self, X, method="leaf_depth"):
            return np.array([0.9, 0.8, 0.95])

    model = MockModel()
    X = pd.DataFrame({"feature": [1, 2, 3]})

    # Test with threshold
    result = te.predict_with_threshold(model, X, threshold=0.85)

    # Should return dictionary with expected keys
    assert isinstance(result, dict)
    assert "filtered_predictions" in result
    assert "all_predictions" in result
    assert "confidence_threshold" in result
    assert "samples_kept_ratio" in result

    # With threshold 0.85, only predictions with confidence >= 0.85 should be kept
    # (0.9 and 0.95, but not 0.8)
    assert len(result["filtered_predictions"]) <= len(result["all_predictions"])


def test_calculate_investment_metrics():
    """Test investment metrics calculation"""
    te = ThresholdEvaluator(investment_amount=100.0)

    predictions = np.array([0.1, -0.05, 0.2, 0.0])
    actual = np.array([0.12, -0.03, 0.18, 0.05])

    metrics = te.calculate_investment_metrics(predictions, actual)

    assert isinstance(metrics, dict)
    assert "investments_made" in metrics
    assert "investment_rate" in metrics
    assert "total_samples" in metrics

    # Should have made investments for positive predictions (3 out of 4)
    assert metrics["investments_made"] == 3
    assert metrics["investment_rate"] == 0.75  # 3/4
    assert metrics["total_samples"] == 4
