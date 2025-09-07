import numpy as np
import pandas as pd

from src.models.evaluation.threshold_evaluator import ThresholdEvaluator


def test_vectorized_profit_calculation_basic():
    te = ThresholdEvaluator(investment_amount=100.0)

    # y_true: actual percentage returns; y_pred: predicted percentage returns
    y_true = np.array([0.1, -0.05, 0.2, 0.0])
    y_pred = np.array([0.05, -0.02, 0.03, 0.0])
    current_prices = np.array([10.0, 20.0, 5.0, 8.0])

    profits = te._vectorized_profit_calculation(y_true, y_pred, current_prices)

    # Only positions 0 and 2 have y_pred > 0
    # position 0: shares = 100/10 = 10 -> profit = 10 * 10 * 0.1 = 10
    # position 2: shares = 100/5 = 20 -> profit = 20 * 5 * 0.2 = 20
    assert np.isclose(profits[0], 10.0)
    assert profits[1] == 0
    assert np.isclose(profits[2], 20.0)
    assert profits[3] == 0


def test_calculate_filtered_profit_edge_cases():
    te = ThresholdEvaluator(investment_amount=50.0)

    # No positive predictions -> zero profit
    y_true = np.array([0.1, 0.2])
    y_pred = np.array([-0.01, -0.02])
    prices = np.array([10.0, 20.0])
    assert te.calculate_filtered_profit(y_true, y_pred, prices) == 0.0

    # Some positive predictions
    y_pred2 = np.array([0.05, -0.01])
    profit = te.calculate_filtered_profit(y_true, y_pred2, prices)
    # invest in first: shares = 50 / 10 =5 -> profit = 5*10*0.1 =5
    assert np.isclose(profit, 5.0)


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

    assert res["status"] == "success"
    assert "optimal_threshold" in res
    assert res["n_thresholds_tested"] >= 1
