import numpy as np
import pandas as pd
import pytest

from src.models.evaluation.threshold_evaluator import ThresholdEvaluator


class DummyModel:
    def __init__(self, predictions: np.ndarray, confidence: np.ndarray):
        self._pred = predictions
        self._conf = confidence

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pred[: len(X)]

    def get_prediction_confidence(
        self, X: pd.DataFrame, method: str = "simple"
    ) -> np.ndarray:
        return self._conf[: len(X)]


@pytest.mark.unit
def test_vectorized_profit_calculation_positive_only():
    te = ThresholdEvaluator(investment_amount=100.0)
    y_true = np.array([0.1, -0.2, 0.05, 0.0])
    y_pred = np.array([0.05, -0.1, 0.02, 0.0])
    prices = np.array([10.0, 20.0, 5.0, 8.0])

    profits = te._vectorized_profit_calculation(y_true, y_pred, prices)

    # Invest only in indices 0 and 2 (positive predictions)
    # profit = (100/price) * price * actual_return = 100 * actual_return
    expected = np.array([100 * 0.1, 0.0, 100 * 0.05, 0.0])
    np.testing.assert_allclose(profits, expected)


@pytest.mark.unit
def test_vectorized_threshold_testing_monotonic_mask():
    n = 10000
    X = pd.DataFrame({"f": np.arange(n)})
    y = np.linspace(-0.30, 0.50, n)
    preds = y + 0.0
    conf = np.linspace(0.0, 1.0, n)
    prices = np.random.uniform(5.0, 15.0, n)  # Random prices between $5-15

    model = DummyModel(predictions=preds, confidence=conf)
    te = ThresholdEvaluator(investment_amount=100.0)

    res = te.optimize_prediction_threshold(
        model=model,
        X_test=X,
        y_test=pd.Series(y),
        current_prices_test=prices,
        confidence_method="simple",
        threshold_range=(0.01, 0.99),
        n_thresholds=99,
    )

    if res.get("status") != "success":
        raise AssertionError("Optimization did not complete successfully")
    df = res["all_results"]
    # Monotonicity: as threshold increases, samples_kept should not increase
    kept = df.sort_values("threshold")["test_samples_kept"].values
    if not np.all(kept[:-1] >= kept[1:]):
        raise AssertionError(
            "Monotonicity violation: samples_kept increases with threshold"
        )


@pytest.mark.unit
def test_predict_with_threshold_filters_and_returns_confidence():
    n = 20
    X = pd.DataFrame({"f": np.arange(n)})
    y = np.linspace(-0.01, 0.02, n)
    preds = y
    conf = np.linspace(0.0, 1.0, n)
    model = DummyModel(predictions=preds, confidence=conf)
    te = ThresholdEvaluator(investment_amount=100.0)

    out = te.predict_with_threshold(
        model, X, threshold=0.5, confidence_method="simple", return_confidence=True
    )
    if out["filtered_samples"] > n:
        raise AssertionError("Filtered samples count exceeds input samples")
    if not ("all_confidence" in out and "filtered_confidence" in out):
        raise AssertionError(
            "Missing confidence outputs in predict_with_threshold result"
        )
