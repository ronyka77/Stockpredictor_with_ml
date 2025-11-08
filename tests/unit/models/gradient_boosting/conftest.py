from typing import Iterator

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

# polyfactory not required for the deterministic SimpleDataFactory
from src.models.gradient_boosting.random_forest_model import RandomForestModel
from src.models.gradient_boosting.lightgbm_model import LightGBMModel


class SimpleDataFactory:
    """Lightweight factory to generate deterministic tabular data for tests."""

    @staticmethod
    def create_dataframe(n_rows: int = 100, n_features: int = 10) -> pd.DataFrame:
        cols = [f"f_{i}" for i in range(n_features)]
        data = np.arange(n_rows * n_features).reshape(n_rows, n_features).astype(float)
        df = pd.DataFrame(data, columns=cols)
        # Add required columns used by models
        df["close"] = np.linspace(10, 20, n_rows)
        df["ticker_id"] = 1
        df["date_int"] = np.arange(n_rows)
        return df

    @staticmethod
    def create_target_series(n_rows: int = 100) -> pd.Series:
        return pd.Series(np.linspace(10.5, 21.0, n_rows))


@pytest.fixture(scope="session")
def sample_tabular_data() -> Iterator[dict]:
    """Provide a small, deterministic dataset for models to consume in tests."""

    X = SimpleDataFactory.create_dataframe(n_rows=120, n_features=8)
    y = SimpleDataFactory.create_target_series(n_rows=120)

    yield {"X": X, "y": y}


@pytest.fixture()
def rf_model_instance() -> Iterator[RandomForestModel]:
    """Return a freshly initialized RandomForestModel for isolated tests."""

    model = RandomForestModel(model_name="rf_test_model", prediction_horizon=10)
    yield model


@pytest.fixture()
def lgb_model_instance() -> Iterator[LightGBMModel]:
    """Return a freshly initialized LightGBMModel for isolated tests."""

    model = LightGBMModel(model_name="lgb_test_model", prediction_horizon=10)
    yield model


@pytest.fixture()
def mock_threshold_evaluator() -> Mock:
    """Provide a mock ThresholdEvaluator with controlled return values."""

    mock = Mock()
    # Default behavior: optimization returns a successful thin result
    mock.optimize_prediction_threshold.return_value = {
        "status": "success",
        "optimal_threshold": 0.6,
        "best_result": {
            "test_profit_per_investment": 1.5,
            "test_samples_ratio": 0.5,
            "test_investment_success_rate": 0.6,
            "test_custom_accuracy": 0.7,
            "test_profit": 150.0,
            "profitable_investments": 10,
            "investment_success_rate": 0.6,
        },
    }
    mock.evaluate_threshold_performance.return_value = {
        "profit_per_investment": 1.5,
        "total_profit": 150.0,
        "samples_evaluated": 60,
        "samples_kept_ratio": 0.5,
        "investment_success_rate": 0.6,
        "mse": 0.1,
        "mae": 0.05,
        "r2_score": 0.8,
    }
    yield mock
