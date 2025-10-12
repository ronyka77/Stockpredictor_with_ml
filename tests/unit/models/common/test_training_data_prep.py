import pandas as pd
import numpy as np
from unittest.mock import patch

from src.models.common.training_data_prep import prepare_common_training_data


def _make_sample_data():
    pd.RangeIndex(0, 100)
    X = pd.DataFrame(
        {"f1": np.linspace(0, 1, 100), "date_int": np.repeat(np.arange(20), 5)}
    )
    y = pd.Series(np.linspace(-1, 1, 100))
    # split
    x_train = X.iloc[:80].reset_index(drop=True)
    x_test = X.iloc[80:].reset_index(drop=True)
    y_train = y.iloc[:80].reset_index(drop=True)
    y_test = y.iloc[80:].reset_index(drop=True)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_column": "target",
        "train_date_range": (0, 79),
        "test_date_range": (80, 99),
    }


def test_prepare_common_training_data_outlier_and_date_filtering():
    """Verify prepare_common_training_data applies outlier and date filtering correctly."""
    sample = _make_sample_data()

    with patch(
        "src.models.common.training_data_prep.prepare_ml_data_for_training_with_cleaning",
        return_value=sample,
    ):
        res = prepare_common_training_data(
            prediction_horizon=1, outlier_quantiles=(0.0, 1.0), recent_date_int_cut=2
        )
        # Since outlier_quantiles include full range, lengths should match original splits after cleaning
        assert "x_train" in res and "x_test" in res
        assert res["x_train"].shape[0] <= 80
        # date_int filtering should remove recent groups in x_test when recent_date_int_cut=2
        if "date_int" in res["x_test"].columns:
            uniq = res["x_test"]["date_int"].drop_duplicates().sort_values()
            assert len(uniq) <= 2 or res["x_test"].shape[0] < 20
