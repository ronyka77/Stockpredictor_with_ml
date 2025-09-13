import pandas as pd
import numpy as np
from unittest.mock import patch

from src.models.common.training_data_prep import prepare_common_training_data


def _make_sample_data():
    idx = pd.RangeIndex(0, 100)
    X = pd.DataFrame({"f1": np.linspace(0, 1, 100), "date_int": np.repeat(np.arange(20), 5)})
    y = pd.Series(np.linspace(-1, 1, 100))
    # split
    X_train = X.iloc[:80].reset_index(drop=True)
    X_test = X.iloc[80:].reset_index(drop=True)
    y_train = y.iloc[:80].reset_index(drop=True)
    y_test = y.iloc[80:].reset_index(drop=True)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_column": "target",
        "train_date_range": (0, 79),
        "test_date_range": (80, 99),
    }


def test_prepare_common_training_data_outlier_and_date_filtering():
    sample = _make_sample_data()

    with patch("src.models.common.training_data_prep.prepare_ml_data_for_training_with_cleaning", return_value=sample):
        res = prepare_common_training_data(prediction_horizon=1, outlier_quantiles=(0.0, 1.0), recent_date_int_cut=2)
        # Since outlier_quantiles include full range, lengths should match original splits after cleaning
        assert "X_train" in res and "X_test" in res
        assert res["X_train"].shape[0] <= 80
        # date_int filtering should remove recent groups in X_test when recent_date_int_cut=2
        if "date_int" in res["X_test"].columns:
            uniq = res["X_test"]["date_int"].drop_duplicates().sort_values()
            assert len(uniq) <= 2 or res["X_test"].shape[0] < 20
