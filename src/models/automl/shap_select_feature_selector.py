"""One-time SHAP-based feature selector using shap-select.

Loads training data via `prepare_common_training_data`, trains a compact
AutoGluon predictor on a sample, runs ShapSelect, and writes the selected
feature list to a JSON file under `predictions/`.
"""
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
# AutoGluon predictor import kept for historical reference but unused in current flow

# psutil removed: not required in this module
from shap_select import shap_select
from lightgbm import LGBMRegressor

from src.models.automl.autogluon_trainer import top_dataframes
from src.utils.cleaned_data_cache import collect_garbage
from src.utils.logger import get_logger
from src.models.common.training_data_prep import prepare_common_training_data

logger = get_logger(__name__)

# Local metric replicated from project to remain compatible with AutoGluon
def conditional_mean_absolute_error(y_true, y_pred, *args, **kwargs):
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if y_t.shape != y_p.shape:
        y_p = np.reshape(y_p, y_t.shape)
    abs_errors = np.abs(y_t - y_p)
    mask_positive_over = (y_t > 0) & (y_t > y_p)
    mask_negative_under = (y_t < 0) & (y_p < y_t)
    abs_errors[mask_positive_over] = 0.0
    abs_errors[mask_negative_under] = 0.0
    return float(np.mean(abs_errors))


def select_features(
    train_df: pd.DataFrame,
    label: str,
    sample_frac: float = 0.25,
    max_features: Optional[int] = None,
) -> List[str]:
    """Return a list of selected feature names (excluding label).

    Steps:
    - Sample training data to make SHAP feasible on large datasets.
    - Train a lightweight AutoGluon model (presets='medium_quality')
    - Use the trained model's predictions as the basis for ShapSelect.
    - Fit ShapSelect and return selected columns.
    """

    # Sample
    if sample_frac < 1.0:
        train_sample = train_df.sample(frac=sample_frac, random_state=42)
    else:
        train_sample = train_df
    
    # Prepare X, y for shap-select
    X = train_sample.drop(columns=[label])
    y = train_sample[label]

    # Fit a lightweight tree model for SHAP-based selection (LightGBM)
    tree_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbose=0)

    # LightGBM eval function wrapper using the project's metric
    def lgb_conditional_mae(y_pred, dataset):
        """Robust wrapper for LightGBM eval metric.

        LightGBM may pass different types for `dataset` depending on API/version:
        - a lightgbm.Dataset with `get_label()`
        - a tuple/list like (X_eval, y_eval)
        - a numpy array containing y_true
        This function attempts to extract y_true from all of the above shapes.
        """
        # Extract labels from dataset in a robust way
        try:
            if hasattr(dataset, "get_label"):
                y_true = dataset.get_label()
            elif isinstance(dataset, (tuple, list)) and len(dataset) >= 2:
                # common scikit-learn style eval_set entry (X_val, y_val)
                y_true = np.asarray(dataset[1])
            elif isinstance(dataset, np.ndarray):
                y_true = dataset
            else:
                # Last-resort coercion
                y_true = np.asarray(dataset)
        except Exception as exc:
            logger.warning(f"Failed to extract labels from LightGBM dataset object: {exc}")
            return "cond_mae", float("inf"), False

        # If extraction succeeded but yielded empty labels, bail out
        try:
            if getattr(y_true, "size", None) == 0:
                return "cond_mae", float("inf"), False
        except Exception:
            # defensive fallback
            return "cond_mae", float("inf"), False

        try:
            val = conditional_mean_absolute_error(y_true, y_pred)
        except Exception as exc:
            logger.warning(f"conditional_mean_absolute_error failed inside LGB eval: {exc}")
            val = float("inf")

        # (name, value, is_higher_better) format for LightGBM
        return "cond_mae", float(val), False

    try:
        tree_model.fit(X, y, eval_set=[(X, y)], eval_metric=lgb_conditional_mae)
    except Exception as exc:
        # If LightGBM fails, we log and continue with tree_model set to None
        logger.warning(f"LightGBM training failed, falling back to AutoGluon predictor: {exc}")
        tree_model = None

    # Use the functional API shap_select which returns a DataFrame of features
    try:
        if tree_model is not None:
            result_df = shap_select(
                tree_model=tree_model,
                validation_df=X,
                target=y,
                feature_names=X.columns.tolist(),
                task="regression",
                threshold=0.05,
                return_extended_data=False,
                alpha=1e-6,
            )
        # If no tree_model is available we will fall back to selecting all features below
    except Exception as exc:
        logger.error(f"shap_select failed: {exc}. Falling back to selecting all features.")
        selection_df = pd.DataFrame({"feature": X.columns.tolist(), "Selected": [True] * len(X.columns)})
        result_df = selection_df

    # If tree_model was None and shap_select was not executed, ensure result_df is defined
    if "result_df" not in locals() or result_df is None:
        logger.warning(
            "shap_select was not executed (no tree model available); defaulting to selecting all features."
        )
        result_df = pd.DataFrame({"feature": X.columns.tolist(), "Selected": [True] * len(X.columns)})

    # Expect result_df to contain a boolean 'Selected' column and 'feature' name
    if isinstance(result_df, tuple):
        # shap_select may return (selection_df, extended_df)
        selection_df = result_df[0]
    else:
        selection_df = result_df

    if "Selected" in selection_df.columns and "feature" in selection_df.columns:
        feature_names = selection_df.loc[selection_df["Selected"], "feature"].tolist()
    else:
        # Fallback: if selection_df lists features directly, coerce to list
        try:
            feature_names = list(selection_df["feature"])
        except Exception:
            feature_names = list(X.columns)

    # Optional: limit number of features returned
    if max_features is not None and len(feature_names) > max_features:
        feature_names = feature_names[:max_features]

    return feature_names


def main():
    # Configuration variables (user-editable)
    prediction_horizon = 20  # matches Future_Return_20D label
    prediction_horizon_label = f"Future_Return_{prediction_horizon}D"
    # groups variable reserved for backward compatibility in other modules

    # Data preparation params
    sample_frac = 0.25
    max_features = 120

    output_path = "predictions/selected_features_autogluon.json"

    # Load prepared data from project utility
    data = prepare_common_training_data(
        prediction_horizon=prediction_horizon,
        recent_date_int_cut=25,
    )
    logger.info(f"target_column: {data.get('target_column')}")
    x_train: pd.DataFrame = data["x_train"]
    logger.info(f"x_train: {x_train.columns.tolist()}")
    y_train: pd.Series = data["y_train"]
    x_test: pd.DataFrame = data["x_test"]
    y_test: pd.Series = data["y_test"]
    valid_df: pd.DataFrame = pd.concat([x_test, y_test], axis=1)
    valid_df.reset_index(drop=True, inplace=True)

    # Combine x_train and y_train into a single DataFrame expected by select_features
    train_df = x_train.copy()
    train_df[prediction_horizon_label] = y_train.values

    top_dataframes(10)
    collect_garbage()
    top_dataframes(10)
    

    logger.info("Running ShapSelect-based feature selection")
    features = select_features(
        train_df=train_df,
        label=prediction_horizon_label,
        sample_frac=sample_frac,
        max_features=max_features,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"selected_features": features}, fh, indent=2)

    logger.info(
        f"Wrote {len(features)} selected features to {output_path}, features: {', '.join(features)}"
    )

    collect_garbage()


if __name__ == "__main__":
    main()


