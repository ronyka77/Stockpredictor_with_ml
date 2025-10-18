"""
AutoGluon training entrypoint

Uses prepare_ml_data_for_training_with_cleaning to obtain x_train/y_train and
x_test/y_test, constructs train_df/valid_df with `Future_Return_XD`, trains an
AutoGluonModel, and logs artifacts/metrics to MLflow.

Run:
uv run python -m src.models.automl.autogluon_trainer --preset high_quality
"""

import os
import json
import psutil
import gc
import pandas as pd
from typing import Any, Dict, Tuple

from src.utils.logger import get_logger
from src.models.common.training_data_prep import prepare_common_training_data
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.automl.autogluen_load_model import run_model_evaluation


logger = get_logger(__name__)


def filter_to_selected_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Filter training and test datasets to only include features selected by AutoGluon.

    Args:
        x_train: Training feature DataFrame
        x_test: Test feature DataFrame

    Returns:
        Tuple of (filtered_x_train, filtered_x_test, metadata_dict)
    """
    selected_features_path = "predictions/selected_features_autogluon.json"

    if os.path.exists(selected_features_path):
        try:
            with open(selected_features_path, 'r') as f:
                selected_features_data = json.load(f)
                selected_features = selected_features_data["selected_features"]


            # Filter to only include selected features that exist in the dataset
            available_features = [col for col in selected_features if col in x_train.columns]
            features = available_features

            if len(available_features) != len(selected_features):
                missing_features = [col for col in selected_features if col not in x_train.columns]
                logger.warning(f"Missing features in dataset: {missing_features}")

            x_train_filtered = x_train[available_features]
            x_test_filtered = x_test[available_features]

            logger.info(f"Filtered to {len(available_features)} selected features")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading selected features file: {e}. Using all features.")
            x_train_filtered, x_test_filtered = x_train, x_test
    else:
        logger.warning(f"Selected features file not found: {selected_features_path}. Using all features.")
        x_train_filtered, x_test_filtered = x_train, x_test

    return x_train_filtered, x_test_filtered


def top_dataframes(top_n: int = 10):
    dfs = []
    logger.info("TOP dataframes:")
    for obj in gc.get_objects():
        try:
            if isinstance(obj, pd.DataFrame):
                size = obj.memory_usage(deep=True).sum()
                dfs.append((size, obj))
        except Exception:
            continue
    dfs.sort(reverse=True, key=lambda x: x[0])
    for size, df in dfs[:top_n]:
        logger.info(
            f"{size / 1024**2:8.2f} MB | shape={df.shape} | columns={len(df.columns)}"
        )
    return dfs


def train_autogluon(
    *, prediction_horizon: int = 20, presets: str = "best_quality"
) -> Dict[str, Any]:
    # Use centralized common training data preparation
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

    # Filter datasets to only include selected features
    x_train, x_test = filter_to_selected_features(x_train, x_test)
    valid_df: pd.DataFrame = pd.concat([x_test, y_test], axis=1)
    valid_df.reset_index(drop=True, inplace=True)

    collected = gc.collect()
    logger.info(f"Garbage collected: {collected}")
    proc = psutil.Process(os.getpid())
    print("RSS MB:", proc.memory_info().rss / 1024**2)
    top_dataframes(10)
    # Build model
    config = {
        "label": f"Future_Return_{prediction_horizon}D",
        "presets": presets,
        "groups": "year",
    }
    model = AutoGluonModel(model_name="autogluon", config=config)
    model.fit(x_train, y_train, X_val=x_test, y_val=y_test)

    model_dir = model.predictor.path
    logger.info(f"Model directory: {model_dir}")
    run_model_evaluation(model_dir, prediction_horizon)

    return {
        "model": model,
        "feature_names": model.feature_names,
    }


def main():
    prediction_horizon = 20
    train_autogluon(
        prediction_horizon=prediction_horizon,
        presets="best_quality",
    )


if __name__ == "__main__":
    main()
