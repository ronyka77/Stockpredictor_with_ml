"""
AutoGluon training entrypoint

Uses prepare_ml_data_for_training_with_cleaning to obtain x_train/y_train and
x_test/y_test, constructs train_df/valid_df with `Future_Return_XD`, trains an
AutoGluonModel, and logs artifacts/metrics to MLflow.

Run:
uv run python -m src.models.automl.autogluon_trainer --preset high_quality
"""

import contextlib
import os
import psutil
import gc
import pandas as pd
from typing import Any, Dict

from src.utils.core.logger import get_logger
from src.models.base_model import (
    filter_to_selected_features,
    load_and_prepare_data_polars_as_pandas,
)
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.automl.autogluen_load_model import run_model_evaluation


logger = get_logger(__name__)


def top_dataframes(top_n: int = 10):
    dfs = []
    logger.info("TOP dataframes:")
    for obj in gc.get_objects():
        with contextlib.suppress(Exception):
            if isinstance(obj, pd.DataFrame):
                size = obj.memory_usage(deep=True).sum()
                dfs.append((size, obj))
    dfs.sort(reverse=True, key=lambda x: x[0])
    for size, df in dfs[:top_n]:
        logger.info(f"{size / 1024**2:8.2f} MB | shape={df.shape} | columns={len(df.columns)}")
    return dfs


def train_autogluon(
    *, prediction_horizon: int = 20, presets: str = "best_quality"
) -> Dict[str, Any]:
    data = load_and_prepare_data_polars_as_pandas(
        prediction_horizon=prediction_horizon, clean_features=True
    )
    logger.info(f"target_column: {data.get('target_column')}")
    x_train: pd.DataFrame = data["x_train"]
    logger.info(f"x_train: {x_train.columns.tolist()}")
    y_train: pd.Series = data["y_train"]
    x_test: pd.DataFrame = data["x_test"]
    y_test: pd.Series = data["y_test"]

    x_train, x_test, selected_features_path = filter_to_selected_features(x_train, x_test)

    collected = gc.collect()
    logger.info(f"Garbage collected: {collected}")
    proc = psutil.Process(os.getpid())
    logger.info(f"RSS MB: {proc.memory_info().rss / 1024**2}")
    top_dataframes(10)

    config = {"label": f"Future_Return_{prediction_horizon}D", "presets": presets, "groups": "year"}
    model = AutoGluonModel(model_name="autogluon", config=config)
    model.fit(x_train, y_train, X_val=x_test, y_val=y_test)

    model_dir = model.predictor.path
    logger.info(f"Model directory: {model_dir}")
    run_model_evaluation(model_dir, prediction_horizon)

    return {
        "model": model,
        "feature_names": model.feature_names,
        "selected_features_path": selected_features_path,
    }


def main():
    prediction_horizon = 20
    train_autogluon(prediction_horizon=prediction_horizon, presets="best_quality")


if __name__ == "__main__":
    main()
