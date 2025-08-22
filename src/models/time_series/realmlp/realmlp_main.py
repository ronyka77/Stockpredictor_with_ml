"""
RealMLP Main Runner

This module provides a standalone training/evaluation runner for the RealMLP
model with preprocessing, MLflow logging, and basic metrics.

Usage (Windows with uv):
    uv run python -m src.models.time_series.realmlp.realmlp_main
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.models.time_series.common.training_data_prep import prepare_common_training_data
from src.models.time_series.realmlp.realmlp_preprocessing import RealMLPPreprocessor
from src.models.time_series.realmlp.realmlp_predictor import RealMLPPredictor
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    cat_cols = ["ticker_id"]
    numeric_cols = [c for c in df.columns if c not in cat_cols]
    return numeric_cols

def main() -> None:
    logger.info("=" * 80)
    logger.info("🎯 RealMLP TRAINING & EVALUATION")
    logger.info("=" * 80)

    num_trials = 10
    # 1) Centralized data preparation
    prep = prepare_common_training_data(
        prediction_horizon=10,
        outlier_quantiles=(0.05, 0.95),
        recent_date_int_cut=15,
    )
    X_train: pd.DataFrame = prep["X_train"].copy()
    X_test: pd.DataFrame = prep["X_test"].copy()
    y_train: pd.Series = prep["y_train"].copy()
    y_test: pd.Series = prep["y_test"].copy()

    # 2) Preprocessing (clean numeric → clip + robust scale; categorical mapping)
    numeric_cols = _get_numeric_columns(X_train) 
    y_train_array = np.asarray(y_train.values, dtype=np.float32)
    y_test_array = np.asarray(y_test.values, dtype=np.float32)

    pre = RealMLPPreprocessor()
    pre.fit(X_train, numeric_cols=numeric_cols)
    X_train_num, train_cat_idx = pre.transform(X_train, numeric_cols=numeric_cols)
    X_test_num, test_cat_idx = pre.transform(X_test, numeric_cols=numeric_cols)    

    train_finite_mask = np.isfinite(X_train_num).all(axis=1) & np.isfinite(y_train_array)
    test_finite_mask = np.isfinite(X_test_num).all(axis=1) & np.isfinite(y_test_array)

    dropped_train = int(len(train_finite_mask) - int(train_finite_mask.sum()))
    dropped_test = int(len(test_finite_mask) - int(test_finite_mask.sum()))
    if dropped_train or dropped_test:
        logger.warning(f"Data contained non-finite values; dropping rows — train:{dropped_train} test:{dropped_test}")

    X_train_num = X_train_num[train_finite_mask]
    y_train_array = y_train_array[train_finite_mask]
    if train_cat_idx is not None:
        train_cat_idx = train_cat_idx[train_finite_mask]

    X_test_num = X_test_num[test_finite_mask]
    y_test_array = y_test_array[test_finite_mask]
    if test_cat_idx is not None:
        test_cat_idx = test_cat_idx[test_finite_mask]
    
    predictor = RealMLPPredictor(model_name="RealMLP")
    results = predictor.optuna_hypertune(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        preprocessor=pre,
        confidence_method="latent_mahalanobis",
        n_trials=num_trials
    )
    metrics = results.get("best_trial_info")

    # 4) Save to MLflow
    run_id = predictor.save_model(
        metrics=metrics,
        params=results.get("best_params"),
        X_eval=pd.DataFrame(X_test_num, columns=pre.feature_names),
        preprocessor=pre,
    )
    logger.info(f"✅ RealMLP run completed and saved to MLflow: run_id={run_id}")


if __name__ == "__main__":
    main()


