"""
RealMLP Main Runner

This module provides a standalone training/evaluation runner for the RealMLP
model with preprocessing, MLflow logging, and basic metrics.

Usage (Windows with uv):
    uv run python -m src.models.time_series.realmlp.realmlp_main
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch

from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.models.time_series.realmlp.realmlp_preprocessing import RealMLPPreprocessor
from src.models.time_series.realmlp.realmlp_predictor import RealMLPPredictor
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.1
    activation: str = "gelu"
    batch_norm: bool = True
    use_diagonal: bool = True
    use_numeric_embedding: bool = True
    numeric_embedding_dim: int = 16
    hidden_sizes: Tuple[int, ...] = (512, 256, 128, 64)
    use_huber: bool = False
    huber_delta: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = False


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    cat_cols = ["ticker_id"]
    numeric_cols = [c for c in df.columns if c not in cat_cols]
    return numeric_cols

def _build_run_parameters(cfg: TrainingConfig, input_size: int, num_categories: Optional[int]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "input_size": input_size,
        "hidden_sizes": list(cfg.hidden_sizes),
        "activation": cfg.activation,
        "dropout": cfg.dropout,
        "batch_norm": cfg.batch_norm,
        "use_diagonal": cfg.use_diagonal,
        "use_numeric_embedding": cfg.use_numeric_embedding,
        "numeric_embedding_dim": cfg.numeric_embedding_dim,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "use_huber": cfg.use_huber,
        "huber_delta": cfg.huber_delta,
        "grad_clip": cfg.grad_clip,
        "use_amp": cfg.use_amp,
    }
    if num_categories is not None:
        params["num_categories"] = int(num_categories)
    return params

def main() -> None:
    logger.info("=" * 80)
    logger.info("🎯 RealMLP TRAINING & EVALUATION")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_trials = 10
    # 1) Data preparation (reusing existing project pipeline)
    data = prepare_ml_data_for_training_with_cleaning(
        prediction_horizon=10,
        split_date="2025-02-01",
        ticker=None,
        clean_features=True,
    )
    X_train: pd.DataFrame = data["X_train"].copy()
    X_test: pd.DataFrame = data["X_test"].copy()
    y_train: pd.Series = data["y_train"].copy()
    y_test: pd.Series = data["y_test"].copy()

    # 2) Preprocessing (clean numeric → clip + robust scale; categorical mapping)
    numeric_cols = _get_numeric_columns(X_train) 
    X_train_clean = MLPDataUtils.validate_and_clean_data(X_train)
    X_test_clean = MLPDataUtils.validate_and_clean_data(X_test)
    y_train_array = np.asarray(y_train.values, dtype=np.float32)
    y_test_array = np.asarray(y_test.values, dtype=np.float32)

    pre = RealMLPPreprocessor()
    pre.fit(X_train_clean, numeric_cols=numeric_cols)
    scaler = pre.scaler
    X_train_num, train_cat_idx = pre.transform(X_train_clean, numeric_cols=numeric_cols)
    X_test_num, test_cat_idx = pre.transform(X_test_clean, numeric_cols=numeric_cols)    

    train_finite_mask = np.isfinite(X_train_num).all(axis=1) & np.isfinite(y_train_array)
    test_finite_mask = np.isfinite(X_test_num).all(axis=1) & np.isfinite(y_test_array)

    dropped_train = int(len(train_finite_mask) - int(train_finite_mask.sum()))
    dropped_test = int(len(test_finite_mask) - int(test_finite_mask.sum()))
    if dropped_train or dropped_test:
        logger.warning(
            f"Data contained non-finite values after preprocessing; dropping rows — train:{dropped_train} test:{dropped_test}"
        )

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
        X_train=X_train_clean,
        y_train=y_train,
        X_val=X_test_clean,
        y_val=y_test,
        preprocessor=pre,
        confidence_method="variance",
        n_trials=num_trials
    )
    results = results.get("best_trial_info")

    # 4) Save to MLflow
    # params = _build_run_parameters(cfg, input_size=X_train_num.shape[1], num_categories=None)
    run_id = predictor.save_model(
        metrics=results,
        params=results.get("best_params"),
        X_eval=pd.DataFrame(X_test_num, columns=pre.feature_names),
        preprocessor=pre,
    )
    logger.info(f"✅ RealMLP run completed and saved to MLflow: run_id={run_id}")


if __name__ == "__main__":
    main()


