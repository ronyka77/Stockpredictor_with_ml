"""
AutoGluon training entrypoint

Uses prepare_ml_data_for_training_with_cleaning to obtain X_train/y_train and
X_test/y_test, constructs train_df/valid_df with `Future_Return_10`, trains an
AutoGluonModel, and logs artifacts/metrics to MLflow.

Run:
uv run python -m src.models.automl.autogluon_trainer --preset high_quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.logger import get_logger
from src.models.common.training_data_prep import prepare_common_training_data
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.evaluation.threshold_evaluator import ThresholdEvaluator


logger = get_logger(__name__)

def train_autogluon(*,
                    prediction_horizon: int = 10,
                    presets: str = 'high_quality') -> Dict[str, Any]:
    # Use centralized common training data preparation
    data = prepare_common_training_data(
        prediction_horizon=prediction_horizon,
        recent_date_int_cut=15,
    )

    X_train: pd.DataFrame = data['X_train']
    y_train: pd.Series = data['y_train']
    X_test: pd.DataFrame = data['X_test']
    y_test: pd.Series = data['y_test']

    # Build model
    config = {
        'label': 'Future_Return_10',
        'presets': presets,
    }
    model = AutoGluonModel(model_name='autogluon', config=config)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Evaluate baseline RMSE/MAE/R2
    preds = model.predict(X_test)
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
        'mae': float(mean_absolute_error(y_test, preds)),
        'r2': float(r2_score(y_test, preds)),
    }
    logger.info(f"Baseline metrics: {metrics}")
    model.feature_importance(X_test)
    # Threshold optimization with profit evaluation
    evaluator = ThresholdEvaluator()
    try:
        current_prices_test = X_test['close'].to_numpy()
        best_info, results_df = evaluator.optimize_prediction_threshold(
            model=model,
            X_test=X_test,  
            y_test=y_test,
            current_prices_test=current_prices_test,
        )
        metrics.update({
            'final_optimal_threshold': float(best_info.get('optimal_threshold', np.nan)),
            'profit_per_investment': float(best_info.get('test_profit_per_investment', np.nan)),
            'investment_success_rate': float(best_info.get('investment_success_rate', np.nan)),
        })
    except Exception as e:
        logger.warning(f"Threshold evaluation failed: {e}")

    # Log to MLflow
    params_to_log = {
        'prediction_horizon': prediction_horizon,
        'presets': presets,
    }
    model.save_to_mlflow(params=params_to_log, metrics=metrics, experiment_name='stock_prediction_autogluon')

    return {
        'model': model,
        'metrics': metrics,
        'feature_names': model.feature_names,
    }


def main():
    train_autogluon(
        prediction_horizon=10,
        presets='high_quality',
    )


if __name__ == "__main__":
    main()


