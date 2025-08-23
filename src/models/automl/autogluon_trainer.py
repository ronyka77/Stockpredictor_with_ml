"""
AutoGluon training entrypoint

Uses prepare_ml_data_for_training_with_cleaning to obtain X_train/y_train and
X_test/y_test, constructs train_df/valid_df with `Future_Return_10`, trains an
AutoGluonModel, and logs artifacts/metrics to MLflow.

Run:
uv run python -m src.models.automl.autogluon_trainer --preset high_quality
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.evaluation.threshold_evaluator import ThresholdEvaluator


logger = get_logger(__name__)


def train_autogluon(*,
                    prediction_horizon: int = 10,
                    split_date: str = '2025-02-01',
                    presets: str = 'high_quality',
                    num_cpus: int = 12,
                    eval_metric: str = 'rmse') -> Dict[str, Any]:
    data = prepare_ml_data_for_training_with_cleaning(
        prediction_horizon=prediction_horizon,
        split_date=split_date,
        clean_features=True,
    )

    X_train: pd.DataFrame = data['X_train']
    y_train: pd.Series = data['y_train']
    X_test: pd.DataFrame = data['X_test']
    y_test: pd.Series = data['y_test']

    # Build model
    config = {
        'label': 'Future_Return_10',
        'eval_metric': eval_metric,
        'presets': presets,
        'ag_args_fit': {"num_cpus": num_cpus, "num_gpus": 0},
        'hyperparameters': {
            "GBM": {}, "XGB": {}, "CAT": {}, "RF": {}, "XT": {}, "REALMLP": {},
        }
    }
    model = AutoGluonModel(model_name='autogluon', config=config)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Evaluate baseline RMSE/MAE/R2
    preds = model.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
        'mae': float(mean_absolute_error(y_test, preds)),
        'r2': float(r2_score(y_test, preds)),
    }
    logger.info(f"Baseline metrics: {metrics}")

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
        'num_cpus': num_cpus,
        'eval_metric': eval_metric,
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
        split_date='2025-02-01',
        presets='high_quality',
        num_cpus=12,
        eval_metric='rmse',
    )


if __name__ == "__main__":
    main()


