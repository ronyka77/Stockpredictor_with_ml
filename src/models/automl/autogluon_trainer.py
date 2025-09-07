"""
AutoGluon training entrypoint

Uses prepare_ml_data_for_training_with_cleaning to obtain X_train/y_train and
X_test/y_test, constructs train_df/valid_df with `Future_Return_10D`, trains an
AutoGluonModel, and logs artifacts/metrics to MLflow.

Run:
uv run python -m src.models.automl.autogluon_trainer --preset high_quality
"""



import pandas as pd
from typing import Any, Dict

from src.utils.logger import get_logger
from src.models.common.training_data_prep import prepare_common_training_data
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.automl.autogluen_load_model import run_model_evaluation


logger = get_logger(__name__)

def train_autogluon(*,
                    prediction_horizon: int = 10,
                    presets: str = 'high_quality') -> Dict[str, Any]:
    # Use centralized common training data preparation
    data = prepare_common_training_data(
        prediction_horizon=prediction_horizon,
        recent_date_int_cut=25,
    )
    logger.info(f"target_column: {data.get('target_column')}")

    X_train: pd.DataFrame = data['X_train']
    logger.info(f"X_train: {X_train.columns.tolist()}")
    y_train: pd.Series = data['y_train']
    X_test: pd.DataFrame = data['X_test']
    y_test: pd.Series = data['y_test']
    valid_df: pd.DataFrame = pd.concat([X_test, y_test], axis=1)
    valid_df.reset_index(drop=True, inplace=True)

    # Build model
    config = {
        'label': 'Future_Return_10D',
        'presets': presets,
    }
    model = AutoGluonModel(model_name='autogluon', config=config)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    model_dir = model.predictor.path
    logger.info(f"Model directory: {model_dir}")
    run_model_evaluation(model_dir, prediction_horizon)

    return {
        'model': model,
        'feature_names': model.feature_names,
    }


def main():
    train_autogluon(
        prediction_horizon=10,
        presets='high_quality',
    )


if __name__ == "__main__":
    main()


