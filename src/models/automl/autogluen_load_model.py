from typing import Any, Dict
import os
import json
import pandas as pd
from datetime import datetime
import warnings
import logging

from autogluon.tabular import TabularPredictor

from src.utils.logger import get_logger
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_prediction_with_cleaning
from src.models.predictors.autogluon_predictor import AutoGluonPredictor

logger = get_logger(__name__)


def generate_and_log_leaderboard(
    predictor: TabularPredictor, valid_df: pd.DataFrame, model_names: list[str] = None
) -> pd.DataFrame:
    """Generate the Autogluon leaderboard using valid_df and log it.
    Returns the leaderboard DataFrame.
    """
    try:
        logger.info(f"Generating leaderboard on valid_df (n={len(valid_df)})")
        leaderboard = predictor.leaderboard(
            valid_df, silent=True, extra_metrics=["mape", "mse", "mae", "rmse", "r2"]
        )
        logger.info(f"Autogluon leaderboard:\n{leaderboard.to_string()}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        leaderboard.to_excel(
            f"pipeline_stats/leaderboard_{timestamp}.xlsx", index=False
        )
        for model_name in model_names:
            feature_importance = predictor.feature_importance(
                valid_df,
                confidence_level=0.95,
                model=model_name,
                subsample_size=20000,
                num_shuffle_sets=10,
            )
            feature_importance.to_excel(
                f"pipeline_stats/feature_importance_{model_name}_{timestamp}.xlsx"
            )
            logger.info(f"Feature importance: {feature_importance}")
        return leaderboard
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Failed to create leaderboard: {exc}")
        raise


def run_model_evaluation(
    model_dir: str, prediction_horizon: int = 10
) -> Dict[str, Any]:
    """Run the model evaluation.
    Returns a dict containing the evaluation results.
    """
    try:
        # 1) Prepare data
        logger.info(
            "Preparing test data with prediction_horizon=%d", prediction_horizon
        )
        label_name = f"Future_Return_{prediction_horizon}D"
        predictor_class = AutoGluonPredictor(model_dir=model_dir)
        predictor_class.load_model_from_mlflow()
        predictor_class._load_metadata()
        data = prepare_ml_data_for_prediction_with_cleaning(
            prediction_horizon=prediction_horizon, days_back=60
        )
        x_test = data.get("x_test")
        y_test = data.get("y_test")
        valid_df = pd.concat([x_test, pd.Series(y_test, name=label_name)], axis=1)
        valid_df = valid_df.reset_index(drop=True)
        initial_rows = len(valid_df)
        valid_df = valid_df.dropna(subset=[label_name])
        dropped_rows = initial_rows - len(valid_df)
        logger.info(
            f"Dropped {dropped_rows} rows with missing values in {label_name} and length: {len(valid_df)}"
        )
        x_test_valid = valid_df.copy()
        y_test_valid = valid_df[label_name]
        # 2) Load predictor
        model = predictor_class.model
        if model is None:
            logger.error("Model not found")
            raise ValueError("Model not found")
        predictor = model.predictor
        if predictor is None:
            logger.error("Predictor not found")
            raise ValueError("Predictor not found")
        feature_names = model.feature_names
        logger.info(f"feature_names: {len(feature_names)}")
        x_test_valid = x_test_valid[feature_names]
        close_price_min = x_test_valid["close"].min()
        close_price_max = x_test_valid["close"].max()
        logger.info(
            f"close_price_min: {close_price_min}, close_price_max: {close_price_max}"
        )

        # 3) Generate and log leaderboard
        model_names = predictor.model_names()
        predictor.persist(models=model_names)

        best_score = 0
        optimal_threshold = None
        for model_name in model_names:
            logger.info(f"Running threshold evaluation for model: {model_name}")
            model.selected_model_name = model_name
            results = model.run_threshold_evaluation(x_test_valid, y_test_valid)
            if results["status"] == "success":
                score = results["best_result"]["test_profit_per_investment"]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_result = results["best_result"]
                    optimal_threshold = best_result["threshold"]

        model.optimal_threshold = optimal_threshold
        logger.info(f"Best model: {best_model_name} with score: {best_score:.2f}")
        logger.info(f"Best result: {best_result}")

        # Save optimal threshold and best model name to JSON
        metadata_path = os.path.join(model_dir, "best_model_metadata.json")
        metadata = {
            "optimal_threshold": optimal_threshold,
            "best_model_name": best_model_name,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to: {metadata_path}")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    # Narrowly ignore common user warnings from Autogluon modules
    for lg in ("autogluon", "autogluon.tabular", "autogluon.common", "autogluon.core"):
        logging.getLogger(lg).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"autogluon.*")
    try:
        model_dir = "AutogluonModels/ag-20251003_214902"
        prediction_horizon = 5
        run_model_evaluation(model_dir, prediction_horizon)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
