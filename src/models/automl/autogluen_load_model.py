from typing import Any, Dict
import os
import json
import pandas as pd
from datetime import datetime

from autogluon.tabular import TabularPredictor

from src.utils.logger import get_logger
from src.models.common.training_data_prep import prepare_common_training_data
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_prediction_with_cleaning
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.predictors.autogluon_predictor import AutoGluonPredictor


logger = get_logger(__name__)


def load_autogluon_predictor(model_dir: str) -> TabularPredictor:
    """Load a fitted Autogluon TabularPredictor from disk.
    Raises FileNotFoundError when the directory does not exist.
    """
    logger.info("Loading Autogluon predictor from '%s'", model_dir)
    if not os.path.exists(model_dir):
        logger.error("Model directory not found: %s", model_dir)
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    predictor = TabularPredictor.load(model_dir)
    logger.info("Loaded predictor. Problem type: %s", getattr(predictor, "problem_type", "unknown"))
    return predictor

def generate_and_log_leaderboard(predictor: TabularPredictor, valid_df: pd.DataFrame) -> pd.DataFrame:
    """Generate the Autogluon leaderboard using valid_df and log it.
    Returns the leaderboard DataFrame.
    """
    try:
        logger.info("Generating leaderboard on valid_df (n=%d)", len(valid_df))
        leaderboard = predictor.leaderboard(valid_df, silent=True, extra_metrics=['mape', 'mse', 'mae', 'rmse', 'r2'])
        logger.info("Autogluon leaderboard:\n%s", leaderboard.to_string())
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        leaderboard.to_excel(f"pipeline_stats/leaderboard_{timestamp}.xlsx", index=False)

        feature_importance = predictor.feature_importance(valid_df)
        feature_importance.to_excel(f"pipeline_stats/feature_importance_{timestamp}.xlsx")
        logger.info(f"Feature importance: {feature_importance}")
        return leaderboard
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to create leaderboard: %s", exc)
        raise

def load_model_and_log(model_dir: str, prediction_horizon: int = 1) -> Dict[str, Any]:
    """Load training data, load the Autogluon model and log leaderboard computed on valid_df.
    Returns a dict containing the predictor, leaderboard, X_test and y_test.
    """
    # 1) Prepare data
    logger.info("Preparing test data with prediction_horizon=%d", prediction_horizon)
    data = prepare_common_training_data(prediction_horizon=prediction_horizon)
    X_test: pd.DataFrame = data.get("X_test")
    y_test = data.get("y_test")
    valid_df = pd.concat([X_test, pd.Series(y_test, name='Future_Return_10')], axis=1)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)

    # 2) Load predictor
    ag_model_wrapper = AutoGluonModel.load_from_dir(model_dir)
    predictor = ag_model_wrapper.predictor

    # 3) Generate and log leaderboard
    leaderboard = generate_and_log_leaderboard(predictor, valid_df)

    return {
        "predictor": predictor,
        "leaderboard": leaderboard,
        "valid_df": valid_df,
    }

def make_prediction(model_dir: str, prediction_horizon: int = 10) -> Dict[str, Any]:
    """Make a prediction using the Autogluon model.
    Returns a dict containing the prediction, X_test and y_test.
    """
    data = prepare_ml_data_for_prediction_with_cleaning(prediction_horizon=prediction_horizon)
    X_test: pd.DataFrame = data.get("X_test")
    y_test = data.get("y_test")
    valid_df = pd.concat([X_test, pd.Series(y_test, name='Future_Return_10')], axis=1)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    predictor = load_autogluon_predictor(model_dir)
    prediction = predictor.predict(valid_df)
    return {
        "prediction": prediction,
        "X_test": X_test,
        "y_test": y_test,
    }

def run_model_evaluation(model_dir: str, prediction_horizon: int = 10) -> Dict[str, Any]:
    """Run the model evaluation.
    Returns a dict containing the evaluation results.
    """
    try:
        # 1) Prepare data
        logger.info("Preparing test data with prediction_horizon=%d", prediction_horizon)
        predictor_class = AutoGluonPredictor(model_dir=model_dir)
        predictor_class.load_model_from_mlflow()
        predictor_class._load_metadata()
        features_df, metadata_df = predictor_class.load_recent_data(days_back=60)
        X_test = features_df.copy()
        y_test = metadata_df['target_values'].copy()
        valid_df = pd.concat([X_test, pd.Series(y_test, name='Future_Return_10')], axis=1)
        valid_df = valid_df.sample(frac=1).reset_index(drop=True)
        valid_df = valid_df.dropna(subset=['Future_Return_10'])
        X_test_valid = valid_df.drop(columns=['Future_Return_10'])
        y_test_valid = valid_df['Future_Return_10']
        # 2) Load predictor
        model = predictor_class.model
        predictor = model.predictor
        X_test_valid = X_test_valid[model.feature_names]

        # 3) Generate and log leaderboard
        leaderboard = generate_and_log_leaderboard(predictor, valid_df)

        model_names = predictor.model_names()
        predictor.persist(models=model_names)
        best_score = 0
        optimal_threshold = None
        for model_name in model_names:
            logger.info(f"Running threshold evaluation for model: {model_name}")
            model.selected_model_name = model_name
            results = model.run_threshold_evaluation(X_test_valid, y_test_valid)
            if results['status'] == 'success':
                score = results['best_result']['test_profit_per_investment']
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_result = results['best_result']
                    optimal_threshold = best_result['threshold']
        
        model.optimal_threshold = optimal_threshold
        logger.info(f"Best model: {best_model_name} with score: {best_score:.2f}")
        logger.info(f"Best result: {best_result}")
        # Save optimal threshold and best model name to JSON
        metadata_path = os.path.join(model_dir, "best_model_metadata.json")
        metadata = {
            "optimal_threshold": optimal_threshold,
            "best_model_name": best_model_name
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to: {metadata_path}")

        # 4) Delete models
        models_to_delete = [model_name for model_name in model_names if model_name != best_model_name]
        predictor.delete_models(models_to_delete=models_to_delete, delete_from_disk=True)
        
    except Exception as e:
        logger.error("Script failed: %s", e)
        raise

if __name__ == "__main__":
    try:
        model_dir = "AutogluonModels/ag-20250823_214827"
        prediction_horizon = 10
        run_model_evaluation(model_dir, prediction_horizon)
    except Exception as e:
        logger.error("Script failed: %s", e)
        raise
        


