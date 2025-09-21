"""
AutoGluon Predictor

Loads an AutoGluonModel from MLflow (via run_id) and runs the standard
prediction pipeline, exporting to predictions/autogluon/ as Excel.
"""

import os
import json
import warnings
import logging
from src.models.predictors.base_predictor import BasePredictor
from src.models.automl.autogluon_model import AutoGluonModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AutoGluonPredictor(BasePredictor):
    def __init__(self, model_dir: str):
        super().__init__(run_id=model_dir.split("/")[-1], model_type="autogluon")
        self.model_dir = model_dir

    def load_model_from_mlflow(self) -> None:
        if os.path.exists(self.model_dir):
            try:
                self.model = AutoGluonModel()
                self.model.load_from_dir(self.model_dir)
                logger.info(
                    f"✅ AutoGluon predictor loaded from AG_MODEL_DIR: {self.model_dir}"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load AutoGluon predictor from AG_MODEL_DIR '{self.model_dir}': {e}"
                )
        else:
            logger.warning(
                f"AG_MODEL_DIR is set but path does not exist: {self.model_dir}"
            )

    def _load_metadata(self) -> None:
        if os.path.exists(self.model_dir):
            try:
                with open(
                    os.path.join(self.model_dir, "best_model_metadata.json"), "r"
                ) as f:
                    data = json.load(f)

                threshold = data["optimal_threshold"]
                model_name = data["best_model_name"]
                self.model.optimal_threshold = threshold
                self.optimal_threshold = threshold
                self.model.selected_model_name = model_name
                logger.info(
                    f"✅ AutoGluon optimal threshold and best model name loaded: {self.optimal_threshold}, {self.model.selected_model_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load AutoGluon metadata from AG_MODEL_DIR '{self.model_dir}': {e}"
                )
        else:
            logger.warning(
                f"AG_MODEL_DIR is set but path does not exist: {self.model_dir}"
            )


def predict_best_model(model_dir: str):
    predictor = AutoGluonPredictor(model_dir=model_dir)
    predictor.load_model_from_mlflow()
    predictor._load_metadata()
    features_df, metadata_df = predictor.load_recent_data(days_back=30)
    predictions = predictor.make_predictions(features_df)
    predictor.save_predictions_to_excel(features_df, metadata_df, predictions)


def predict_all_model(model_dir: str):
    predictor = AutoGluonPredictor(model_dir=model_dir)
    predictor.load_model_from_mlflow()
    predictor._load_metadata()
    features_df_base, metadata_df_base = predictor.load_recent_data(days_back=30)
    model_names = predictor.model.predictor.model_names()
    for model_name in model_names:
        logger.info(
            f"Predicting model: {model_name} with AG_MODEL_DIR: {predictor.model_dir} and optimal threshold: {predictor.optimal_threshold}"
        )
        features_df = features_df_base.copy()
        metadata_df = metadata_df_base.copy()
        predictor.model.selected_model_name = model_name
        predictions = predictor.make_predictions(features_df)
        predictor.save_predictions_to_excel(
            features_df, metadata_df, predictions, model_name
        )


def predict_all_model_folders():
    """
    Loops through all model folders in AutogluonModels directory and runs predictions for each.
    """
    base_dir = "AutogluonModels"

    if not os.path.exists(base_dir):
        logger.error(f"AutogluonModels directory not found at {base_dir}")
        return

    model_folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    if not model_folders:
        logger.warning(f"No model folders found in {base_dir}")
        return

    for folder in model_folders:
        model_dir = os.path.join(base_dir, folder)
        logger.info(f"Processing model folder: {folder}")
        try:
            predict_all_model(model_dir)
        except Exception as e:
            logger.error(f"Failed to process {folder}: {str(e)}")
            continue


if __name__ == "__main__":
    for lg in ("autogluon", "autogluon.tabular", "autogluon.common", "autogluon.core"):
        logging.getLogger(lg).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

    model_dir = "AutogluonModels/ag-20250920_160248"
    # predict_all_model(model_dir)
    predict_all_model_folders()
