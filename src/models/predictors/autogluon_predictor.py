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
# for lg in ("autogluon", "autogluon.tabular", "autogluon.common", "autogluon.core"):
#     logging.getLogger(lg).setLevel(logging.WARNING)
# warnings.filterwarnings("ignore")


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
                    "✅ AutoGluon predictor loaded from AG_MODEL_DIR: %s",
                    self.model_dir,
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to load AutoGluon predictor from AG_MODEL_DIR '%s': %s",
                    self.model_dir,
                    e,
                )
        else:
            logger.warning(
                "AG_MODEL_DIR is set but path does not exist: %s", self.model_dir
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
                    "✅ AutoGluon optimal threshold and best model name loaded: %s, %s",
                    self.optimal_threshold,
                    self.model.selected_model_name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load AutoGluon metadata from AG_MODEL_DIR '%s': %s",
                    self.model_dir,
                    e,
                )
        else:
            logger.warning(
                "AG_MODEL_DIR is set but path does not exist: %s", self.model_dir
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
        logger.info(f"Predicting model: {model_name}")
        features_df = features_df_base.copy()
        metadata_df = metadata_df_base.copy()
        predictor.model.selected_model_name = model_name
        predictions = predictor.make_predictions(features_df)
        predictor.save_predictions_to_excel(
            features_df, metadata_df, predictions, model_name
        )


if __name__ == "__main__":
    model_dir = "AutogluonModels/ag-20250907_125628"
    predict_all_model(model_dir)
