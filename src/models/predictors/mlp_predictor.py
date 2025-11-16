"""
MLP Model Predictor

This module loads trained MLP models from MLflow and makes predictions
on the most recent data, saving results to Excel files.
"""

import pandas as pd
from src.models.time_series.mlp.mlp_main import MLPPredictorWithMLflow
from src.models.predictors.base_predictor import BasePredictor
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


class MLPPredictorWrapper(BasePredictor):
    """
    MLP Model Predictor for loading models from MLflow and making predictions
    """

    def __init__(self, run_id: str):
        """
        Initialize the predictor with a specific MLflow run ID

        Args:
            run_id: MLflow run ID to load the model from
        """
        super().__init__(run_id=run_id, model_type="mlp")

    def load_model_from_mlflow(self) -> None:
        """
        Load the MLP model from MLflow using the specified run ID and model ID
        """
        # Create MLPPredictorWithMLflow instance (single instance for entire pipeline)
        self.model = MLPPredictorWithMLflow(
            model_name="mlp_prediction_pipeline",
            config={"input_size": 0},  # Will be updated from MLflow
        )

        # Load the model from MLflow using only the run_id (runs URI)
        success = self.model.load_model(self.run_id)

        if not success:
            raise RuntimeError(f"Failed to load MLP model from MLflow run {self.run_id}")

        # Set model to evaluation mode for prediction
        if hasattr(self.model, "model") and self.model.model is not None:
            self.model.model.eval()

        logger.info(f"MLP model loaded: {self.model.model_name}")
        logger.info(
            f"Expected features: {len(self.model.feature_names) if hasattr(self.model, 'feature_names') and self.model.feature_names else 'Unknown'}"
        )
        logger.info(f"Input size: {self.model.config.get('input_size', 'Unknown')}")

        # Check if scaler was loaded
        if hasattr(self.model, "scaler") and self.model.scaler is not None:
            logger.info("Scaler loaded from MLflow artifacts")
        else:
            logger.info("ℹ️ No scaler found - will use raw features")

        # Log single instance architecture confirmation
        logger.info(
            "Single Instance Architecture: Using one MLPPredictorWithMLflow for prediction pipeline"
        )

    def _reorder_features_for_inference(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure incoming features strictly match the saved feature_names order.
        """
        if not hasattr(self.model, "feature_names") or not self.model.feature_names:
            return features_df
        missing = [c for c in self.model.feature_names if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for inference: {missing}")
        # Strict order and dtype
        return features_df.loc[:, self.model.feature_names].astype("float32")


def main():
    """
    Main function for standalone prediction using single instance architecture
    """
    run_id = "101e66e2564f402ebe95a15a3b4094e2"
    days_back = 30

    # Create predictor with single instance architecture
    predictor = MLPPredictorWrapper(run_id=run_id)

    # Load model and display information
    predictor.load_model_from_mlflow()

    # Run prediction pipeline
    output_file = predictor.run_prediction_pipeline(days_back=days_back)
    logger.info(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
