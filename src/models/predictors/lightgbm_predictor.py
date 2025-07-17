"""
LightGBM Model Predictor

This module loads trained LightGBM models from MLflow and makes predictions
on the most recent data, saving results to Excel files.
"""
from src.models.gradient_boosting.lightgbm_model import LightGBMModel
from src.models.predictors.base_predictor import BasePredictor

class LightGBMPredictor(BasePredictor):
    """
    LightGBM Model Predictor for loading models from MLflow and making predictions
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the predictor with a specific MLflow run ID
        
        Args:
            run_id: MLflow run ID to load the model from
        """
        super().__init__(run_id=run_id, model_type='lightgbm')
    
    def load_model_from_mlflow(self) -> None:
        """
        Load the LightGBM model from MLflow using the specified run ID
        """
        self.model = LightGBMModel.load_from_mlflow(self.run_id)
        print(f"✅ LightGBM model loaded: {self.model.model_name}")
        print(f"   Expected features: {len(self.model.feature_names) if self.model.feature_names else 'Unknown'}")


def main():
    """
    Main function for standalone prediction
    """
    run_id = "eab7a07f2734434e810e2c4f0a590197"
    days_back = 30
    predictor = LightGBMPredictor(run_id=run_id)
    output_file = predictor.run_prediction_pipeline(days_back=days_back)
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    main() 