"""
XGBoost Model Predictor

This module loads trained XGBoost models from MLflow and makes predictions
on the most recent data, saving results to Excel files.
"""

from src.models.gradient_boosting.xgboost_model import XGBoostModel
from src.models.predictors.base_predictor import BasePredictor

class XGBoostPredictor(BasePredictor):
    """
    XGBoost Model Predictor for loading models from MLflow and making predictions
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the predictor with a specific MLflow run ID
        
        Args:
            run_id: MLflow run ID to load the model from
        """
        super().__init__(run_id=run_id, model_type='xgboost')
        
    def load_model_from_mlflow(self) -> None:
        """
        Load the XGBoost model from MLflow using the specified run ID
        """
        self.model = XGBoostModel.load_from_mlflow(self.run_id)
        print(f"✅ XGBoost model loaded: {self.model.model_name}")
        print(f"   Expected features: {len(self.model.feature_names) if self.model.feature_names else 'Unknown'}")
    
def main():
    """
    Main function for standalone prediction
    """
    run_id = "38c5cfb19df14285aba5d2136c890058"
    days_back = 15

    predictor = XGBoostPredictor(run_id=run_id)
    output_file = predictor.run_prediction_pipeline(days_back=days_back)
    
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    main() 