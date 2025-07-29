"""
LSTM Model Predictor

This module loads trained LSTM models from MLflow and makes predictions
on the most recent data, saving results to Excel files.
"""
from src.models.time_series.lstm_model import LSTMPredictor
from src.models.predictors.base_predictor import BasePredictor

class LSTMPredictorWrapper(BasePredictor):
    """
    LSTM Model Predictor for loading models from MLflow and making predictions
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the predictor with a specific MLflow run ID
        
        Args:
            run_id: MLflow run ID to load the model from
        """
        super().__init__(run_id=run_id, model_type='lstm')
    
    def load_model_from_mlflow(self) -> None:
        """
        Load the LSTM model from MLflow using the specified run ID
        """
        self.model = LSTMPredictor.load_from_mlflow(self.run_id)
        print(f"✅ LSTM model loaded: {self.model.model_name}")
        print(f"   Expected features: {len(self.model.feature_names) if self.model.feature_names else 'Unknown'}")
        print(f"   Sequence length: {getattr(self.model, 'sequence_length', 'Unknown')}")


def main():
    """
    Main function for standalone prediction
    """
    # You'll need to replace this with an actual LSTM model run ID
    run_id = "4441699bd65f41f4a2ce9ef7cb3f0e6b"
    days_back = 30
    predictor = LSTMPredictorWrapper(run_id=run_id)
    output_file = predictor.run_prediction_pipeline(days_back=days_back)
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    main() 