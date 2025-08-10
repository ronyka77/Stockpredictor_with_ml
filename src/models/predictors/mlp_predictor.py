"""
MLP Model Predictor

This module loads trained MLP models from MLflow and makes predictions
on the most recent data, saving results to Excel files.
"""
from src.models.time_series.mlp.mlp_main import MLPPredictorWithMLflow
from src.models.predictors.base_predictor import BasePredictor

class MLPPredictorWrapper(BasePredictor):
    """
    MLP Model Predictor for loading models from MLflow and making predictions
    """
    
    def __init__(self, run_id: str, model_id: str = None):
        """
        Initialize the predictor with a specific MLflow run ID and optional model ID
        
        Args:
            run_id: MLflow run ID to load the model from
            model_id: Optional model ID for specific model loading
        """
        super().__init__(run_id=run_id, model_type='mlp')
        self.model_id = model_id
    
    def load_model_from_mlflow(self) -> None:
        """
        Load the MLP model from MLflow using the specified run ID and model ID
        """
        # Create MLPPredictorWithMLflow instance (single instance for entire pipeline)
        self.model = MLPPredictorWithMLflow(
            model_name="mlp_prediction_pipeline",
            config={'input_size': 0}  # Will be updated from MLflow
        )
        
        # Load the model from MLflow
        success = self.model.load_model(self.run_id, model_id=self.model_id)
        
        if not success:
            raise RuntimeError(f"Failed to load MLP model from MLflow run {self.run_id}")
        
        # Set model to evaluation mode for prediction
        if hasattr(self.model, 'model') and self.model.model is not None:
            self.model.model.eval()
        
        print(f"✅ MLP model loaded: {self.model.model_name}")
        print(f"   Expected features: {len(self.model.feature_names) if hasattr(self.model, 'feature_names') and self.model.feature_names else 'Unknown'}")
        print(f"   Input size: {self.model.config.get('input_size', 'Unknown')}")
        
        # Check if scaler was loaded
        if hasattr(self.model, 'scaler') and self.model.scaler is not None:
            print("   ✅ Scaler loaded from MLflow artifacts")
        else:
            print("   ℹ️ No scaler found - will use raw features")
        
        # Log single instance architecture confirmation
        print("   🔄 Single Instance Architecture: Using one MLPPredictorWithMLflow for prediction pipeline")
    
    def get_prediction_with_confidence(self, features_df, confidence_method='variance'):
        """
        Get predictions with confidence scores using the single instance architecture
        
        Args:
            features_df: DataFrame with features for prediction
            confidence_method: Method for confidence calculation ('variance', 'simple', 'margin')
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_from_mlflow() first.")
        
        # Use the single instance's prediction and confidence methods
        predictions = self.model.predict(features_df)
        confidence_scores = self.model.get_prediction_confidence(features_df, method=confidence_method)
        
        return predictions, confidence_scores


def main():
    """
    Main function for standalone prediction using single instance architecture
    """
    run_id = "83df1178e2c94ebd84d288de8510e48c"
    model_id = "m-15b29e83c9724627b81a4bae5bb4bd7e"
    days_back = 30
    
    # Create predictor with single instance architecture
    predictor = MLPPredictorWrapper(run_id=run_id, model_id=model_id)
    
    # Load model and display information
    predictor.load_model_from_mlflow()
    
    # Run prediction pipeline
    output_file = predictor.run_prediction_pipeline(days_back=days_back)
    print(f"✅ Predictions saved to: {output_file}")
    

if __name__ == "__main__":
    main() 