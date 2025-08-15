import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.time_series.mlp.mlp_architecture import MLPDataUtils, MLPModule
from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


class TestMLPOptimizationScalerIntegration:
    def setup_method(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.randn(200),
            "close": np.random.uniform(50, 150, 200),
        })
        self.y_train = pd.Series(np.random.randn(200))
        self.X_test = pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randn(50),
            "close": np.random.uniform(50, 150, 50),
        })
        self.y_test = pd.Series(np.random.randn(50))

        self.predictor = MLPPredictor(model_name="test_mlp_optimization", config={"layer_sizes": [10, 5], "input_size": 4, "batch_size": 16})
        self.predictor.model = MLPModule(input_size=4, layer_sizes=[10, 5], output_size=1)
        self.predictor.device = torch.device("cpu")
        self.predictor.model = self.predictor.model.to("cpu")

        self.optimization_mixin = MLPOptimizationMixin()
        self.optimization_mixin.model_name = "test_mlp_optimization"
        self.optimization_mixin.config = self.predictor.config.copy()
        self.optimization_mixin.threshold_evaluator = None

    def test_prepare_data_for_training_with_scaler(self):
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_train)
        X_clean, scaler = MLPDataUtils.scale_data(cleaned, None, True)
        self.optimization_mixin.current_trial_scaler = scaler
        assert isinstance(self.optimization_mixin.current_trial_scaler, StandardScaler)

    def test_create_model_for_tuning_with_scaler(self):
        trial_predictor = MLPPredictor(model_name="test_trial", config={"input_size": 4, "layer_sizes": [10, 5]})
        scaler = StandardScaler()
        trial_predictor.set_scaler(scaler)
        assert trial_predictor.scaler is scaler

    def test_objective_function_scaler_integration(self):
        cleaned_train = MLPDataUtils.validate_and_clean_data(self.X_train)
        X_train_clean, scaler = MLPDataUtils.scale_data(cleaned_train, None, True)
        cleaned_test = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_test_scaled, _ = MLPDataUtils.scale_data(cleaned_test, scaler, False)

        objective_func = self.optimization_mixin.objective(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            X_test_scaled=X_test_scaled,
            y_test=self.y_test,
            fitted_scaler=scaler,
        )
        assert callable(objective_func)


