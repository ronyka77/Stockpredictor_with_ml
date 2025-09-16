import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.time_series.mlp.mlp_architecture import MLPDataUtils, MLPModule
from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


class TestMLPOptimizationScalerIntegration:
    def setup_method(self):
        """
        Test setup that creates deterministic synthetic train/test datasets and configures an MLP predictor and optimization mixin.
        
        Creates:
        - X_train (DataFrame, 200 rows) with features 'feature1', 'feature2', 'feature3' (standard normal) and 'close' (uniform 50–150).
        - y_train (Series, 200 rows), X_test (DataFrame, 50 rows), y_test (Series, 50 rows) with the same feature distributions.
        Also initializes an MLPPredictor named "test_mlp_optimization" with a simple MLPModule (input_size=4, layer_sizes=[10,5], output_size=1) on CPU, and prepares an MLPOptimizationMixin configured for that predictor.
        """
        rng = np.random.RandomState(42)
        self.X_train = pd.DataFrame(
            {
                "feature1": rng.randn(200),
                "feature2": rng.randn(200),
                "feature3": rng.randn(200),
                "close": rng.uniform(50, 150, 200),
            }
        )
        self.y_train = pd.Series(rng.randn(200))
        self.X_test = pd.DataFrame(
            {
                "feature1": rng.randn(50),
                "feature2": rng.randn(50),
                "feature3": rng.randn(50),
                "close": rng.uniform(50, 150, 50),
            }
        )
        self.y_test = pd.Series(rng.randn(50))

        self.predictor = MLPPredictor(
            model_name="test_mlp_optimization",
            config={"layer_sizes": [10, 5], "input_size": 4, "batch_size": 16},
        )
        self.predictor.model = MLPModule(
            input_size=4, layer_sizes=[10, 5], output_size=1
        )
        self.predictor.device = torch.device("cpu")
        self.predictor.model = self.predictor.model.to("cpu")

        self.optimization_mixin = MLPOptimizationMixin()
        self.optimization_mixin.model_name = "test_mlp_optimization"
        self.optimization_mixin.config = self.predictor.config.copy()
        self.optimization_mixin.threshold_evaluator = None

    def test_prepare_data_for_training_with_scaler(self):
        """Prepare and scale training data, ensuring a StandardScaler is produced for trials."""
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_train)
        X_clean, scaler = MLPDataUtils.scale_data(cleaned, None, True)
        self.optimization_mixin.current_trial_scaler = scaler
        if not isinstance(self.optimization_mixin.current_trial_scaler, StandardScaler):
            raise AssertionError("current_trial_scaler should be a StandardScaler")

    def test_create_model_for_tuning_with_scaler(self):
        """Set a scaler on a trial predictor and verify it's stored on the predictor."""
        trial_predictor = MLPPredictor(
            model_name="test_trial", config={"input_size": 4, "layer_sizes": [10, 5]}
        )
        scaler = StandardScaler()
        trial_predictor.set_scaler(scaler)
        if trial_predictor.scaler is not scaler:
            raise AssertionError("Trial predictor scaler not set correctly")

    def test_objective_function_scaler_integration(self):
        """Objective callable should integrate with provided fitted scaler for trial evaluation."""
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
        if not callable(objective_func):
            raise AssertionError("Objective function should be callable")
