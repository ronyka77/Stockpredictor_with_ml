"""
Test MLP Scaler Implementation

This test file verifies that the MLP predictor scaler implementation works correctly,
testing both scenarios with and without scaler availability.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.models.time_series.mlp.mlp_architecture import MLPDataUtils, MLPModule
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


class TestMLPScalerImplementation:
    """Test class for MLP scaler implementation."""

    def setup_method(self):
        """Set up test data and model."""
        # Create test data
        np.random.seed(42)
        self.X_test = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )

        # Add some NaN and Inf values to test cleaning
        self.X_test.loc[0, "feature1"] = np.nan
        self.X_test.loc[1, "feature2"] = np.inf
        self.X_test.loc[2, "feature3"] = -np.inf

        # Create a simple MLP model for testing
        self.predictor = MLPPredictor(
            model_name="test_mlp",
            config={"layer_sizes": [10, 5], "input_size": 3, "batch_size": 16},
        )

        # Create a simple model for testing using MLPModule
        self.predictor.model = MLPModule(
            input_size=3, layer_sizes=[10, 5], output_size=1
        )

        # Move model to CPU for testing to avoid device issues
        self.predictor.device = torch.device("cpu")
        self.predictor.model = self.predictor.model.to("cpu")

    def test_validate_and_clean_data_method(self):
        """Test the validate_and_clean_data method from MLPDataUtils."""
        # Test with new scaler (training scenario)
        # First clean data, then fit scaler using scale_data
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_clean, scaler = MLPDataUtils.scale_data(cleaned, None, True)

        # Verify data is cleaned
        assert not X_clean.isnull().any().any()
        assert not np.isinf(X_clean.values).any()
        assert scaler is not None

        # Test with existing scaler (prediction scenario)
        cleaned2 = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_clean2, _ = MLPDataUtils.scale_data(cleaned2, scaler, False)

        # Verify data is cleaned consistently
        assert not X_clean2.isnull().any().any()
        assert not np.isinf(X_clean2.values).any()

    def test_predictor_with_scaler(self):
        """Test predictor with scaler set."""
        # Clean the data first before fitting scaler
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_clean, _ = MLPDataUtils.scale_data(cleaned, None, True)

        # Create and fit a scaler on cleaned data
        scaler = StandardScaler()
        scaler.fit(X_clean)

        # Set scaler on predictor
        self.predictor.set_scaler(scaler)

        # Test prediction
        predictions = self.predictor.predict(self.X_test)

        # Verify predictions are made
        assert predictions is not None
        assert len(predictions) == len(self.X_test)
        assert not np.isnan(predictions).any()

    def test_predictor_without_scaler(self):
        """Test predictor without scaler (fallback scenario)."""
        # Ensure no scaler is set
        if hasattr(self.predictor, "scaler"):
            delattr(self.predictor, "scaler")

        # Test prediction (should use fallback preprocessing)
        predictions = self.predictor.predict(self.X_test)

        # Verify predictions are made
        assert predictions is not None
        assert len(predictions) == len(self.X_test)
        assert not np.isnan(predictions).any()

    def test_confidence_calculation_with_scaler(self):
        """Test confidence calculation with scaler."""
        # Clean the data first before fitting scaler
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_clean, _ = MLPDataUtils.scale_data(cleaned, None, True)

        # Create and fit a scaler on cleaned data
        scaler = StandardScaler()
        scaler.fit(X_clean)

        # Set scaler on predictor
        self.predictor.set_scaler(scaler)

        # Test confidence calculation
        confidence = self.predictor.get_prediction_confidence(
            self.X_test, method="variance"
        )

        # Verify confidence scores are calculated
        assert confidence is not None
        assert len(confidence) == len(self.X_test)
        assert not np.isnan(confidence).any()
        assert np.all(confidence >= 0)  # Confidence should be non-negative

    def test_confidence_calculation_without_scaler(self):
        """Test confidence calculation without scaler (fallback scenario)."""
        # Ensure no scaler is set
        if hasattr(self.predictor, "scaler"):
            delattr(self.predictor, "scaler")

        # Test confidence calculation (should use fallback preprocessing)
        confidence = self.predictor.get_prediction_confidence(
            self.X_test, method="variance"
        )

        # Verify confidence scores are calculated
        assert confidence is not None
        assert len(confidence) == len(self.X_test)
        assert not np.isnan(confidence).any()
        assert np.all(confidence >= 0)  # Confidence should be non-negative

    def test_set_scaler_method(self):
        """Test the set_scaler method."""
        scaler = StandardScaler()

        # Set scaler
        self.predictor.set_scaler(scaler)

        # Verify scaler is set
        assert hasattr(self.predictor, "scaler")
        assert self.predictor.scaler is scaler

    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        # Create invalid data with all NaN values
        invalid_data = pd.DataFrame(
            {
                "feature1": [np.nan] * 10,
                "feature2": [np.nan] * 10,
                "feature3": [np.nan] * 10,
            }
        )

        # Test that validate_and_clean_data handles this gracefully
        cleaned_invalid = MLPDataUtils.validate_and_clean_data(invalid_data)
        X_clean, scaler = MLPDataUtils.scale_data(cleaned_invalid, None, True)

        # Verify data is cleaned
        assert not X_clean.isnull().any().any()
        assert scaler is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
