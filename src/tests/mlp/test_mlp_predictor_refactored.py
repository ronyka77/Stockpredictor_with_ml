"""
Test file for refactored MLP predictor preprocessing.

This module tests the updated predict() method to ensure it uses
MLPDataUtils.validate_and_clean_data() instead of duplicated preprocessing logic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.time_series.mlp.mlp_predictor import MLPPredictor


class TestMLPPredictorRefactored:
    """Test class for refactored MLP predictor preprocessing."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.X_test = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, np.nan, 5.0],
            'feature2': [0.1, 0.2, np.inf, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        
        # Create a real predictor and attach a stub model
        self.predictor = MLPPredictor(model_name="test_mlp")
        self.predictor.model = stub_model = MagicMock()
        # ensure callable mock returns a tensor of appropriate shape
        stub_model.return_value = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
        stub_model.eval.return_value = None
        self.predictor.device = 'cpu'

    def test_predict_uses_validate_and_clean_data(self):
        """Test that predict() method uses MLPDataUtils.validate_and_clean_data()."""
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data') as mock_validate:
            # Mock the validate_and_clean_data method
            mock_validate.return_value = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 0.0, 5.0],
                'feature2': [0.1, 0.2, 0.0, 0.4, 0.5],
                'feature3': [10, 20, 30, 40, 50]
            })
            
            # Call predict method
            result = self.predictor.predict(self.X_test)
            
            # Verify that validate_and_clean_data was called
            mock_validate.assert_called_once_with(self.X_test)
            
            # Verify result is numpy array
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

    def test_predict_with_scaler_available(self):
        """Test predict() method when scaler is available."""
        # Mock scaler
        mock_scaler = MagicMock()
        self.predictor.scaler = mock_scaler
        
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.scale_data') as mock_scale:
            # Mock the scale_data method
            mock_scale.return_value = (pd.DataFrame({
                'feature1': [0.1, 0.2, 0.3, 0.0, 0.5],
                'feature2': [0.01, 0.02, 0.0, 0.04, 0.05],
                'feature3': [1, 2, 3, 4, 5]
            }), mock_scaler)
            
            # Call predict method
            result = self.predictor.predict(self.X_test)
            
            # Verify that scale_data was called with correct parameters
            mock_scale.assert_called_once_with(self.X_test, mock_scaler, False)
            
            # Verify result is numpy array
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

    def test_predict_fallback_preprocessing(self):
        """Test predict() method fallback preprocessing when no scaler available."""
        # Ensure no scaler is set
        self.predictor.scaler = None
        
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data') as mock_validate:
            # Mock the validate_and_clean_data method
            mock_validate.return_value = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 0.0, 5.0],
                'feature2': [0.1, 0.2, 0.0, 0.4, 0.5],
                'feature3': [10, 20, 30, 40, 50]
            })
            
            # Call predict method
            result = self.predictor.predict(self.X_test)
            
            # Verify that validate_and_clean_data was called
            mock_validate.assert_called_once_with(self.X_test)
            
            # Verify result is numpy array
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

    def test_predict_exception_handling(self):
        """Test predict() method exception handling with fallback preprocessing."""
        # Ensure no scaler is set
        self.predictor.scaler = None
        
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data') as mock_validate:
            # Mock the validate_and_clean_data method to raise an exception first, then return clean data
            mock_validate.side_effect = [ValueError("Test error"), pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 0.0, 5.0],
                'feature2': [0.1, 0.2, 0.0, 0.4, 0.5],
                'feature3': [10, 20, 30, 40, 50]
            })]
            
            # Call predict method
            result = self.predictor.predict(self.X_test)
            
            # Verify that validate_and_clean_data was called twice (once in try, once in except)
            assert mock_validate.call_count == 2
            
            # Verify result is numpy array
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

    def test_predict_with_nan_inf_data(self):
        """Test predict() method with data containing NaN/Inf values."""
        # Create data with NaN/Inf values
        X_with_nan_inf = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, np.inf, 5.0],
            'feature2': [0.1, 0.2, -np.inf, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data') as mock_validate:
            # Mock the validate_and_clean_data method to return cleaned data
            mock_validate.return_value = pd.DataFrame({
                'feature1': [1.0, 0.0, 3.0, 0.0, 5.0],
                'feature2': [0.1, 0.2, 0.0, 0.4, 0.5],
                'feature3': [10, 20, 30, 40, 50]
            })
            
            # Call predict method
            result = self.predictor.predict(X_with_nan_inf)
            
            # Verify that validate_and_clean_data was called with the data containing NaN/Inf
            mock_validate.assert_called_once_with(X_with_nan_inf)
            
            # Verify result is numpy array
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

    def test_predict_normalization_applied(self):
        """Test that basic normalization is applied after validate_and_clean_data()."""
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data') as mock_validate:
            # Mock the validate_and_clean_data method
            cleaned_data = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
                'feature3': [10, 20, 30, 40, 50]
            })
            mock_validate.return_value = cleaned_data
            
            # Call predict method
            result = self.predictor.predict(self.X_test)
            
            # Verify that validate_and_clean_data was called
            mock_validate.assert_called_once_with(self.X_test)
            
            # Verify result is numpy array
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

    def test_predict_model_not_trained_error(self):
        """Test that predict() raises error when model is not trained."""
        # Set model to None to simulate untrained model
        self.predictor.model = None
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            self.predictor.predict(self.X_test)

    def test_predict_output_shape(self):
        """Test that predict() returns correct output shape."""
        with patch('src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data') as mock_validate:
            # Mock the validate_and_clean_data method
            mock_validate.return_value = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 0.0, 5.0],
                'feature2': [0.1, 0.2, 0.0, 0.4, 0.5],
                'feature3': [10, 20, 30, 40, 50]
            })
            
            # Mock model to return 2D tensor
            self.predictor.model.return_value = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
            
            # Call predict method
            result = self.predictor.predict(self.X_test)
            
            # Verify output is 1D (flattened)
            assert result.ndim == 1
            assert len(result) == 5 