"""
Integration tests for dividend features in indicator pipeline.
"""

import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock

from src.data_collector.indicator_pipeline.indicator_pipeline import (
    BatchFeatureProcessor,
    BatchJobConfig,
)


class TestDividendIntegration:
    """Integration tests for dividend features in indicator pipeline."""

    @patch("src.data_collector.polygon_data.data_storage.DataStorage.load_dividends_for_ticker")
    @patch("src.feature_engineering.data_loader.StockDataLoader.load_stock_data")
    @patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator.calculate_all_features"
    )
    @patch("src.data_collector.indicator_pipeline.feature_storage.FeatureStorage.save_features")
    def test_dividend_features_included_in_processing(
        self, mock_save_features, mock_calculate_features, mock_load_stock_data, mock_load_dividends
    ):
        """Test that dividend features are included in the processing pipeline."""
        # Setup mocks
        # Mock stock data
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        stock_data = pd.DataFrame(
            {
                "close": [100.0] * len(dates),
                "open": [99.0] * len(dates),
                "high": [101.0] * len(dates),
                "low": [98.0] * len(dates),
                "volume": [1000] * len(dates),
            },
            index=dates,
        )
        mock_load_stock_data.return_value = stock_data

        # Mock regular features result
        mock_feature_result = MagicMock()
        mock_feature_result.data = pd.DataFrame(
            {"sma_20": [100.0] * len(dates), "rsi_14": [50.0] * len(dates)}, index=dates
        )
        mock_feature_result.metadata = {}
        mock_feature_result.quality_score = 95.0
        mock_feature_result.warnings = []
        mock_calculate_features.return_value = mock_feature_result

        # Mock dividend data
        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 5)],
                "cash_amount": [2.0],
                "pay_date": [date(2023, 1, 15)],
                "record_date": [date(2023, 1, 6)],
                "id": ["test-123"],
                "currency": ["USD"],
            }
        )
        mock_load_dividends.return_value = dividends_df

        # Setup processor
        processor = BatchFeatureProcessor()
        config = BatchJobConfig(
            start_date="2023-01-01",
            end_date="2023-01-10",
            min_data_points=5,
            save_to_parquet=True,
            save_to_database=False,
            feature_categories=None,
        )

        # Process ticker
        result = processor.process_single_ticker("AAPL", config, "test-job-123")

        # Verify result
        assert result["success"] is True
        assert result["features_calculated"] > 2  # Should include dividend features

        # Verify dividend features were added to the data
        save_call_args = mock_save_features.call_args
        saved_data = save_call_args[0][1]  # Second argument is the data

        # Check that dividend columns were added
        dividend_columns = [
            "dividend_amount",
            "dividend_yield",
            "is_ex_dividend",
            "days_since_last_dividend",
            "days_to_next_dividend",
        ]
        for col in dividend_columns:
            assert col in saved_data.columns, f"Missing dividend column: {col}"

        # Check dividend values
        jan_5_idx = pd.Timestamp(date(2023, 1, 5))
        assert np.isclose(saved_data.loc[jan_5_idx, "dividend_amount"], 2.0)
        assert np.isclose(saved_data.loc[jan_5_idx, "dividend_yield"], 0.02)  # 2.0 / 100.0
        assert saved_data.loc[jan_5_idx, "is_ex_dividend"] == 1

        # Verify metadata includes dividend info
        save_metadata = save_call_args[0][2]  # Third argument is metadata
        assert save_metadata["dividend_features"] is True
        assert save_metadata["dividend_source"] == "polygon"

    @patch("src.data_collector.polygon_data.data_storage.DataStorage.load_dividends_for_ticker")
    @patch("src.feature_engineering.data_loader.StockDataLoader.load_stock_data")
    @patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator.calculate_all_features"
    )
    @patch("src.data_collector.indicator_pipeline.feature_storage.FeatureStorage.save_features")
    def test_no_dividend_data_available(
        self, mock_save_features, mock_calculate_features, mock_load_stock_data, mock_load_dividends
    ):
        """Test processing when no dividend data is available."""
        # Setup mocks
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        stock_data = pd.DataFrame({"close": [100.0] * len(dates)}, index=dates)
        mock_load_stock_data.return_value = stock_data

        mock_feature_result = MagicMock()
        mock_feature_result.data = pd.DataFrame({"sma_20": [100.0] * len(dates)}, index=dates)
        mock_feature_result.metadata = {}
        mock_feature_result.quality_score = 95.0
        mock_feature_result.warnings = []
        mock_calculate_features.return_value = mock_feature_result

        # Return empty DataFrame for dividends
        mock_load_dividends.return_value = pd.DataFrame(
            columns=["ex_dividend_date", "cash_amount", "id", "currency"]
        )

        # Setup processor
        processor = BatchFeatureProcessor()
        config = BatchJobConfig(
            start_date="2023-01-01",
            end_date="2023-01-05",
            min_data_points=3,
            save_to_parquet=True,
            save_to_database=False,
        )

        # Process ticker
        result = processor.process_single_ticker("TSLA", config, "test-job-456")

        # Verify result
        assert result["success"] is True
        assert result["features_calculated"] == 1  # Only the original feature

        # Verify metadata indicates no dividend features
        save_call_args = mock_save_features.call_args
        save_metadata = save_call_args[0][2]
        assert save_metadata["dividend_features"] is False
        assert save_metadata["dividend_source"] is None

    @patch("src.data_collector.polygon_data.data_storage.DataStorage.load_dividends_for_ticker")
    @patch("src.feature_engineering.data_loader.StockDataLoader.load_stock_data")
    @patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator.calculate_all_features"
    )
    def test_dividend_processing_failure_handling(
        self, mock_calculate_features, mock_load_stock_data, mock_load_dividends
    ):
        """Test that dividend processing failures are handled gracefully."""
        # Setup mocks
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        stock_data = pd.DataFrame({"close": [100.0] * len(dates)}, index=dates)
        mock_load_stock_data.return_value = stock_data

        mock_feature_result = MagicMock()
        mock_feature_result.data = pd.DataFrame({"sma_20": [100.0] * len(dates)}, index=dates)
        mock_feature_result.metadata = {}
        mock_feature_result.quality_score = 95.0
        mock_feature_result.warnings = []
        mock_calculate_features.return_value = mock_feature_result

        # Make dividend loading fail
        mock_load_dividends.side_effect = Exception("Database connection failed")

        # Setup processor
        processor = BatchFeatureProcessor()
        config = BatchJobConfig(
            start_date="2023-01-01",
            end_date="2023-01-05",
            min_data_points=3,
            save_to_parquet=False,
            save_to_database=False,
        )

        # Process ticker
        result = processor.process_single_ticker("MSFT", config, "test-job-789")

        # Verify result - should still succeed but with warnings
        assert result["success"] is True
        assert result["warnings"] > 0  # Should have dividend failure warning

    @patch("src.data_collector.polygon_data.data_storage.DataStorage.load_dividends_for_ticker")
    @patch("src.feature_engineering.data_loader.StockDataLoader.load_stock_data")
    @patch(
        "src.data_collector.indicator_pipeline.feature_calculator.FeatureCalculator.calculate_all_features"
    )
    @patch("src.data_collector.indicator_pipeline.feature_storage.FeatureStorage.save_features")
    def test_dividend_metadata_in_feature_result(
        self, mock_save_features, mock_calculate_features, mock_load_stock_data, mock_load_dividends
    ):
        """Test that dividend metadata is properly set in the feature result."""
        # Setup mocks
        dates = pd.date_range("2023-01-01", "2023-01-03", freq="D")
        stock_data = pd.DataFrame({"close": [100.0] * len(dates)}, index=dates)
        mock_load_stock_data.return_value = stock_data

        mock_feature_result = MagicMock()
        mock_feature_result.data = pd.DataFrame({"sma_20": [100.0] * len(dates)}, index=dates)
        mock_feature_result.metadata = {}  # Start with empty metadata
        mock_feature_result.quality_score = 95.0
        mock_feature_result.warnings = []
        mock_calculate_features.return_value = mock_feature_result

        # Mock dividend data
        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 2)],
                "cash_amount": [1.5],
                "id": ["test-456"],
                "currency": ["USD"],
            }
        )
        mock_load_dividends.return_value = dividends_df

        # Setup processor
        processor = BatchFeatureProcessor()
        config = BatchJobConfig(
            start_date="2023-01-01",
            end_date="2023-01-03",
            min_data_points=2,
            save_to_parquet=True,
            save_to_database=False,
        )

        # Process ticker
        processor.process_single_ticker("NVDA", config, "test-job-999")

        # Verify that metadata was updated
        # Check that dividend metadata was added to the feature result
        # This is verified by checking the metadata in the save call
        save_call_args = mock_save_features.call_args
        save_metadata = save_call_args[0][2]
        assert save_metadata["dividend_features"] is True
        assert save_metadata["dividend_source"] == "polygon"
