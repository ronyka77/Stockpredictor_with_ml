"""
Unit tests for dividend feature calculation functions.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import date

from src.data_collector.indicator_pipeline.dividend_features import compute_dividend_features


class TestComputeDividendFeatures:
    """Test cases for compute_dividend_features function."""

    def test_basic_dividend_features_single_date(self):
        """Test basic dividend feature calculation with single ex-dividend date."""
        # Create test price data
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        price_df = pd.DataFrame(
            {
                "close": [100.0] * len(dates),
                "open": [99.0] * len(dates),
                "high": [101.0] * len(dates),
                "low": [98.0] * len(dates),
                "volume": [1000] * len(dates),
            },
            index=dates,
        )

        # Create test dividend data
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

        result = compute_dividend_features(price_df, dividends_df)

        # Check result shape
        assert len(result) == len(price_df)
        assert list(result.index) == list(price_df.index)

        # Check dividend features
        assert np.isclose(result.loc[dates[4], "dividend_amount"], 2.0)  # Jan 5
        assert np.isclose(result.loc[dates[4], "dividend_yield"], 0.02)  # 2.0 / 100.0
        assert result.loc[dates[4], "is_ex_dividend"] == 1

        # Check non-ex-dividend dates
        for i, d in enumerate(dates):
            if i != 4:  # Not Jan 5
                assert pd.isna(result.loc[d, "dividend_amount"])
                assert pd.isna(result.loc[d, "dividend_yield"])
                assert result.loc[d, "is_ex_dividend"] == 0

    def test_multiple_dividends_same_date(self):
        """Test handling of multiple dividends on same ex-dividend date."""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        price_df = pd.DataFrame({"close": [100.0] * len(dates)}, index=dates)

        # Two dividends on same date
        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 3), date(2023, 1, 3)],
                "cash_amount": [1.0, 1.5],
                "id": ["test-1", "test-2"],
                "currency": ["USD", "USD"],
            }
        )

        result = compute_dividend_features(price_df, dividends_df)

        # Should sum the amounts
        assert np.isclose(result.loc[dates[2], "dividend_amount"], 2.5)  # 1.0 + 1.5
        assert np.isclose(result.loc[dates[2], "dividend_yield"], 0.025)  # 2.5 / 100.0

    def test_ex_date_alignment_prior_day(self):
        """Test ex-dividend date mapping to nearest prior trading day."""
        # Create price data with weekend gaps
        dates = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 6)]  # Mon, Tue, Fri
        price_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=pd.DatetimeIndex(dates))

        # Ex-dividend date falls on Saturday (non-trading day)
        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 7)],  # Saturday
                "cash_amount": [1.0],
                "id": ["test-123"],
                "currency": ["USD"],
            }
        )

        result = compute_dividend_features(price_df, dividends_df)

        # Should map to Friday (nearest prior trading day)
        friday_idx = pd.Timestamp(date(2023, 1, 6))
        assert np.isclose(result.loc[friday_idx, "dividend_amount"], 1.0)
        assert result.loc[friday_idx, "is_ex_dividend"] == 1

    def test_missing_close_price(self):
        """Test handling of missing close price on ex-dividend date."""
        dates = pd.date_range("2023-01-01", "2023-01-03", freq="D")
        price_df = pd.DataFrame(
            {
                "close": [100.0, np.nan, 102.0]  # Missing close on Jan 2
            },
            index=dates,
        )

        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 2)],
                "cash_amount": [1.0],
                "id": ["test-123"],
                "currency": ["USD"],
            }
        )

        result = compute_dividend_features(price_df, dividends_df)

        # Should set dividend_amount but NaN yield due to missing close
        assert np.isclose(result.loc[dates[1], "dividend_amount"], 1.0)
        assert pd.isna(result.loc[dates[1], "dividend_yield"])
        assert result.loc[dates[1], "is_ex_dividend"] == 1

    def test_days_since_and_to_next_dividend(self):
        """Test calculation of days since last and to next dividend."""
        dates = pd.date_range("2023-01-01", "2023-01-15", freq="D")
        price_df = pd.DataFrame({"close": [100.0] * len(dates)}, index=dates)

        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 5), date(2023, 1, 12)],
                "cash_amount": [1.0, 1.5],
                "id": ["test-1", "test-2"],
                "currency": ["USD", "USD"],
            }
        )

        result = compute_dividend_features(price_df, dividends_df)

        # Check days_since_last_dividend
        # Jan 1-4: NaN (before first dividend)
        for d in dates[:4]:
            assert pd.isna(result.loc[d, "days_since_last_dividend"])

        # Jan 6-11: days since Jan 5
        assert result.loc[dates[5], "days_since_last_dividend"] == 1  # Jan 6 - Jan 5 = 1
        assert result.loc[dates[10], "days_since_last_dividend"] == 6  # Jan 11 - Jan 5 = 6

        # Check days_to_next_dividend
        # Jan 13-15: NaN (after last dividend)
        for d in dates[12:]:
            assert pd.isna(result.loc[d, "days_to_next_dividend"])

        # Jan 6-11: days until Jan 12
        assert result.loc[dates[5], "days_to_next_dividend"] == 6  # Jan 12 - Jan 6 = 6
        assert result.loc[dates[10], "days_to_next_dividend"] == 1  # Jan 12 - Jan 11 = 1

    def test_empty_dividends(self):
        """Test handling of empty dividend DataFrame."""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        price_df = pd.DataFrame({"close": [100.0] * len(dates)}, index=dates)

        dividends_df = pd.DataFrame(columns=["ex_dividend_date", "cash_amount", "id", "currency"])

        result = compute_dividend_features(price_df, dividends_df)

        # All features should be NaN or 0
        assert all(pd.isna(result["dividend_amount"]))
        assert all(pd.isna(result["dividend_yield"]))
        assert all(pd.isna(result["days_since_last_dividend"]))
        assert all(pd.isna(result["days_to_next_dividend"]))
        assert all(result["is_ex_dividend"] == 0)

    def test_empty_price_data(self):
        """Test handling of empty price DataFrame."""
        price_df = pd.DataFrame()
        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 1)],
                "cash_amount": [1.0],
                "id": ["test-123"],
                "currency": ["USD"],
            }
        )

        result = compute_dividend_features(price_df, dividends_df)

        assert result.empty

    def test_missing_required_columns(self):
        """Test error handling for missing required columns."""
        dates = pd.date_range("2023-01-01", "2023-01-03", freq="D")
        price_df = pd.DataFrame({"close": [100.0] * 3}, index=dates)

        # Missing ex_dividend_date column
        dividends_df = pd.DataFrame({"cash_amount": [1.0], "id": ["test-123"]})

        with pytest.raises(ValueError, match="missing required columns"):
            compute_dividend_features(price_df, dividends_df)

    def test_price_df_missing_close_column(self):
        """Test error handling when price_df lacks close column."""
        dates = pd.date_range("2023-01-01", "2023-01-03", freq="D")
        price_df = pd.DataFrame({"open": [100.0] * 3}, index=dates)  # No close column

        dividends_df = pd.DataFrame(
            {
                "ex_dividend_date": [date(2023, 1, 2)],
                "cash_amount": [1.0],
                "id": ["test-123"],
                "currency": ["USD"],
            }
        )

        with pytest.raises(ValueError, match="must contain 'close' column"):
            compute_dividend_features(price_df, dividends_df)
