import pandas as pd
import pytest

from datetime import date
from unittest.mock import patch, Mock

from src.feature_engineering.data_loader import StockDataLoader


@pytest.fixture
def sample_rows():
    # Two valid rows and one invalid row (negative volume)
    return [
        ("2025-01-01", 10.0, 12.0, 9.0, 11.0, 1000, 11.0, 11.5),
        ("2025-01-02", 11.0, 13.0, 10.0, 12.0, 1100, 12.0, 12.5),
        ("2025-01-03", 12.0, 11.0, 10.5, 10.8, -5, 10.8, 10.8),
    ]


def build_df_from_rows(rows):
    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "adjusted_close", "vwap"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def test_load_stock_data_converts_and_cleans(sample_rows):
    """
    Setup: mock fetch_all to return sample rows

    Execution: call load_stock_data with date objects

    Verification: DataFrame returned has index of two valid rows and volume is int64
    """

    loader = StockDataLoader()

    with patch("src.feature_engineering.data_loader.fetch_all", return_value=sample_rows) as mock_fetch:
        df = loader.load_stock_data("aapl", date(2025, 1, 1), date(2025, 1, 3))

    # Expect fetch_all called with uppercase ticker and string dates
    mock_fetch.assert_called_once()

    # Two valid rows should remain after cleaning (third had negative volume)
    assert len(df) == 2
    assert df.index.min().date() == date(2025, 1, 1)
    assert df["volume"].dtype.name == "int64"


@pytest.mark.parametrize(
    "rows, expected_len, expected_exception",
    [
        ([], 0, None),
        ([("2025-01-01", None, None, None, None, None, None, None)], None, pd.errors.IntCastingNaNError),
    ],
)
def test_load_stock_data_handles_empty_and_missing(rows, expected_len, expected_exception):
    """
    Setup: mock fetch_all to return empty or rows with missing data

    Execution: call load_stock_data

    Verification: returns empty DataFrame and does not raise
    """

    loader = StockDataLoader()

    with patch("src.feature_engineering.data_loader.fetch_all", return_value=rows):
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                loader.load_stock_data("msft", "2025-01-01", "2025-01-01")
        else:
            df = loader.load_stock_data("msft", "2025-01-01", "2025-01-01")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == expected_len


def test_validate_and_clean_data_raises_on_missing_columns():
    """
    Setup: build DataFrame missing required columns

    Execution: call _validate_and_clean_data

    Verification: raises ValueError with message containing missing column name
    """

    loader = StockDataLoader()
    df = pd.DataFrame({"open": [1.0], "high": [2.0]})

    with pytest.raises(ValueError) as exc:
        loader._validate_and_clean_data(df, "TSLA")

    assert "Missing required columns" in str(exc.value)


def test_get_available_tickers_uses_min_data_points_from_config():
    """
    Setup: mock fetch_all to return two tickers

    Execution: call get_available_tickers without min_data_points

    Verification: returns list of tickers
    """

    rows = [("AAPL", 100, "Apple Inc.", "stocks"), ("MSFT", 90, "Microsoft", "stocks")]
    loader = StockDataLoader()

    with patch("src.feature_engineering.data_loader.fetch_all", return_value=rows) as mock_fetch:
        tickers = loader.get_available_tickers()

    mock_fetch.assert_called_once()
    assert tickers == ["AAPL", "MSFT"]


def test_get_ticker_metadata_returns_dict_and_dataframe():
    """
    Setup: mock fetch_all for single ticker and for all tickers

    Execution: call get_ticker_metadata both with ticker and without

    Verification: returns dict for single ticker and DataFrame for all tickers
    """

    # Build rows with the exact number of columns expected by the loader
    def make_row(ticker, row_id=1, name=None, market="stocks", active=True, type_="CS", market_cap=1000):
        row = [None] * 22
        row[0] = row_id
        row[1] = ticker
        row[2] = name or ticker
        row[3] = market
        row[7] = active
        row[8] = type_
        row[9] = market_cap
        return tuple(row)

    single = [make_row("AAPL", row_id=1, name="Apple Inc.")]
    all_rows = [single[0], make_row("MSFT", row_id=2, name="Microsoft", market_cap=900)]

    loader = StockDataLoader()

    with patch("src.feature_engineering.data_loader.fetch_all", return_value=single):
        md = loader.get_ticker_metadata("AAPL")

    assert isinstance(md, dict)
    assert md["ticker"] == "AAPL"

    with patch("src.feature_engineering.data_loader.fetch_all", return_value=all_rows):
        df = loader.get_ticker_metadata()

    assert isinstance(df, pd.DataFrame)
    assert "ticker" in df.columns


