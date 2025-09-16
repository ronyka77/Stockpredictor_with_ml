import pandas as pd
import pytest

from datetime import date
from unittest.mock import patch, Mock

from src.feature_engineering.data_loader import StockDataLoader


@pytest.fixture
def sample_rows():
    # Two valid rows and one invalid row (negative volume)
    """
    Return a list of sample stock data rows for tests.
    
    Each item is an 8-tuple representing:
    (date_str, open, high, low, close, volume, adjusted_close, extra_price)
    
    Includes two valid rows and a third row with a negative volume to exercise cleaning/validation logic.
    """
    return [
        ("2025-01-01", 10.0, 12.0, 9.0, 11.0, 1000, 11.0, 11.5),
        ("2025-01-02", 11.0, 13.0, 10.0, 12.0, 1100, 12.0, 12.5),
        ("2025-01-03", 12.0, 11.0, 10.5, 10.8, -5, 10.8, 10.8),
    ]


def build_df_from_rows(rows):
    """
    Build a pandas DataFrame from raw row records, convert the "date" column to datetime, and set it as the index.
    
    Parameters:
        rows (Iterable[Sequence]): Iterable of row records where each row provides values in the order
            ["date", "open", "high", "low", "close", "volume", "adjusted_close", "vwap"].
    
    Returns:
        pandas.DataFrame: DataFrame indexed by the parsed "date" column with columns
        ["open", "high", "low", "close", "volume", "adjusted_close", "vwap"] (the "date" column removed after indexing).
    """
    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "adjusted_close", "vwap"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def test_load_stock_data_converts_and_cleans(sample_rows):
    """
    Test that StockDataLoader.load_stock_data converts input dates, calls the data fetcher, and cleans invalid rows.
    
    Uses the `sample_rows` fixture and patches `fetch_all` to return those rows. Verifies:
    - `fetch_all` is invoked once,
    - returned DataFrame contains only the two valid rows (rows with negative volume are removed),
    - the index minimum date is 2025-01-01,
    - the `volume` column has dtype int64.
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
    Ensure _validate_and_clean_data raises a ValueError when required columns are missing.
    
    This test constructs a DataFrame that lacks the required columns and verifies that
    StockDataLoader._validate_and_clean_data raises a ValueError whose message
    contains "Missing required columns".
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
        """
        Create a 22-field tuple representing a ticker metadata row for tests.
        
        Only a subset of fields are populated; the rest are None. Populated indices:
        - 0: row_id (int)
        - 1: ticker (str)
        - 2: name (str) — defaults to ticker when not provided
        - 3: market (str)
        - 7: active (bool)
        - 8: type_ (str)
        - 9: market_cap (numeric)
        
        Parameters:
            ticker (str): The ticker symbol to set at index 1.
            row_id (int, optional): Identifier placed at index 0. Defaults to 1.
            name (str | None, optional): Display name placed at index 2; defaults to ticker when None.
            market (str, optional): Market value placed at index 3. Defaults to "stocks".
            active (bool, optional): Active flag placed at index 7. Defaults to True.
            type_ (str, optional): Security type placed at index 8. Defaults to "CS".
            market_cap (numeric, optional): Market capitalization placed at index 9. Defaults to 1000.
        
        Returns:
            tuple: A 22-element tuple representing the metadata row.
        """
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


