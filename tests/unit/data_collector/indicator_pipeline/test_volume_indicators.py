import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data_collector.indicator_pipeline.volume_indicators import (
    VolumeIndicatorCalculator,
    calculate_ad_line,
    calculate_money_flow_index,
    calculate_obv,
    calculate_volume_profile,
    calculate_vpt,
)


def _make_ohlcv_dataframe(num_rows: int = 120) -> pd.DataFrame:
    start = datetime(2020, 1, 1)
    index = pd.date_range(start=start, periods=num_rows, freq="D")

    # Create price series with gentle trend and noise
    base = np.linspace(100, 120, num_rows)
    noise = np.sin(np.linspace(0, 8 * math.pi, num_rows)) * 0.5
    close = base + noise
    open_ = close - np.random.default_rng(0).normal(0.1, 0.05, size=num_rows)
    high = np.maximum(close, open_) + 0.2
    low = np.minimum(close, open_) - 0.2
    volume = np.abs(
        np.random.default_rng(1).integers(1_000_000, 5_000_000, size=num_rows)
    )

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.mark.unit
def test_calculate_obv_success():
    df = _make_ohlcv_dataframe(60)
    result = calculate_obv(df)

    assert not result.data.empty
    assert {"OBV", "OBV_SMA_10", "OBV_SMA_20", "OBV_Rising", "OBV_Falling"}.issubset(
        set(result.data.columns)
    )
    assert result.metadata["indicator_name"] == "On-Balance Volume"
    assert 0.0 <= result.quality_score <= 100.0


@pytest.mark.unit
def test_calculate_vpt_success():
    df = _make_ohlcv_dataframe(60)
    result = calculate_vpt(df)

    assert not result.data.empty
    assert {"VPT", "VPT_SMA_10", "VPT_SMA_20", "VPT_Rising", "VPT_Falling"}.issubset(
        set(result.data.columns)
    )
    assert result.metadata["indicator_name"] == "Volume Price Trend"


@pytest.mark.unit
def test_calculate_ad_line_success():
    df = _make_ohlcv_dataframe(60)
    result = calculate_ad_line(df)

    assert not result.data.empty
    assert {"AD_Line", "AD_SMA_10", "AD_SMA_20", "AD_Rising", "AD_Falling"}.issubset(
        set(result.data.columns)
    )
    assert result.metadata["indicator_name"] == "Accumulation/Distribution Line"


@pytest.mark.unit
def test_calculate_volume_profile_success_and_features():
    df = _make_ohlcv_dataframe(100)
    result = calculate_volume_profile(df)

    expected_cols = {
        # MAs and ratios
        "Volume_MA_10",
        "Volume_MA_20",
        "Volume_MA_50",
        "Volume_Ratio_10",
        "Volume_Ratio_20",
        "Volume_Ratio_50",
        # High/low volume flags
        "High_Volume_10",
        "High_Volume_20",
        "High_Volume_50",
        "Low_Volume_10",
        "Low_Volume_20",
        "Low_Volume_50",
        # Trends
        "Volume_Trend_10",
        "Volume_Trend_20",
        "Volume_Trend_50",
        # Price-volume relationship
        "PV_Correlation_10",
        "PV_Correlation_20",
        "Volume_Confirms_Uptrend",
        "Volume_Confirms_Downtrend",
        "Volume_Divergence",
        # Distribution/regime
        "Volume_Percentile_50",
        "Volume_Regime_High",
        "Volume_Regime_Low",
    }
    assert expected_cols.issubset(set(result.data.columns))
    assert result.metadata["indicator_name"] == "Volume Profile Analysis"
    assert 0.0 <= result.quality_score <= 100.0


@pytest.mark.unit
def test_calculate_volume_profile_raises_with_insufficient_rows():
    df = _make_ohlcv_dataframe(5)
    with pytest.raises(
        ValueError, match="No Volume Profile indicators could be calculated"
    ):
        calculate_volume_profile(df)


@pytest.mark.unit
def test_calculate_mfi_default_and_warnings():
    # Enough rows for default period
    df_ok = _make_ohlcv_dataframe(20)
    res_ok = calculate_money_flow_index(df_ok)
    assert {"MFI", "MFI_Overbought", "MFI_Oversold"}.issubset(set(res_ok.data.columns))
    assert res_ok.metadata["parameters"]["period"] == 14

    # Insufficient rows produces warnings but still returns a result
    df_short = _make_ohlcv_dataframe(10)
    res_short = calculate_money_flow_index(df_short)
    assert isinstance(res_short.warnings, list)
    assert any("Insufficient data for MFI" in w for w in res_short.warnings)


@pytest.mark.unit
def test_volume_indicator_calculator_combines_all():
    df = _make_ohlcv_dataframe(120)
    calc = VolumeIndicatorCalculator(df)
    combined = calc.calculate()

    # Spot-check presence of representative columns from each component
    representative = {
        "OBV",
        "VPT",
        "AD_Line",
        "Volume_MA_10",
        "MFI",
    }
    assert representative.issubset(set(combined.data.columns))
    assert combined.metadata["indicator_name"] == "Combined Volume Indicators"
    assert "individual_results" in combined.metadata
    assert combined.metadata["total_features"] == len(combined.data.columns)
