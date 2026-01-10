"""
Volume Technical Indicators

This module implements volume-based technical indicators including
On-Balance Volume, Volume Price Trend, and Accumulation/Distribution Line.
"""

import pandas as pd
from typing import List, Optional, Callable, Dict, Any

try:
    import ta
except ImportError:
    raise ImportError("ta is required. Install with: pip install ta")

from src.data_collector.indicator_pipeline.base import (
    BaseIndicator,
    IndicatorResult,
    create_indicator_result,
)
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


def _add_ma_features(df: pd.DataFrame, col: str, periods: List[int] = [10, 20]) -> None:
    """Add moving average and comparison features in-place"""
    values = df[col]
    for p in periods:
        ma_col = f"{col}_SMA_{p}"
        df[ma_col] = values.rolling(p).mean()
        df[f"{col}_Above_SMA_{p}"] = (values > df[ma_col]).astype(int)


def _add_trend_features(df: pd.DataFrame, col: str) -> None:
    """Add rising/falling trend features in-place"""
    values = df[col]
    df[f"{col}_Rising"] = (values > values.shift(1)).astype(int)
    df[f"{col}_Falling"] = (values < values.shift(1)).astype(int)


def _add_momentum_features(df: pd.DataFrame, col: str, periods: List[int] = [5, 10]) -> None:
    """Add momentum features in-place"""
    values = df[col]
    for p in periods:
        df[f"{col}_Momentum_{p}"] = values - values.shift(p)


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Calculate percentage change with inf/nan handling"""
    result = series.pct_change(periods) * 100
    return result.replace([float("inf"), float("-inf")], float("nan"))


def _indicator_wrapper(
    data: pd.DataFrame,
    calc_func: Callable,
    indicator_name: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> IndicatorResult:
    """Generic wrapper for indicator calculations with timing and error handling"""
    warnings = []

    try:
        result_data, features, calc_warnings = calc_func(data, warnings)
        warnings.extend(calc_warnings)

        metadata = {
            "indicator_type": "volume",
            "indicator_name": indicator_name,
            "data_points": len(data),
            "features": features,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return create_indicator_result(data=result_data, metadata=metadata, warnings=warnings)
    except Exception as e:
        logger.error(f"Error calculating {indicator_name}: {str(e)}")
        raise


def calculate_obv(data: pd.DataFrame) -> IndicatorResult:
    """Calculate On-Balance Volume"""

    def _calc(df: pd.DataFrame, warnings: List[str]) -> tuple:
        obv = ta.volume.on_balance_volume(df["close"], df["volume"])
        result = pd.DataFrame({"OBV": obv}, index=df.index)

        _add_ma_features(result, "OBV")
        _add_trend_features(result, "OBV")
        _add_momentum_features(result, "OBV")

        # Divergence using delta instead of pct_change for stability
        price_change = df["close"].pct_change(5)
        obv_delta = obv.diff(5)
        result["OBV_Price_Divergence"] = (
            (((price_change > 0) & (obv_delta < 0)) | ((price_change < 0) & (obv_delta > 0)))
            .fillna(False)
            .astype(int)
        )

        return result, list(result.columns), []

    return _indicator_wrapper(data, _calc, "On-Balance Volume")


def calculate_vpt(data: pd.DataFrame) -> IndicatorResult:
    """Calculate Volume Price Trend"""

    def _calc(df: pd.DataFrame, warnings: List[str]) -> tuple:
        vpt = ta.volume.volume_price_trend(df["close"], df["volume"])
        result = pd.DataFrame({"VPT": vpt}, index=df.index)

        _add_ma_features(result, "VPT")
        _add_trend_features(result, "VPT")
        _add_momentum_features(result, "VPT")

        result["VPT_ROC_5"] = _safe_pct_change(vpt, 5)
        result["VPT_ROC_10"] = _safe_pct_change(vpt, 10)

        return result, list(result.columns), []

    return _indicator_wrapper(data, _calc, "Volume Price Trend")


def calculate_ad_line(data: pd.DataFrame) -> IndicatorResult:
    """Calculate Accumulation/Distribution Line"""

    def _calc(df: pd.DataFrame, warnings: List[str]) -> tuple:
        ad = ta.volume.acc_dist_index(df["high"], df["low"], df["close"], df["volume"])
        result = pd.DataFrame({"AD_Line": ad}, index=df.index)

        _add_ma_features(result, "AD_Line")
        _add_trend_features(result, "AD_Line")
        _add_momentum_features(result, "AD_Line")

        ad_ema = ad.ewm(span=10).mean()
        result["AD_Oscillator"] = ad - ad_ema
        result["AD_Oscillator_Above_Zero"] = (result["AD_Oscillator"] > 0).astype(int)

        return result, list(result.columns), []

    return _indicator_wrapper(data, _calc, "Accumulation/Distribution Line")


def calculate_volume_profile(
    data: pd.DataFrame, periods: Optional[List[int]] = None
) -> IndicatorResult:
    """Calculate Volume Profile and related volume analysis"""

    def _calc(df: pd.DataFrame, warnings: List[str]) -> tuple:
        periods_list = periods or [10, 20, 50]
        result = pd.DataFrame(index=df.index)

        for period in periods_list:
            if len(df) < period:
                warnings.append(
                    f"Insufficient data for Volume Profile_{period}. Need {period} points, have {len(df)}"
                )
                continue

            vol_ma = df["volume"].rolling(period).mean()
            result[f"Volume_MA_{period}"] = vol_ma
            result[f"Volume_Ratio_{period}"] = df["volume"] / vol_ma
            result[f"High_Volume_{period}"] = (df["volume"] > vol_ma * 1.5).astype(int)
            result[f"Low_Volume_{period}"] = (df["volume"] < vol_ma * 0.5).astype(int)

            volume_trend = (
                df["volume"]
                .rolling(period)
                .apply(
                    lambda x: 1
                    if len(x) >= 2
                    and not pd.isna(x.iloc[-1])
                    and not pd.isna(x.iloc[0])
                    and x.iloc[-1] > x.iloc[0]
                    else 0
                )
            )
            result[f"Volume_Trend_{period}"] = volume_trend.fillna(0).astype(int)

        if len(df) >= 20:
            price_change = df["close"].pct_change()
            volume_change = df["volume"].pct_change()
            result["PV_Correlation_10"] = price_change.rolling(10).corr(volume_change)
            result["PV_Correlation_20"] = price_change.rolling(20).corr(volume_change)

            vol_ma_10 = df["volume"].rolling(10).mean()
            result["Volume_Confirms_Uptrend"] = (
                ((price_change > 0) & (df["volume"] > vol_ma_10)).fillna(False).astype(int)
            )
            result["Volume_Confirms_Downtrend"] = (
                ((price_change < 0) & (df["volume"] > vol_ma_10)).fillna(False).astype(int)
            )
            result["Volume_Divergence"] = (
                (
                    ((price_change > 0) & (volume_change < 0))
                    | ((price_change < 0) & (volume_change > 0))
                )
                .fillna(False)
                .astype(int)
            )

        if len(df) >= 50:
            vol_50 = df["volume"].rolling(50)
            result["Volume_Percentile_50"] = vol_50.apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
                if x.max() != x.min()
                and not pd.isna(x.iloc[-1])
                and not pd.isna(x.min())
                and not pd.isna(x.max())
                else 0.5
            )

            from data_collector.indicator_pipeline.volume_regimes import (
                RollingRegimePolicy,
                compute_volume_regimes,
            )

            policy = RollingRegimePolicy(lookback=50, method="std", k=2.0)
            high_flags, low_flags = compute_volume_regimes(df["volume"], policy)
            result["Volume_Regime_High"] = high_flags
            result["Volume_Regime_Low"] = low_flags

        if result.empty:
            raise ValueError("No Volume Profile indicators could be calculated")

        return result, list(result.columns), warnings

    extra_meta = {
        "periods": periods or [10, 20, 50],
        "analysis_types": [
            "Moving Averages",
            "Ratios",
            "Trends",
            "Price-Volume Correlation",
            "Regime Detection",
        ],
    }
    return _indicator_wrapper(data, _calc, "Volume Profile Analysis", extra_meta)


def calculate_money_flow_index(data: pd.DataFrame, period: Optional[int] = None) -> IndicatorResult:
    """Calculate Money Flow Index"""

    def _calc(df: pd.DataFrame, warnings: List[str]) -> tuple:
        p = period or 14
        if len(df) < p:
            warnings.append(f"Insufficient data for MFI. Need {p} points, have {len(df)}")

        mfi = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=p)
        result = pd.DataFrame({"MFI": mfi}, index=df.index)

        result["MFI_Overbought"] = (mfi > 80).astype(int)
        result["MFI_Oversold"] = (mfi < 20).astype(int)
        result["MFI_Neutral"] = ((mfi >= 20) & (mfi <= 80)).astype(int)

        _add_trend_features(result, "MFI")
        _add_momentum_features(result, "MFI")

        price_change_5 = df["close"].pct_change(5)
        mfi_change_5 = mfi.pct_change(5)
        result["MFI_Price_Divergence"] = (
            ((price_change_5 > 0) & (mfi_change_5 < 0))
            | ((price_change_5 < 0) & (mfi_change_5 > 0))
        ).astype(int)

        return result, list(result.columns), warnings

    extra_meta = {
        "parameters": {"period": period or 14},
        "thresholds": {"overbought": 80, "oversold": 20},
    }
    return _indicator_wrapper(data, _calc, "Money Flow Index", extra_meta)


class VolumeIndicatorCalculator(BaseIndicator):
    """Calculator for all volume-based technical indicators"""

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def calculate(self) -> IndicatorResult:
        """Calculate all volume indicators and combine results"""
        try:
            results = [
                calculate_obv(self.data),
                calculate_vpt(self.data),
                calculate_ad_line(self.data),
                calculate_volume_profile(self.data),
                calculate_money_flow_index(self.data),
            ]

            combined_data = pd.concat([r.data for r in results], axis=1)
            all_warnings = [w for r in results for w in r.warnings]

            metadata = {
                "indicator_type": "volume",
                "indicator_name": "Combined Volume Indicators",
                "data_points": len(self.data),
                "total_features": len(combined_data.columns),
                "individual_results": {
                    name: r.metadata
                    for name, r in zip(["obv", "vpt", "ad_line", "volume_profile", "mfi"], results)
                },
            }

            return create_indicator_result(
                data=combined_data, metadata=metadata, warnings=all_warnings
            )

        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            raise
