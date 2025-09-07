"""
Volume Technical Indicators

This module implements volume-based technical indicators including
On-Balance Volume, Volume Price Trend, and Accumulation/Distribution Line.
"""

import pandas as pd
import time
from typing import List, Optional

try:
    import ta
except ImportError:
    raise ImportError("ta is required. Install with: pip install ta")

from src.data_collector.indicator_pipeline.base import (
    BaseIndicator,
    IndicatorResult,
    create_indicator_result,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="feature_engineering")


def calculate_obv(data: pd.DataFrame) -> IndicatorResult:
    """
    Calculate On-Balance Volume

    Args:
        data: OHLCV DataFrame

    Returns:
        IndicatorResult containing OBV values
    """
    start_time = time.time()
    warnings = []

    try:
        # Calculate OBV using ta library
        obv_values = ta.volume.on_balance_volume(data["close"], data["volume"])

        result_data = pd.DataFrame(index=data.index)
        result_data["OBV"] = obv_values

        # Calculate additional OBV-based features
        result_data["OBV_SMA_10"] = obv_values.rolling(10).mean()
        result_data["OBV_SMA_20"] = obv_values.rolling(20).mean()
        result_data["OBV_Above_SMA_10"] = (
            obv_values > result_data["OBV_SMA_10"]
        ).astype(int)
        result_data["OBV_Above_SMA_20"] = (
            obv_values > result_data["OBV_SMA_20"]
        ).astype(int)

        # OBV trend analysis
        result_data["OBV_Rising"] = (obv_values > obv_values.shift(1)).astype(int)
        result_data["OBV_Falling"] = (obv_values < obv_values.shift(1)).astype(int)

        # OBV momentum
        result_data["OBV_Momentum_5"] = obv_values - obv_values.shift(5)
        result_data["OBV_Momentum_10"] = obv_values - obv_values.shift(10)

        # OBV divergence signals (simplified)
        price_change = data["close"].pct_change(5)
        obv_change = obv_values.pct_change(5)
        result_data["OBV_Price_Divergence"] = (
            ((price_change > 0) & (obv_change < 0))
            | ((price_change < 0) & (obv_change > 0))
        ).astype(int)

        metadata = {
            "indicator_type": "volume",
            "indicator_name": "On-Balance Volume",
            "data_points": len(data),
            "features": [
                "OBV",
                "OBV_SMA_10",
                "OBV_SMA_20",
                "OBV_Above_SMA_10",
                "OBV_Above_SMA_20",
                "OBV_Rising",
                "OBV_Falling",
                "OBV_Momentum_5",
                "OBV_Momentum_10",
                "OBV_Price_Divergence",
            ],
        }

        calculation_time = time.time() - start_time

        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time,
        )

    except Exception as e:
        logger.error(f"Error calculating OBV: {str(e)}")
        raise


def calculate_vpt(data: pd.DataFrame) -> IndicatorResult:
    """
    Calculate Volume Price Trend

    Args:
        data: OHLCV DataFrame

    Returns:
        IndicatorResult containing VPT values
    """
    start_time = time.time()
    warnings = []

    try:
        # Calculate VPT using ta library
        vpt_values = ta.volume.volume_price_trend(data["close"], data["volume"])

        result_data = pd.DataFrame(index=data.index)
        result_data["VPT"] = vpt_values

        # Calculate additional VPT-based features
        result_data["VPT_SMA_10"] = vpt_values.rolling(10).mean()
        result_data["VPT_SMA_20"] = vpt_values.rolling(20).mean()
        result_data["VPT_Above_SMA_10"] = (
            vpt_values > result_data["VPT_SMA_10"]
        ).astype(int)
        result_data["VPT_Above_SMA_20"] = (
            vpt_values > result_data["VPT_SMA_20"]
        ).astype(int)

        # VPT trend analysis
        result_data["VPT_Rising"] = (vpt_values > vpt_values.shift(1)).astype(int)
        result_data["VPT_Falling"] = (vpt_values < vpt_values.shift(1)).astype(int)

        # VPT momentum
        result_data["VPT_Momentum_5"] = vpt_values - vpt_values.shift(5)
        result_data["VPT_Momentum_10"] = vpt_values - vpt_values.shift(10)

        # VPT rate of change
        result_data["VPT_ROC_5"] = vpt_values.pct_change(5) * 100
        result_data["VPT_ROC_10"] = vpt_values.pct_change(10) * 100

        metadata = {
            "indicator_type": "volume",
            "indicator_name": "Volume Price Trend",
            "data_points": len(data),
            "features": [
                "VPT",
                "VPT_SMA_10",
                "VPT_SMA_20",
                "VPT_Above_SMA_10",
                "VPT_Above_SMA_20",
                "VPT_Rising",
                "VPT_Falling",
                "VPT_Momentum_5",
                "VPT_Momentum_10",
                "VPT_ROC_5",
                "VPT_ROC_10",
            ],
        }

        calculation_time = time.time() - start_time

        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time,
        )

    except Exception as e:
        logger.error(f"Error calculating VPT: {str(e)}")
        raise


def calculate_ad_line(data: pd.DataFrame) -> IndicatorResult:
    """
    Calculate Accumulation/Distribution Line

    Args:
        data: OHLCV DataFrame

    Returns:
        IndicatorResult containing A/D Line values
    """
    start_time = time.time()
    warnings = []

    try:
        # Calculate A/D Line using ta library
        ad_values = ta.volume.acc_dist_index(
            data["high"], data["low"], data["close"], data["volume"]
        )

        result_data = pd.DataFrame(index=data.index)
        result_data["AD_Line"] = ad_values

        # Calculate additional A/D Line features
        result_data["AD_SMA_10"] = ad_values.rolling(10).mean()
        result_data["AD_SMA_20"] = ad_values.rolling(20).mean()
        result_data["AD_Above_SMA_10"] = (ad_values > result_data["AD_SMA_10"]).astype(
            int
        )
        result_data["AD_Above_SMA_20"] = (ad_values > result_data["AD_SMA_20"]).astype(
            int
        )

        # A/D Line trend analysis
        result_data["AD_Rising"] = (ad_values > ad_values.shift(1)).astype(int)
        result_data["AD_Falling"] = (ad_values < ad_values.shift(1)).astype(int)

        # A/D Line momentum
        result_data["AD_Momentum_5"] = ad_values - ad_values.shift(5)
        result_data["AD_Momentum_10"] = ad_values - ad_values.shift(10)

        # A/D Line oscillator (A/D Line - EMA of A/D Line)
        ad_ema = ad_values.ewm(span=10).mean()
        result_data["AD_Oscillator"] = ad_values - ad_ema
        result_data["AD_Oscillator_Above_Zero"] = (
            result_data["AD_Oscillator"] > 0
        ).astype(int)

        metadata = {
            "indicator_type": "volume",
            "indicator_name": "Accumulation/Distribution Line",
            "data_points": len(data),
            "features": [
                "AD_Line",
                "AD_SMA_10",
                "AD_SMA_20",
                "AD_Above_SMA_10",
                "AD_Above_SMA_20",
                "AD_Rising",
                "AD_Falling",
                "AD_Momentum_5",
                "AD_Momentum_10",
                "AD_Oscillator",
                "AD_Oscillator_Above_Zero",
            ],
        }

        calculation_time = time.time() - start_time

        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time,
        )

    except Exception as e:
        logger.error(f"Error calculating A/D Line: {str(e)}")
        raise


def calculate_volume_profile(
    data: pd.DataFrame, periods: Optional[List[int]] = None
) -> IndicatorResult:
    """
    Calculate Volume Profile and related volume analysis

    Args:
        data: OHLCV DataFrame
        periods: List of periods for volume analysis

    Returns:
        IndicatorResult containing volume profile features
    """
    start_time = time.time()
    warnings = []

    if periods is None:
        periods = [10, 20, 50]  # Default volume analysis periods

    try:
        result_data = pd.DataFrame(index=data.index)

        for period in periods:
            if len(data) < period:
                warning_msg = f"Insufficient data for Volume Profile_{period}. Need {period} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue

            # Volume moving averages
            vol_ma = data["volume"].rolling(period).mean()
            result_data[f"Volume_MA_{period}"] = vol_ma

            # Volume relative to moving average
            result_data[f"Volume_Ratio_{period}"] = data["volume"] / vol_ma

            # High volume detection
            vol_threshold = vol_ma * 1.5
            result_data[f"High_Volume_{period}"] = (
                data["volume"] > vol_threshold
            ).astype(int)

            # Low volume detection
            vol_low_threshold = vol_ma * 0.5
            result_data[f"Low_Volume_{period}"] = (
                data["volume"] < vol_low_threshold
            ).astype(int)

            # Volume trend
            volume_trend = (
                data["volume"]
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
            result_data[f"Volume_Trend_{period}"] = volume_trend.fillna(0).astype(int)

        # Volume-Price relationship analysis
        if len(data) >= 20:
            # Price-Volume correlation
            price_change = data["close"].pct_change()
            volume_change = data["volume"].pct_change()

            result_data["PV_Correlation_10"] = price_change.rolling(10).corr(
                volume_change
            )
            result_data["PV_Correlation_20"] = price_change.rolling(20).corr(
                volume_change
            )

            # Volume confirmation signals
            vol_ma_10 = data["volume"].rolling(10).mean()
            result_data["Volume_Confirms_Uptrend"] = (
                ((price_change > 0) & (data["volume"] > vol_ma_10))
                .fillna(False)
                .astype(int)
            )

            result_data["Volume_Confirms_Downtrend"] = (
                ((price_change < 0) & (data["volume"] > vol_ma_10))
                .fillna(False)
                .astype(int)
            )

            # Volume divergence
            result_data["Volume_Divergence"] = (
                (
                    ((price_change > 0) & (volume_change < 0))
                    | ((price_change < 0) & (volume_change > 0))
                )
                .fillna(False)
                .astype(int)
            )

        # Volume clustering and distribution
        if len(data) >= 50:
            # Volume percentiles
            vol_50 = data["volume"].rolling(50)
            result_data["Volume_Percentile_50"] = vol_50.apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
                if x.max() != x.min()
                and not pd.isna(x.iloc[-1])
                and not pd.isna(x.min())
                and not pd.isna(x.max())
                else 0.5
            )

            # Volume regime detection
            vol_std = data["volume"].rolling(50).std()
            vol_mean = data["volume"].rolling(50).mean()
            result_data["Volume_Regime_High"] = (
                (data["volume"] > vol_mean + 2 * vol_std).fillna(False).astype(int)
            )

            result_data["Volume_Regime_Low"] = (
                (data["volume"] < vol_mean - vol_std).fillna(False).astype(int)
            )

        if result_data.empty:
            raise ValueError("No Volume Profile indicators could be calculated")

        metadata = {
            "indicator_type": "volume",
            "indicator_name": "Volume Profile Analysis",
            "periods": periods,
            "data_points": len(data),
            "features": [col for col in result_data.columns],
            "analysis_types": [
                "Moving Averages",
                "Ratios",
                "Trends",
                "Price-Volume Correlation",
                "Regime Detection",
            ],
        }

        calculation_time = time.time() - start_time

        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time,
        )

    except Exception as e:
        logger.error(f"Error calculating Volume Profile: {str(e)}")
        raise


def calculate_money_flow_index(
    data: pd.DataFrame, period: Optional[int] = None
) -> IndicatorResult:
    """
    Calculate Money Flow Index

    Args:
        data: OHLCV DataFrame
        period: Period for MFI calculation

    Returns:
        IndicatorResult containing MFI values
    """
    start_time = time.time()
    warnings = []

    if period is None:
        period = 14  # Default MFI period

    try:
        # Check minimum data requirements
        if len(data) < period:
            warning_msg = (
                f"Insufficient data for MFI. Need {period} points, have {len(data)}"
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)

        # Calculate MFI using ta library
        mfi_values = ta.volume.money_flow_index(
            data["high"], data["low"], data["close"], data["volume"], window=period
        )

        result_data = pd.DataFrame(index=data.index)
        result_data["MFI"] = mfi_values

        # Add MFI-based signals
        result_data["MFI_Overbought"] = (mfi_values > 80).astype(int)
        result_data["MFI_Oversold"] = (mfi_values < 20).astype(int)
        result_data["MFI_Neutral"] = ((mfi_values >= 20) & (mfi_values <= 80)).astype(
            int
        )

        # MFI trend analysis
        result_data["MFI_Rising"] = (mfi_values > mfi_values.shift(1)).astype(int)
        result_data["MFI_Falling"] = (mfi_values < mfi_values.shift(1)).astype(int)

        # MFI momentum
        result_data["MFI_Momentum_5"] = mfi_values - mfi_values.shift(5)
        result_data["MFI_Momentum_10"] = mfi_values - mfi_values.shift(10)

        # MFI divergence with price
        price_change_5 = data["close"].pct_change(5)
        mfi_change_5 = mfi_values.pct_change(5)
        result_data["MFI_Price_Divergence"] = (
            ((price_change_5 > 0) & (mfi_change_5 < 0))
            | ((price_change_5 < 0) & (mfi_change_5 > 0))
        ).astype(int)

        metadata = {
            "indicator_type": "volume",
            "indicator_name": "Money Flow Index",
            "parameters": {"period": period},
            "data_points": len(data),
            "features": [
                "MFI",
                "MFI_Overbought",
                "MFI_Oversold",
                "MFI_Neutral",
                "MFI_Rising",
                "MFI_Falling",
                "MFI_Momentum_5",
                "MFI_Momentum_10",
                "MFI_Price_Divergence",
            ],
            "thresholds": {"overbought": 80, "oversold": 20},
        }

        calculation_time = time.time() - start_time

        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time,
        )

    except Exception as e:
        logger.error(f"Error calculating MFI: {str(e)}")
        raise


class VolumeIndicatorCalculator(BaseIndicator):
    """Calculator for all volume-based technical indicators"""

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def calculate(self) -> IndicatorResult:
        """
        Calculate all volume indicators and combine results

        Returns:
            IndicatorResult containing all volume indicators
        """
        start_time = time.time()

        try:
            # Calculate individual indicators
            obv_result = calculate_obv(self.data)
            vpt_result = calculate_vpt(self.data)
            ad_result = calculate_ad_line(self.data)
            volume_profile_result = calculate_volume_profile(self.data)
            mfi_result = calculate_money_flow_index(self.data)

            # Combine all results
            combined_data = pd.concat(
                [
                    obv_result.data,
                    vpt_result.data,
                    ad_result.data,
                    volume_profile_result.data,
                    mfi_result.data,
                ],
                axis=1,
            )

            # Combine warnings
            all_warnings = (
                obv_result.warnings
                + vpt_result.warnings
                + ad_result.warnings
                + volume_profile_result.warnings
                + mfi_result.warnings
            )

            # Create combined metadata
            metadata = {
                "indicator_type": "volume",
                "indicator_name": "Combined Volume Indicators",
                "data_points": len(self.data),
                "total_features": len(combined_data.columns),
                "individual_results": {
                    "obv": obv_result.metadata,
                    "vpt": vpt_result.metadata,
                    "ad_line": ad_result.metadata,
                    "volume_profile": volume_profile_result.metadata,
                    "mfi": mfi_result.metadata,
                },
            }

            calculation_time = time.time() - start_time

            return create_indicator_result(
                data=combined_data,
                metadata=metadata,
                warnings=all_warnings,
                calculation_time=calculation_time,
            )

        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            raise
