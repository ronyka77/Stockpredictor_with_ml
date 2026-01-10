"""
Feature Calculator for Technical Indicators

This module provides the main feature calculation engine that orchestrates
all technical indicator calculations and provides a unified interface for
feature generation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Any, Iterable

from src.data_collector.indicator_pipeline.trend_indicators import TrendIndicatorCalculator
from src.data_collector.indicator_pipeline.momentum_indicators import MomentumIndicatorCalculator
from src.data_collector.indicator_pipeline.volatility_indicators import (
    VolatilityIndicatorCalculator,
)
from src.data_collector.indicator_pipeline.volume_indicators import VolumeIndicatorCalculator
from src.data_collector.indicator_pipeline.base import IndicatorResult, IndicatorValidator

from cytoolz import pipe
from src.utils.core.logger import get_logger
from src.data_collector.config import config

logger = get_logger(__name__)


def select_matching_columns(columns: Iterable[str], patterns: Iterable[str]) -> List[str]:
    """Return columns containing any pattern (case-insensitive)."""
    return list(
        pipe(columns, lambda cols: (c for c in cols if any(p in c.lower() for p in patterns)))
    )


class FeatureCalculator:
    """Main feature calculator orchestrating all technical indicator calculations."""

    # Mapping of category names to calculator classes and column patterns
    CALCULATOR_MAP = {
        "trend": (TrendIndicatorCalculator, ["sma", "ema", "macd", "ichimoku"]),
        "momentum": (MomentumIndicatorCalculator, ["rsi", "stoch", "roc", "williams"]),
        "volatility": (VolatilityIndicatorCalculator, ["bb_", "atr", "volatility"]),
        "volume": (VolumeIndicatorCalculator, ["obv", "vpt", "ad_"]),
    }

    def __init__(self, fe_config: Optional[Any] = None):
        """Initialize with optional config (uses default if None)."""
        self.config = fe_config or config
        self.validator = IndicatorValidator()

    def calculate_all_features(
        self, data: pd.DataFrame, include_categories: Optional[List[str]] = None
    ) -> IndicatorResult:
        """Calculate all technical indicators for given OHLCV data."""
        all_warnings = []
        include_categories = include_categories or list(self.CALCULATOR_MAP.keys())

        try:
            combined_data = pd.DataFrame(data=data, index=data.index)
            calculated_components = []

            # Loop through calculator categories
            for category in include_categories:
                if category not in self.CALCULATOR_MAP:
                    continue

                map_val = self.CALCULATOR_MAP.get(category)

                calc_class, _ = map_val
                calc = calc_class(data)
                result = calc.calculate()

                if self.validator.validate_result(result):
                    combined_data = pd.concat([combined_data, result.data], axis=1)
                    calculated_components.append(category)
                    all_warnings.extend(result.warnings)
                else:
                    logger.warning(f"{category.capitalize()} indicators failed validation")

            if combined_data.empty:
                raise ValueError("No valid features could be calculated")

            # Add derived features
            combined_data = self._add_basic_price_features(combined_data, data)
            combined_data = self._add_future_price_targets(combined_data, data)

            metadata = {
                "indicator_type": "all_features",
                "indicator_name": "Complete Technical Feature Set",
                "components": calculated_components,
                "requested_categories": include_categories,
                "data_points": len(data),
                "total_features": len(combined_data.columns),
                "feature_categories": self._build_feature_counts(combined_data),
            }

            return IndicatorResult(
                data=combined_data,
                metadata=metadata,
                quality_score=self._calculate_overall_quality_score(combined_data),
                warnings=all_warnings,
            )

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise

    def _build_feature_counts(self, df: pd.DataFrame) -> dict:
        """Build feature category counts using CALCULATOR_MAP patterns."""
        counts: dict[str, int] = {}
        for cat, val in self.CALCULATOR_MAP.items():
            try:
                _, patterns = val
            except Exception as e:
                logger.error(
                    f"Invalid CALCULATOR_MAP entry for {cat!r}: {val!r} "
                    f"(type={type(val).__name__}, error={e})"
                )
                raise
            counts[cat] = len(select_matching_columns(df.columns, patterns))
        counts["basic"] = len(select_matching_columns(df.columns, ["price_", "return_", "range_"]))
        counts["future_targets"] = len(select_matching_columns(df.columns, ["future_"]))
        return counts

    def _add_future_price_targets(
        self, features_df: pd.DataFrame, price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add future price targets for multiple horizons."""
        try:
            work_dates = pd.to_datetime(features_df.get("date", features_df.index))
            features_df["dayofweek"] = work_dates.dayofweek

            temp = features_df.copy()
            temp["_work_date"] = work_dates
            temp = temp.sort_values("_work_date")
            price_aligned = price_data.reindex(pd.DatetimeIndex(temp["_work_date"]))

            # Vectorized horizon calculations
            for h in [5, 10, 20]:
                temp[f"Future_High_{h}D"] = price_aligned["high"].shift(-h)
                temp[f"Future_Close_{h}D"] = price_aligned["close"].shift(-h)

            temp = temp.reindex(features_df.index)
            for col in [c for c in temp.columns if c.startswith("Future_")]:
                features_df[col] = temp[col]

        except Exception as e:
            logger.error(f"Error adding future price targets: {str(e)}")
            raise

        return features_df

    def _add_basic_price_features(
        self, features_df: pd.DataFrame, price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add basic price-based features."""
        try:
            # Price ratios and ranges
            features_df["Price_High_Low_Ratio"] = price_data["high"] / price_data["low"]
            features_df["Price_Close_Open_Ratio"] = price_data["close"] / price_data["open"]
            features_df["Price_Range"] = price_data["high"] - price_data["low"]
            features_df["Price_Range_Pct"] = features_df["Price_Range"] / price_data[
                "close"
            ].replace({0: np.nan})

            # Candlestick features
            body = abs(price_data["close"] - price_data["open"])
            upper_shadow = price_data["high"] - np.maximum(price_data["close"], price_data["open"])
            lower_shadow = np.minimum(price_data["close"], price_data["open"]) - price_data["low"]

            features_df["Body_Size"] = body
            features_df["Upper_Shadow"] = upper_shadow
            features_df["Lower_Shadow"] = lower_shadow
            features_df["Body_Shadow_Ratio"] = body / (upper_shadow + lower_shadow + 1e-8)

            # Returns via comprehension
            for h in [1, 5, 10, 20, 30]:
                features_df[f"Return_{h}D"] = price_data["close"].pct_change(h)
                features_df[f"Last_Close_{h}D"] = price_data["close"].shift(h)
                features_df[f"Log_Return_{h}D"] = np.log(
                    price_data["close"] / price_data["close"].shift(h)
                )

            # Volume features
            if "volume" in price_data.columns:
                features_df["Volume_Price_Ratio"] = price_data["volume"] / price_data["close"]
                features_df["Volume_Range_Ratio"] = price_data["volume"] / features_df[
                    "Price_Range"
                ].replace({0: np.nan})

            # Gap analysis
            features_df["Gap_Up"] = (price_data["open"] > price_data["high"].shift(1)).astype(int)
            features_df["Gap_Down"] = (price_data["open"] < price_data["low"].shift(1)).astype(int)
            features_df["Gap_Size"] = price_data["open"] - price_data["close"].shift(1)

            return features_df
        except Exception as e:
            logger.error(f"Error adding basic price features: {str(e)}")
            raise

    def _calculate_overall_quality_score(self, features_df: pd.DataFrame) -> float:
        """Calculate quality score (0-100) based on missing, infinite, and outlier percentages."""
        if features_df.empty:
            return 0.0

        try:
            # Vectorized quality metrics
            missing_pct = features_df.isna().sum().sum() / features_df.size
            numeric_df = features_df.select_dtypes(include=[np.number])
            infinite_pct = (
                np.isinf(numeric_df).sum().sum() / numeric_df.size if numeric_df.size > 0 else 0
            )

            # Simplified outlier detection using IQR vectorization
            outlier_pct = 0.0
            if len(numeric_df.columns) > 0:
                q1 = numeric_df.quantile(0.25)
                q3 = numeric_df.quantile(0.75)
                iqr = q3 - q1
                outliers = (
                    ((numeric_df < (q1 - 1.5 * iqr)) | (numeric_df > (q3 + 1.5 * iqr))).sum().sum()
                )
                outlier_pct = (
                    outliers / numeric_df.count().sum() if numeric_df.count().sum() > 0 else 0
                )

            quality_score = (
                100 * (1 - missing_pct) * (1 - infinite_pct) * (1 - min(outlier_pct, 0.5))
            )
            return max(0.0, min(100.0, quality_score))
        except Exception as e:
            logger.error(f"Error calculating overall quality score: {str(e)}")
            return 0.0
