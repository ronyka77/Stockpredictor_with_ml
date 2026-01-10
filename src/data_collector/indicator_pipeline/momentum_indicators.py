"""
Momentum Technical Indicators

This module implements momentum-based technical indicators including
RSI, Stochastic Oscillator, Rate of Change, and Williams %R.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

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
from src.data_collector.config import feature_config

logger = get_logger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for momentum indicator calculation"""

    name: str
    calc_func: Callable
    default_periods: List[int]
    min_data_multiplier: float
    signal_thresholds: Dict[str, tuple]
    calc_kwargs: Dict[str, Any] = None


# Indicator registry with all configurations
MOMENTUM_CONFIGS = {
    "RSI": IndicatorConfig(
        name="Relative Strength Index",
        calc_func=lambda df, p: ta.momentum.rsi(df["close"], window=p),
        default_periods=lambda: feature_config.RSI_PERIODS,
        min_data_multiplier=2.0,
        signal_thresholds={
            "Overbought": (">", 70),
            "Oversold": ("<", 30),
            "Neutral": ("between", (30, 70)),
        },
    ),
    "ROC": IndicatorConfig(
        name="Rate of Change",
        calc_func=lambda df, p: ta.momentum.roc(df["close"], window=p),
        default_periods=[10, 20, 30],
        min_data_multiplier=1.0,
        signal_thresholds={
            "Positive": (">", 0),
            "Strong_Positive": (">", 5),
            "Strong_Negative": ("<", -5),
        },
    ),
    "Williams_R": IndicatorConfig(
        name="Williams %R",
        calc_func=lambda df, p: ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=p),
        default_periods=[14, 21],
        min_data_multiplier=1.0,
        signal_thresholds={
            "Overbought": (">", -20),
            "Oversold": ("<", -80),
            "Neutral": ("between", (-80, -20)),
        },
    ),
}


def _apply_signals(df: pd.DataFrame, col: str, thresholds: Dict[str, tuple]) -> None:
    """Generate signal columns based on threshold rules"""
    for name, (op, val) in thresholds.items():
        if op == ">":
            df[f"{col}_{name}"] = (df[col] > val).astype(int)
        elif op == "<":
            df[f"{col}_{name}"] = (df[col] < val).astype(int)
        elif op == "between":
            df[f"{col}_{name}"] = ((df[col] >= val[0]) & (df[col] <= val[1])).astype(int)


def calculate_generic_momentum(
    data: pd.DataFrame, config_key: str, periods: Optional[List[int]] = None
) -> IndicatorResult:
    """Generic momentum indicator calculator driven by config"""
    config = MOMENTUM_CONFIGS[config_key]
    warnings = []

    periods = periods or (
        config.default_periods() if callable(config.default_periods) else config.default_periods
    )
    result_data = pd.DataFrame(index=data.index)
    valid_calcs = 0

    try:
        for period in periods:
            min_required = int(period * config.min_data_multiplier) + 1
            if len(data) < min_required:
                msg = f"Insufficient data for {config_key}_{period}. Need {min_required}, have {len(data)}"
                warnings.append(msg)
                logger.warning(msg)
                continue

            col_name = f"{config_key}_{period}"
            result_data[col_name] = config.calc_func(data, period)
            _apply_signals(result_data, col_name, config.signal_thresholds)
            valid_calcs += 1

        if result_data.empty:
            raise ValueError(f"No {config_key} indicators could be calculated")

        metadata = {
            "indicator_type": "momentum",
            "indicator_name": config.name,
            "periods": periods,
            "data_points": len(data),
            "valid_calculations": valid_calcs,
            "thresholds": {k: v[1] for k, v in config.signal_thresholds.items()},
        }

        return create_indicator_result(data=result_data, metadata=metadata, warnings=warnings)
    except Exception as e:
        logger.error(f"Error calculating {config_key}: {str(e)}")
        raise


def calculate_rsi(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """Calculate Relative Strength Index for multiple periods"""
    return calculate_generic_momentum(data, "RSI", periods)


def calculate_roc(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """Calculate Rate of Change for multiple periods"""
    return calculate_generic_momentum(data, "ROC", periods)


def calculate_williams_r(
    data: pd.DataFrame, periods: Optional[List[int]] = None
) -> IndicatorResult:
    """Calculate Williams %R for multiple periods"""
    return calculate_generic_momentum(data, "Williams_R", periods)


def calculate_stochastic(
    data: pd.DataFrame, k_period: Optional[int] = None, d_period: Optional[int] = None
) -> IndicatorResult:
    """Calculate Stochastic Oscillator (%K and %D)"""
    warnings = []

    params = feature_config.STOCHASTIC_PARAMS
    k_period = k_period or params["k_period"]
    d_period = d_period or params["d_period"]

    min_required = k_period + d_period
    if len(data) < min_required:
        msg = f"Insufficient data for Stochastic. Need {min_required}, have {len(data)}"
        warnings.append(msg)
        logger.warning(msg)

    try:
        result_data = pd.DataFrame(
            {
                "Stoch_K": ta.momentum.stoch(
                    data["high"],
                    data["low"],
                    data["close"],
                    window=k_period,
                    smooth_window=d_period,
                ),
                "Stoch_D": ta.momentum.stoch_signal(
                    data["high"],
                    data["low"],
                    data["close"],
                    window=k_period,
                    smooth_window=d_period,
                ),
            },
            index=data.index,
        )

        result_data["Stoch_Overbought"] = (
            (result_data["Stoch_K"] > 80) & (result_data["Stoch_D"] > 80)
        ).astype(int)
        result_data["Stoch_Oversold"] = (
            (result_data["Stoch_K"] < 20) & (result_data["Stoch_D"] < 20)
        ).astype(int)
        result_data["Stoch_K_Above_D"] = (result_data["Stoch_K"] > result_data["Stoch_D"]).astype(
            int
        )
        result_data["Stoch_Crossover"] = result_data["Stoch_K_Above_D"].diff()

        metadata = {
            "indicator_type": "momentum",
            "indicator_name": "Stochastic Oscillator",
            "parameters": {"k_period": k_period, "d_period": d_period},
            "data_points": len(data),
            "features": list(result_data.columns),
            "thresholds": {"overbought": 80, "oversold": 20},
        }

        return create_indicator_result(data=result_data, metadata=metadata, warnings=warnings)
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {str(e)}")
        raise


class MomentumIndicatorCalculator(BaseIndicator):
    """Comprehensive momentum indicator calculator"""

    def calculate(self) -> IndicatorResult:
        """Calculate all momentum indicators"""
        results = [
            func(self.data)
            for func in [calculate_rsi, calculate_stochastic, calculate_roc, calculate_williams_r]
        ]
        combined_data = pd.concat([r.data for r in results], axis=1)
        all_warnings = [w for r in results for w in r.warnings]

        return create_indicator_result(
            data=combined_data,
            metadata={
                "indicator_type": "momentum_combined",
                "indicator_name": "All Momentum Indicators",
                "components": ["RSI", "Stochastic", "ROC", "Williams_R"],
                "data_points": len(self.data),
                "total_features": len(combined_data.columns),
            },
            warnings=all_warnings,
        )
