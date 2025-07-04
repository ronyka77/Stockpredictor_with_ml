"""
Technical Indicators Package

This package provides comprehensive technical indicator calculations
using ta library for financial time series analysis.

Indicator Categories:
- Trend: SMA, EMA, MACD, Ichimoku
- Momentum: RSI, Stochastic, ROC, Williams %R  
- Volatility: Bollinger Bands, ATR, custom volatility measures
- Volume: OBV, VPT, A/D Line, Volume Profile, Money Flow Index

Usage:
    from src.feature_engineering.technical_indicators import calculate_all_indicators
    
    indicators = calculate_all_indicators(data, timeframe='1D')
"""

from src.feature_engineering.technical_indicators.base import BaseIndicator, IndicatorResult
from src.feature_engineering.technical_indicators.trend_indicators import (
    calculate_sma, 
    calculate_ema, 
    calculate_macd,
    calculate_ichimoku,
    TrendIndicatorCalculator
)
from src.feature_engineering.technical_indicators.momentum_indicators import (
    calculate_rsi,
    calculate_stochastic,
    calculate_roc,
    calculate_williams_r,
    MomentumIndicatorCalculator
)
from src.feature_engineering.technical_indicators.volatility_indicators import (
    calculate_bollinger_bands,
    calculate_atr,
    calculate_custom_volatility,
    VolatilityIndicatorCalculator
)
from src.feature_engineering.technical_indicators.volume_indicators import (
    calculate_obv,
    calculate_vpt,
    calculate_ad_line,
    calculate_volume_profile,
    calculate_money_flow_index,
    VolumeIndicatorCalculator
)

__all__ = [
    # Base classes
    'BaseIndicator',
    'IndicatorResult',
    
    # Trend indicators
    'calculate_sma',
    'calculate_ema', 
    'calculate_macd',
    'calculate_ichimoku',
    'TrendIndicatorCalculator',
    
    # Momentum indicators
    'calculate_rsi',
    'calculate_stochastic',
    'calculate_roc',
    'calculate_williams_r',
    'MomentumIndicatorCalculator',
    
    # Volatility indicators
    'calculate_bollinger_bands',
    'calculate_atr',
    'calculate_custom_volatility',
    'VolatilityIndicatorCalculator',
    
    # Volume indicators
    'calculate_obv',
    'calculate_vpt',
    'calculate_ad_line',
    'calculate_volume_profile',
    'calculate_money_flow_index',
    'VolumeIndicatorCalculator'
] 