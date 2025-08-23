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
    from src.data_collector.indicator_pipeline import calculate_all_indicators
    
    indicators = calculate_all_indicators(data, timeframe='1D')
"""