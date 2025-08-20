"""
Volatility Technical Indicators

This module implements volatility-based technical indicators including
Bollinger Bands, Average True Range, and custom volatility measures.
"""

import pandas as pd
import numpy as np
import time
from typing import List, Optional

try:
    import ta
except ImportError:
    raise ImportError("ta is required. Install with: pip install ta")

from src.data_collector.indicator_pipeline.base import BaseIndicator, IndicatorResult, create_indicator_result
from src.utils.logger import get_logger
from src.data_collector.config import feature_config

logger = get_logger(__name__, utility='feature_engineering')

def calculate_bollinger_bands(data: pd.DataFrame, period: Optional[int] = None,
                                std_dev: Optional[float] = None) -> IndicatorResult:
    """
    Calculate Bollinger Bands
    
    Args:
        data: OHLCV DataFrame
        period: Period for moving average
        std_dev: Standard deviation multiplier
        
    Returns:
        IndicatorResult containing Bollinger Bands values
    """
    start_time = time.time()
    warnings = []
    
    # Use config defaults if not provided
    params = feature_config.BOLLINGER_PARAMS
    if period is None:
        period = params['period']
    if std_dev is None:
        std_dev = params['std']
    
    logger.info(f"Calculating Bollinger Bands with parameters: period={period}, std_dev={std_dev}")
    
    try:
        # Check minimum data requirements
        if len(data) < period:
            warning_msg = f"Insufficient data for Bollinger Bands. Need {period} points, have {len(data)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Calculate Bollinger Bands using ta library
        bb_upper = ta.volatility.bollinger_hband(data['close'], window=period, window_dev=std_dev)
        bb_middle = ta.volatility.bollinger_mavg(data['close'], window=period)
        bb_lower = ta.volatility.bollinger_lband(data['close'], window=period, window_dev=std_dev)
        bb_width = ta.volatility.bollinger_wband(data['close'], window=period, window_dev=std_dev)
        bb_percent = ta.volatility.bollinger_pband(data['close'], window=period, window_dev=std_dev)
        
        result_data = pd.DataFrame(index=data.index)
        result_data['BB_Lower'] = bb_lower
        result_data['BB_Middle'] = bb_middle
        result_data['BB_Upper'] = bb_upper
        result_data['BB_Bandwidth'] = bb_width
        result_data['BB_Percent'] = bb_percent
        
        # Calculate additional Bollinger Band features
        result_data['BB_Width'] = result_data['BB_Upper'] - result_data['BB_Lower']
        result_data['BB_Above_Upper'] = (data['close'] > result_data['BB_Upper']).astype(int)
        result_data['BB_Below_Lower'] = (data['close'] < result_data['BB_Lower']).astype(int)
        result_data['BB_Squeeze'] = (result_data['BB_Width'] < result_data['BB_Width'].rolling(20).mean()).astype(int)
        
        # Price position within bands
        result_data['BB_Position'] = (
            (data['close'] - result_data['BB_Lower']) / 
            (result_data['BB_Upper'] - result_data['BB_Lower'])
        )
        
        # logger.info(f"Calculated Bollinger Bands: {result_data['BB_Middle'].notna().sum()} valid values")
        
        metadata = {
            'indicator_type': 'volatility',
            'indicator_name': 'Bollinger Bands',
            'parameters': {'period': period, 'std_dev': std_dev},
            'data_points': len(data),
            'features': [
                'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Bandwidth', 'BB_Percent',
                'BB_Width', 'BB_Above_Upper', 'BB_Below_Lower', 'BB_Squeeze', 'BB_Position'
            ]
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        raise

def calculate_atr(data: pd.DataFrame, period: Optional[int] = None) -> IndicatorResult:
    """
    Calculate Average True Range
    
    Args:
        data: OHLCV DataFrame
        period: Period for ATR calculation
        
    Returns:
        IndicatorResult containing ATR values
    """
    start_time = time.time()
    warnings = []
    
    if period is None:
        period = feature_config.ATR_PERIOD
    
    logger.info(f"Calculating ATR with period: {period}")
    
    try:
        # Check minimum data requirements
        if len(data) < period:
            warning_msg = f"Insufficient data for ATR. Need {period} points, have {len(data)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Calculate ATR using ta library
        atr_values = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=period)
        
        result_data = pd.DataFrame(index=data.index)
        result_data['ATR'] = atr_values
        
        # Calculate additional ATR-based features
        result_data['ATR_Percent'] = (atr_values / data['close']) * 100
        result_data['ATR_High_Volatility'] = (
            atr_values > atr_values.rolling(period * 2).mean() * 1.5
        ).astype(int)
        result_data['ATR_Low_Volatility'] = (
            atr_values < atr_values.rolling(period * 2).mean() * 0.5
        ).astype(int)
        
        # ATR-based support and resistance levels
        result_data['ATR_Upper_Band'] = data['close'] + (atr_values * 2)
        result_data['ATR_Lower_Band'] = data['close'] - (atr_values * 2)
        
        # logger.info(f"Calculated ATR: {atr_values.notna().sum()} valid values")
        
        metadata = {
            'indicator_type': 'volatility',
            'indicator_name': 'Average True Range',
            'parameters': {'period': period},
            'data_points': len(data),
            'features': [
                'ATR', 'ATR_Percent', 'ATR_High_Volatility', 'ATR_Low_Volatility',
                'ATR_Upper_Band', 'ATR_Lower_Band'
            ]
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        raise

def calculate_custom_volatility(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """
    Calculate custom volatility measures and regime detection
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods for volatility calculation
        
    Returns:
        IndicatorResult containing custom volatility measures
    """
    start_time = time.time()
    warnings = []
    
    if periods is None:
        periods = [10, 20, 30]  # Default volatility periods
    
    logger.info(f"Calculating custom volatility for periods: {periods}")
    
    try:
        result_data = pd.DataFrame(index=data.index)
        
        # Calculate returns for volatility measures
        returns = data['close'].pct_change()
        
        for period in periods:
            if len(data) < period:
                warning_msg = f"Insufficient data for volatility_{period}. Need {period} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            # Rolling standard deviation of returns (volatility)
            vol_std = returns.rolling(period).std()
            result_data[f'Volatility_Std_{period}'] = vol_std
            
            # Annualized volatility (assuming 252 trading days)
            result_data[f'Volatility_Annualized_{period}'] = vol_std * np.sqrt(252)
            
            # High-Low volatility measure
            hl_volatility = ((data['high'] - data['low']) / data['close']).rolling(period).mean()
            result_data[f'HL_Volatility_{period}'] = hl_volatility
            
            # Parkinson volatility estimator
            parkinson_vol = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(data['high'] / data['low']) ** 2).rolling(period).mean()
            )
            result_data[f'Parkinson_Vol_{period}'] = parkinson_vol
            
            # logger.info(f"Calculated volatility measures for period {period}")
        
        # Volatility regime detection
        if len(periods) > 0:
            short_vol = result_data[f'Volatility_Std_{periods[0]}']
            long_vol = result_data[f'Volatility_Std_{periods[-1]}']
            
            result_data['Vol_Regime_High'] = (
                short_vol > short_vol.rolling(50).quantile(0.8)
            ).astype(int)
            
            result_data['Vol_Regime_Low'] = (
                short_vol < short_vol.rolling(50).quantile(0.2)
            ).astype(int)
            
            result_data['Vol_Trend_Rising'] = (short_vol > long_vol).astype(int)
        
        # GARCH-like volatility clustering
        if len(data) >= 30:
            vol_clustering = returns.rolling(20).std().rolling(10).std()
            result_data['Vol_Clustering'] = vol_clustering
            result_data['Vol_Clustering_High'] = (
                vol_clustering > vol_clustering.rolling(50).quantile(0.7)
            ).astype(int)
        
        if result_data.empty:
            raise ValueError("No volatility indicators could be calculated")
        
        metadata = {
            'indicator_type': 'volatility',
            'indicator_name': 'Custom Volatility Measures',
            'periods': periods,
            'data_points': len(data),
            'features': [col for col in result_data.columns],
            'methods': ['Standard Deviation', 'High-Low', 'Parkinson', 'Regime Detection', 'Clustering']
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating custom volatility: {str(e)}")
        raise

class VolatilityIndicatorCalculator(BaseIndicator):
    """Calculator for all volatility-based technical indicators"""
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
    
    def calculate(self) -> IndicatorResult:
        """
        Calculate all volatility indicators and combine results
        
        Returns:
            IndicatorResult containing all volatility indicators
        """
        start_time = time.time()
        
        try:
            # Calculate individual indicators
            bb_result = calculate_bollinger_bands(self.data)
            atr_result = calculate_atr(self.data)
            custom_vol_result = calculate_custom_volatility(self.data)
            
            # Combine all results
            combined_data = pd.concat([
                bb_result.data,
                atr_result.data,
                custom_vol_result.data
            ], axis=1)
            
            # Combine warnings
            all_warnings = (
                bb_result.warnings + atr_result.warnings + 
                custom_vol_result.warnings
            )
            
            # Create combined metadata
            metadata = {
                'indicator_type': 'volatility',
                'indicator_name': 'Combined Volatility Indicators',
                'data_points': len(self.data),
                'total_features': len(combined_data.columns),
                'individual_results': {
                    'bollinger_bands': bb_result.metadata,
                    'atr': atr_result.metadata,
                    'custom_volatility': custom_vol_result.metadata
                }
            }
            
            calculation_time = time.time() - start_time
            
            return create_indicator_result(
                data=combined_data,
                metadata=metadata,
                warnings=all_warnings,
                calculation_time=calculation_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            raise 