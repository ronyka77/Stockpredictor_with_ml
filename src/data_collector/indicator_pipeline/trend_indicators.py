"""
Trend Technical Indicators

This module implements trend-following technical indicators including
moving averages, MACD, and Ichimoku cloud components.
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

def calculate_sma(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """
    Calculate Simple Moving Averages for multiple periods
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods for SMA calculation
        
    Returns:
        IndicatorResult containing SMA values
    """
    start_time = time.time()
    warnings = []
    
    if periods is None:
        periods = feature_config.SMA_PERIODS
    
    # logger.info(f"Calculating SMA for periods: {periods}")
    
    try:
        result_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            if len(data) < period:
                warning_msg = f"Insufficient data for SMA_{period}. Need {period} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            sma_values = ta.trend.sma_indicator(data['close'], window=period)
            result_data[f'SMA_{period}'] = sma_values
            
            # logger.info(f"Calculated SMA_{period}: {sma_values.notna().sum()} valid values")
        
        if result_data.empty:
            raise ValueError("No SMA indicators could be calculated")
        
        metadata = {
            'indicator_type': 'trend',
            'indicator_name': 'Simple Moving Average',
            'periods': periods,
            'data_points': len(data),
            'valid_calculations': len([p for p in periods if len(data) >= p])
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}")
        raise

def calculate_ema(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """
    Calculate Exponential Moving Averages for multiple periods
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods for EMA calculation
        
    Returns:
        IndicatorResult containing EMA values
    """
    start_time = time.time()
    warnings = []
    
    if periods is None:
        periods = feature_config.EMA_PERIODS
    
    # logger.info(f"Calculating EMA for periods: {periods}")
    
    try:
        result_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            if len(data) < period:
                warning_msg = f"Insufficient data for EMA_{period}. Need {period} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            ema_values = ta.trend.ema_indicator(data['close'], window=period)
            result_data[f'EMA_{period}'] = ema_values
            
            # logger.info(f"Calculated EMA_{period}: {ema_values.notna().sum()} valid values")
        
        if result_data.empty:
            raise ValueError("No EMA indicators could be calculated")
        
        metadata = {
            'indicator_type': 'trend',
            'indicator_name': 'Exponential Moving Average',
            'periods': periods,
            'data_points': len(data),
            'valid_calculations': len([p for p in periods if len(data) >= p])
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        raise

def calculate_macd(data: pd.DataFrame, fast: Optional[int] = None, 
                    slow: Optional[int] = None, signal: Optional[int] = None) -> IndicatorResult:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: OHLCV DataFrame
        fast: Fast EMA period
        slow: Slow EMA period  
        signal: Signal line EMA period
        
    Returns:
        IndicatorResult containing MACD, signal line, and histogram
    """
    start_time = time.time()
    warnings = []
    
    # Use config defaults if not provided
    if fast is None:
        fast = feature_config.MACD_PARAMS['fast']
    if slow is None:
        slow = feature_config.MACD_PARAMS['slow']
    if signal is None:
        signal = feature_config.MACD_PARAMS['signal']
    
    # logger.info(f"Calculating MACD with parameters: fast={fast}, slow={slow}, signal={signal}")
    
    try:
        # Check minimum data requirements
        min_required = slow + signal
        if len(data) < min_required:
            warning_msg = f"Insufficient data for MACD. Need {min_required} points, have {len(data)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Calculate MACD using ta library
        macd_line = ta.trend.macd(data['close'], window_slow=slow, window_fast=fast)
        macd_signal = ta.trend.macd_signal(data['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        macd_histogram = ta.trend.macd_diff(data['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        
        result_data = pd.DataFrame(index=data.index)
        result_data['MACD'] = macd_line
        result_data['MACD_Signal'] = macd_signal
        result_data['MACD_Histogram'] = macd_histogram
        
        # Calculate additional MACD features
        result_data['MACD_Above_Signal'] = (result_data['MACD'] > result_data['MACD_Signal']).astype(int)
        result_data['MACD_Crossover'] = (result_data['MACD'] > result_data['MACD_Signal']).astype(int).diff()
        
        # logger.info(f"Calculated MACD: {result_data['MACD'].notna().sum()} valid values")
        
        metadata = {
            'indicator_type': 'trend',
            'indicator_name': 'MACD',
            'parameters': {'fast': fast, 'slow': slow, 'signal': signal},
            'data_points': len(data),
            'features': ['MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Above_Signal', 'MACD_Crossover']
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        raise

def calculate_ichimoku(data: pd.DataFrame, tenkan: Optional[int] = None,
                      kijun: Optional[int] = None, senkou_b: Optional[int] = None,
                      displacement: Optional[int] = None) -> IndicatorResult:
    """
    Calculate Ichimoku Cloud components
    
    Args:
        data: OHLCV DataFrame
        tenkan: Tenkan-sen period
        kijun: Kijun-sen period
        senkou_b: Senkou Span B period
        displacement: Displacement for cloud
        
    Returns:
        IndicatorResult containing Ichimoku components
    """
    start_time = time.time()
    warnings = []
    
    # Use config defaults if not provided
    params = feature_config.ICHIMOKU_PARAMS
    if tenkan is None:
        tenkan = params['tenkan']
    if kijun is None:
        kijun = params['kijun']
    if senkou_b is None:
        senkou_b = params['senkou_b']
    if displacement is None:
        displacement = params['displacement']
    
    logger.info(f"Calculating Ichimoku with parameters: tenkan={tenkan}, kijun={kijun}, senkou_b={senkou_b}, displacement={displacement}")
    
    try:
        # Check minimum data requirements
        min_required = max(tenkan, kijun, senkou_b) + displacement
        if len(data) < min_required:
            warning_msg = f"Insufficient data for Ichimoku. Need {min_required} points, have {len(data)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Calculate Ichimoku components using ta library
        ichimoku_a = ta.trend.ichimoku_a(data['high'], data['low'], window1=tenkan, window2=kijun)
        ichimoku_b = ta.trend.ichimoku_b(data['high'], data['low'], window2=kijun, window3=senkou_b)
        ichimoku_base = ta.trend.ichimoku_base_line(data['high'], data['low'], window1=tenkan, window2=kijun)
        ichimoku_conv = ta.trend.ichimoku_conversion_line(data['high'], data['low'], window1=tenkan)
        
        result_data = pd.DataFrame(index=data.index)
        result_data['Ichimoku_Tenkan'] = ichimoku_conv
        result_data['Ichimoku_Kijun'] = ichimoku_base
        result_data['Ichimoku_Senkou_A'] = ichimoku_a.shift(displacement)
        result_data['Ichimoku_Senkou_B'] = ichimoku_b.shift(displacement)
        
        # Calculate Chikou Span (lagging span)
        result_data['Ichimoku_Chikou'] = data['close'].shift(-displacement)
        
        # Calculate additional Ichimoku signals
        result_data['Ichimoku_Tenkan_Above_Kijun'] = (
            result_data['Ichimoku_Tenkan'] > result_data['Ichimoku_Kijun']
        ).astype(int)
        
        result_data['Ichimoku_Price_Above_Cloud'] = (
            data['close'] > np.maximum(result_data['Ichimoku_Senkou_A'], result_data['Ichimoku_Senkou_B'])
        ).astype(int)
        
        result_data['Ichimoku_Price_Below_Cloud'] = (
            data['close'] < np.minimum(result_data['Ichimoku_Senkou_A'], result_data['Ichimoku_Senkou_B'])
        ).astype(int)
        
        result_data['Ichimoku_Cloud_Green'] = (
            result_data['Ichimoku_Senkou_A'] > result_data['Ichimoku_Senkou_B']
        ).astype(int)
        
        # Cloud thickness (volatility measure)
        result_data['Ichimoku_Cloud_Thickness'] = abs(
            result_data['Ichimoku_Senkou_A'] - result_data['Ichimoku_Senkou_B']
        )
        
        # logger.info(f"Calculated Ichimoku: {result_data['Ichimoku_Tenkan'].notna().sum()} valid values")
        
        metadata = {
            'indicator_type': 'trend',
            'indicator_name': 'Ichimoku Cloud',
            'parameters': {
                'tenkan': tenkan, 'kijun': kijun, 
                'senkou_b': senkou_b, 'displacement': displacement
            },
            'data_points': len(data),
            'features': [
                'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B',
                'Ichimoku_Chikou', 'Ichimoku_Tenkan_Above_Kijun', 'Ichimoku_Price_Above_Cloud',
                'Ichimoku_Price_Below_Cloud', 'Ichimoku_Cloud_Green', 'Ichimoku_Cloud_Thickness'
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
        logger.error(f"Error calculating Ichimoku: {str(e)}")
        raise

class TrendIndicatorCalculator(BaseIndicator):
    """Calculator for all trend-based technical indicators"""
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
    
    def calculate(self) -> IndicatorResult:
        """
        Calculate all trend indicators and combine results
        
        Returns:
            IndicatorResult containing all trend indicators
        """
        start_time = time.time()
        # logger.info("Calculating all trend indicators")
        
        try:
            # Calculate individual indicators
            sma_result = calculate_sma(self.data)
            ema_result = calculate_ema(self.data)
            macd_result = calculate_macd(self.data)
            ichimoku_result = calculate_ichimoku(self.data)
            
            # Combine all results
            combined_data = pd.concat([
                sma_result.data,
                ema_result.data,
                macd_result.data,
                ichimoku_result.data
            ], axis=1)
            
            # Combine warnings
            all_warnings = (
                sma_result.warnings + ema_result.warnings + 
                macd_result.warnings + ichimoku_result.warnings
            )
            
            # Create combined metadata
            metadata = {
                'indicator_type': 'trend',
                'indicator_name': 'Combined Trend Indicators',
                'data_points': len(self.data),
                'total_features': len(combined_data.columns),
                'individual_results': {
                    'sma': sma_result.metadata,
                    'ema': ema_result.metadata,
                    'macd': macd_result.metadata,
                    'ichimoku': ichimoku_result.metadata
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
            logger.error(f"Error calculating trend indicators: {str(e)}")
            raise 