"""
Momentum Technical Indicators

This module implements momentum-based technical indicators including
RSI, Stochastic Oscillator, Rate of Change, and Williams %R.
"""

import pandas as pd
import time
from typing import List, Optional

try:
    import ta
except ImportError:
    raise ImportError("ta is required. Install with: pip install ta")

from src.feature_engineering.technical_indicators.base import BaseIndicator, IndicatorResult, create_indicator_result
from src.utils.logger import get_logger
from src.data_collector.config import feature_config

logger = get_logger(__name__, utility='feature_engineering')

def calculate_rsi(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """
    Calculate Relative Strength Index for multiple periods
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods for RSI calculation
        
    Returns:
        IndicatorResult containing RSI values
    """
    start_time = time.time()
    warnings = []
    
    if periods is None:
        periods = feature_config.RSI_PERIODS
    
    # logger.info(f"Calculating RSI for periods: {periods}")
    
    try:
        result_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            if len(data) < period * 2:  # RSI needs more data for stability
                warning_msg = f"Insufficient data for RSI_{period}. Need {period * 2} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            rsi_values = ta.momentum.rsi(data['close'], window=period)
            result_data[f'RSI_{period}'] = rsi_values
            
            # Add RSI-based signals
            result_data[f'RSI_{period}_Overbought'] = (rsi_values > 70).astype(int)
            result_data[f'RSI_{period}_Oversold'] = (rsi_values < 30).astype(int)
            result_data[f'RSI_{period}_Neutral'] = ((rsi_values >= 30) & (rsi_values <= 70)).astype(int)
            
            # logger.info(f"Calculated RSI_{period}: {rsi_values.notna().sum()} valid values")
        
        if result_data.empty:
            raise ValueError("No RSI indicators could be calculated")
        
        metadata = {
            'indicator_type': 'momentum',
            'indicator_name': 'Relative Strength Index',
            'periods': periods,
            'data_points': len(data),
            'valid_calculations': len([p for p in periods if len(data) >= p * 2]),
            'thresholds': {'overbought': 70, 'oversold': 30}
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        raise

def calculate_stochastic(data: pd.DataFrame, k_period: Optional[int] = None,
                        d_period: Optional[int] = None) -> IndicatorResult:
    """
    Calculate Stochastic Oscillator (%K and %D)
    
    Args:
        data: OHLCV DataFrame
        k_period: %K period
        d_period: %D smoothing period
        
    Returns:
        IndicatorResult containing Stochastic values
    """
    start_time = time.time()
    warnings = []
    
    # Use config defaults if not provided
    params = feature_config.STOCHASTIC_PARAMS
    if k_period is None:
        k_period = params['k_period']
    if d_period is None:
        d_period = params['d_period']
    
    # logger.info(f"Calculating Stochastic with parameters: k_period={k_period}, d_period={d_period}")
    
    try:
        # Check minimum data requirements
        min_required = k_period + d_period
        if len(data) < min_required:
            warning_msg = f"Insufficient data for Stochastic. Need {min_required} points, have {len(data)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Calculate Stochastic using ta library
        stoch_k = ta.momentum.stoch(data['high'], data['low'], data['close'], window=k_period, smooth_window=d_period)
        stoch_d = ta.momentum.stoch_signal(data['high'], data['low'], data['close'], window=k_period, smooth_window=d_period)
        
        result_data = pd.DataFrame(index=data.index)
        result_data['Stoch_K'] = stoch_k
        result_data['Stoch_D'] = stoch_d
        
        # Add Stochastic signals
        result_data['Stoch_Overbought'] = (
            (result_data['Stoch_K'] > 80) & (result_data['Stoch_D'] > 80)
        ).astype(int)
        
        result_data['Stoch_Oversold'] = (
            (result_data['Stoch_K'] < 20) & (result_data['Stoch_D'] < 20)
        ).astype(int)
        
        result_data['Stoch_K_Above_D'] = (
            result_data['Stoch_K'] > result_data['Stoch_D']
        ).astype(int)
        
        result_data['Stoch_Crossover'] = (
            result_data['Stoch_K'] > result_data['Stoch_D']
        ).astype(int).diff()
        
        # logger.info(f"Calculated Stochastic: {result_data['Stoch_K'].notna().sum()} valid values")
        
        metadata = {
            'indicator_type': 'momentum',
            'indicator_name': 'Stochastic Oscillator',
            'parameters': {'k_period': k_period, 'd_period': d_period},
            'data_points': len(data),
            'features': ['Stoch_K', 'Stoch_D', 'Stoch_Overbought', 'Stoch_Oversold', 
                        'Stoch_K_Above_D', 'Stoch_Crossover'],
            'thresholds': {'overbought': 80, 'oversold': 20}
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {str(e)}")
        raise

def calculate_roc(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """
    Calculate Rate of Change for multiple periods
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods for ROC calculation
        
    Returns:
        IndicatorResult containing ROC values
    """
    start_time = time.time()
    warnings = []
    
    if periods is None:
        periods = [10, 20, 30]  # Default ROC periods
    
    # logger.info(f"Calculating ROC for periods: {periods}")
    
    try:
        result_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            if len(data) < period + 1:
                warning_msg = f"Insufficient data for ROC_{period}. Need {period + 1} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            roc_values = ta.momentum.roc(data['close'], window=period)
            result_data[f'ROC_{period}'] = roc_values
            
            # Add ROC-based signals
            result_data[f'ROC_{period}_Positive'] = (roc_values > 0).astype(int)
            result_data[f'ROC_{period}_Strong_Positive'] = (roc_values > 5).astype(int)
            result_data[f'ROC_{period}_Strong_Negative'] = (roc_values < -5).astype(int)
            
            # logger.info(f"Calculated ROC_{period}: {roc_values.notna().sum()} valid values")
        
        if result_data.empty:
            raise ValueError("No ROC indicators could be calculated")
        
        metadata = {
            'indicator_type': 'momentum',
            'indicator_name': 'Rate of Change',
            'periods': periods,
            'data_points': len(data),
            'valid_calculations': len([p for p in periods if len(data) >= p + 1]),
            'thresholds': {'strong_positive': 5, 'strong_negative': -5}
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating ROC: {str(e)}")
        raise

def calculate_williams_r(data: pd.DataFrame, periods: Optional[List[int]] = None) -> IndicatorResult:
    """
    Calculate Williams %R for multiple periods
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods for Williams %R calculation
        
    Returns:
        IndicatorResult containing Williams %R values
    """
    start_time = time.time()
    warnings = []
    
    if periods is None:
        periods = [14, 21]  # Default Williams %R periods
    
    # logger.info(f"Calculating Williams %R for periods: {periods}")
    
    try:
        result_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            if len(data) < period:
                warning_msg = f"Insufficient data for Williams_R_{period}. Need {period} points, have {len(data)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            williams_r_values = ta.momentum.williams_r(data['high'], data['low'], data['close'], lbp=period)
            result_data[f'Williams_R_{period}'] = williams_r_values
            
            # Add Williams %R signals
            result_data[f'Williams_R_{period}_Overbought'] = (williams_r_values > -20).astype(int)
            result_data[f'Williams_R_{period}_Oversold'] = (williams_r_values < -80).astype(int)
            result_data[f'Williams_R_{period}_Neutral'] = (
                (williams_r_values >= -80) & (williams_r_values <= -20)
            ).astype(int)
            
            # logger.info(f"Calculated Williams_R_{period}: {williams_r_values.notna().sum()} valid values")
        
        if result_data.empty:
            raise ValueError("No Williams %R indicators could be calculated")
        
        metadata = {
            'indicator_type': 'momentum',
            'indicator_name': 'Williams %R',
            'periods': periods,
            'data_points': len(data),
            'valid_calculations': len([p for p in periods if len(data) >= p]),
            'thresholds': {'overbought': -20, 'oversold': -80}
        }
        
        calculation_time = time.time() - start_time
        
        return create_indicator_result(
            data=result_data,
            metadata=metadata,
            warnings=warnings,
            calculation_time=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating Williams %R: {str(e)}")
        raise

class MomentumIndicatorCalculator(BaseIndicator):
    """
    Comprehensive momentum indicator calculator
    
    This class combines all momentum indicators into a single calculation
    """
    
    def calculate(self) -> IndicatorResult:
        """Calculate all momentum indicators"""
        start_time = time.time()
        all_warnings = []
        
        # logger.info("Calculating all momentum indicators")
        
        try:
            # Calculate individual indicators
            rsi_result = calculate_rsi(self.data)
            stoch_result = calculate_stochastic(self.data)
            roc_result = calculate_roc(self.data)
            willr_result = calculate_williams_r(self.data)
            
            # Combine all results
            combined_data = pd.DataFrame(index=self.data.index)
            
            # Add all indicator data
            for result in [rsi_result, stoch_result, roc_result, willr_result]:
                combined_data = pd.concat([combined_data, result.data], axis=1)
                all_warnings.extend(result.warnings)
            
            metadata = {
                'indicator_type': 'momentum_combined',
                'indicator_name': 'All Momentum Indicators',
                'components': ['RSI', 'Stochastic', 'ROC', 'Williams_R'],
                'data_points': len(self.data),
                'total_features': len(combined_data.columns)
            }
            
            calculation_time = time.time() - start_time
            
            return create_indicator_result(
                data=combined_data,
                metadata=metadata,
                warnings=all_warnings,
                calculation_time=calculation_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            raise 