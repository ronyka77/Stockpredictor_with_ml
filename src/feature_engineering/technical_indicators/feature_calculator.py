"""
Feature Calculator for Technical Indicators

This module provides the main feature calculation engine that orchestrates
all technical indicator calculations and provides a unified interface for
feature generation.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any

from src.feature_engineering.technical_indicators.trend_indicators import TrendIndicatorCalculator
from src.feature_engineering.technical_indicators.momentum_indicators import MomentumIndicatorCalculator
from src.feature_engineering.technical_indicators.volatility_indicators import VolatilityIndicatorCalculator
from src.feature_engineering.technical_indicators.volume_indicators import VolumeIndicatorCalculator
from src.feature_engineering.technical_indicators.base import IndicatorResult, IndicatorValidator

from src.utils.logger import get_logger
from src.feature_engineering.config import config

logger = get_logger(__name__, utility='feature_engineering')

class FeatureCalculator:
    """
    Main feature calculator that orchestrates all technical indicator calculations
    
    This class provides a unified interface for calculating all types of technical
    indicators and combines them into a comprehensive feature set.
    """
    
    def __init__(self, fe_config: Optional[Any] = None):
        """
        Initialize the feature calculator
        
        Args:
            fe_config: Configuration object (uses default if None)
        """
        self.config = fe_config or config
        self.validator = IndicatorValidator()
        
        logger.info("Initialized FeatureCalculator")
    
    def calculate_all_features(self, data: pd.DataFrame, 
                                include_categories: Optional[List[str]] = None) -> IndicatorResult:
        """
        Calculate all technical indicators for the given data
        
        Args:
            data: OHLCV DataFrame
            include_categories: List of indicator categories to include
                                Options: ['trend', 'momentum', 'volatility', 'volume']
                                If None, includes all categories
        
        Returns:
            IndicatorResult containing all calculated features
        """
        start_time = time.time()
        all_warnings = []
        
        if include_categories is None:
            include_categories = ['trend', 'momentum', 'volatility', 'volume']
        
        try:
            combined_data = pd.DataFrame(data=data, index=data.index)
            calculated_components = []
            
            # Calculate trend indicators
            if 'trend' in include_categories:
                # logger.info("Calculating trend indicators...")
                trend_calc = TrendIndicatorCalculator(data)
                trend_result = trend_calc.calculate()
                
                if self.validator.validate_result(trend_result):
                    combined_data = pd.concat([combined_data, trend_result.data], axis=1)
                    calculated_components.append('trend')
                    all_warnings.extend(trend_result.warnings)
                    # logger.info(f"Added {len(trend_result.data.columns)} trend features")
                else:
                    logger.warning("Trend indicators failed validation")
            
            # Calculate momentum indicators
            if 'momentum' in include_categories:
                # logger.info("Calculating momentum indicators...")
                momentum_calc = MomentumIndicatorCalculator(data)
                momentum_result = momentum_calc.calculate()
                
                if self.validator.validate_result(momentum_result):
                    combined_data = pd.concat([combined_data, momentum_result.data], axis=1)
                    calculated_components.append('momentum')
                    all_warnings.extend(momentum_result.warnings)
                    # logger.info(f"Added {len(momentum_result.data.columns)} momentum features")
                else:
                    logger.warning("Momentum indicators failed validation")
            
            # Calculate volatility indicators
            if 'volatility' in include_categories:
                # logger.info("Calculating volatility indicators...")
                volatility_calc = VolatilityIndicatorCalculator(data)
                volatility_result = volatility_calc.calculate()
                
                if self.validator.validate_result(volatility_result):
                    combined_data = pd.concat([combined_data, volatility_result.data], axis=1)
                    calculated_components.append('volatility')
                    all_warnings.extend(volatility_result.warnings)
                    # logger.info(f"Added {len(volatility_result.data.columns)} volatility features")
                else:
                    logger.warning("Volatility indicators failed validation")
            
            # Calculate volume indicators
            if 'volume' in include_categories:
                # logger.info("Calculating volume indicators...")
                volume_calc = VolumeIndicatorCalculator(data)
                volume_result = volume_calc.calculate()
                
                if self.validator.validate_result(volume_result):
                    combined_data = pd.concat([combined_data, volume_result.data], axis=1)
                    calculated_components.append('volume')
                    all_warnings.extend(volume_result.warnings)
                    # logger.info(f"Added {len(volume_result.data.columns)} volume features")
                else:
                    logger.warning("Volume indicators failed validation")
            
            if combined_data.empty:
                raise ValueError("No valid features could be calculated")
            
            # Add basic price features
            combined_data = self._add_basic_price_features(combined_data, data)
            
            # Add future price targets
            combined_data = self._add_future_price_targets(combined_data, data)
            
            # Calculate quality score
            quality_score = self._calculate_overall_quality_score(combined_data)
            
            metadata = {
                'indicator_type': 'all_features',
                'indicator_name': 'Complete Technical Feature Set',
                'components': calculated_components,
                'requested_categories': include_categories,
                'data_points': len(data),
                'total_features': len(combined_data.columns),
                'feature_categories': {
                    'trend': len([col for col in combined_data.columns if any(trend in col.lower() for trend in ['sma', 'ema', 'macd', 'ichimoku'])]),
                    'momentum': len([col for col in combined_data.columns if any(mom in col.lower() for mom in ['rsi', 'stoch', 'roc', 'williams'])]),
                    'volatility': len([col for col in combined_data.columns if any(vol in col.lower() for vol in ['bb_', 'atr', 'volatility'])]),
                    'volume': len([col for col in combined_data.columns if any(vol in col.lower() for vol in ['obv', 'vpt', 'ad_'])]),
                    'basic': len([col for col in combined_data.columns if any(basic in col.lower() for basic in ['price_', 'return_', 'range_'])]),
                    'future_targets': len([col for col in combined_data.columns if 'future_' in col.lower()])
                }
            }
            
            calculation_time = time.time() - start_time
            
            # logger.info(f"Feature calculation completed in {calculation_time:.2f} seconds")
            # logger.info(f"Generated {len(combined_data.columns)} total features")
            
            return IndicatorResult(
                data=combined_data,
                metadata=metadata,
                quality_score=quality_score,
                warnings=all_warnings,
                calculation_time=calculation_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise
    
    def calculate_category_features(self, data: pd.DataFrame, 
                                    category: str) -> IndicatorResult:
        """
        Calculate features for a specific category
        
        Args:
            data: OHLCV DataFrame
            category: Feature category ('trend', 'momentum', 'volatility', 'volume')
        
        Returns:
            IndicatorResult for the specified category
        """
        # logger.info(f"Calculating {category} features")
        
        if category == 'trend':
            calc = TrendIndicatorCalculator(data)
        elif category == 'momentum':
            calc = MomentumIndicatorCalculator(data)
        elif category == 'volatility':
            calc = VolatilityIndicatorCalculator(data)
        elif category == 'volume':
            calc = VolumeIndicatorCalculator(data)
        else:
            raise ValueError(f"Unknown category: {category}")
        
        return calc.calculate()
    
    def _add_future_price_targets(self, features_df: pd.DataFrame, 
                                    price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add future price targets for +10 days and +30 days
        
        Args:
            features_df: Existing features DataFrame
            price_data: Original OHLCV data
        
        Returns:
            DataFrame with future price target features
        """
        logger.debug("Adding future price targets")
        
        # Future high prices (targets for prediction)
        features_df['Future_High_10D'] = price_data['high'].shift(-10)
        features_df['Future_High_30D'] = price_data['high'].shift(-30)
        
        # Future close prices (alternative targets)
        features_df['Future_Close_10D'] = price_data['close'].shift(-10)
        features_df['Future_Close_30D'] = price_data['close'].shift(-30)
        logger.debug(f"Added {len([col for col in features_df.columns if 'Future_' in col])} future price target features")
        
        return features_df

    def _add_basic_price_features(self, features_df: pd.DataFrame, 
                                    price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-based features
        
        Args:
            features_df: Existing features DataFrame
            price_data: Original OHLCV data
        
        Returns:
            DataFrame with additional basic features
        """
        logger.debug("Adding basic price features")
        
        # Price ratios
        features_df['Price_High_Low_Ratio'] = price_data['high'] / price_data['low']
        features_df['Price_Close_Open_Ratio'] = price_data['close'] / price_data['open']
        
        # Price ranges
        features_df['Price_Range'] = price_data['high'] - price_data['low']
        features_df['Price_Range_Pct'] = (price_data['high'] - price_data['low']) / price_data['close']
        
        # Body and shadow ratios (candlestick analysis)
        body = abs(price_data['close'] - price_data['open'])
        upper_shadow = price_data['high'] - np.maximum(price_data['close'], price_data['open'])
        lower_shadow = np.minimum(price_data['close'], price_data['open']) - price_data['low']
        
        features_df['Body_Size'] = body
        features_df['Upper_Shadow'] = upper_shadow
        features_df['Lower_Shadow'] = lower_shadow
        features_df['Body_Shadow_Ratio'] = body / (upper_shadow + lower_shadow + 1e-8)
        
        # Returns
        features_df['Return_1D'] = price_data['close'].pct_change()
        features_df['Return_5D'] = price_data['close'].pct_change(5)
        features_df['Return_10D'] = price_data['close'].pct_change(10)
        features_df['Return_20D'] = price_data['close'].pct_change(20)
        
        # Log returns
        features_df['Log_Return_1D'] = np.log(price_data['close'] / price_data['close'].shift(1))
        features_df['Log_Return_5D'] = np.log(price_data['close'] / price_data['close'].shift(5))
        
        # Volume features
        if 'volume' in price_data.columns:
            features_df['Volume_Price_Ratio'] = price_data['volume'] / price_data['close']
            features_df['Volume_Range_Ratio'] = price_data['volume'] / features_df['Price_Range']
        
        # Gap analysis
        features_df['Gap_Up'] = (price_data['open'] > price_data['high'].shift(1)).astype(int)
        features_df['Gap_Down'] = (price_data['open'] < price_data['low'].shift(1)).astype(int)
        features_df['Gap_Size'] = price_data['open'] - price_data['close'].shift(1)
        
        return features_df
    
    def _calculate_overall_quality_score(self, features_df: pd.DataFrame) -> float:
        """
        Calculate overall quality score for the feature set
        
        Args:
            features_df: Complete features DataFrame
        
        Returns:
            Quality score between 0-100
        """
        if features_df.empty:
            return 0.0
        
        # Calculate missing data percentage
        total_values = features_df.size
        missing_values = features_df.isna().sum().sum()
        missing_pct = missing_values / total_values
        
        # Calculate infinite values percentage
        numeric_df = features_df.select_dtypes(include=[np.number])
        infinite_values = np.isinf(numeric_df).sum().sum()
        infinite_pct = infinite_values / numeric_df.size if numeric_df.size > 0 else 0
        
        # Calculate outlier percentage (simplified)
        outlier_count = 0
        total_numeric_values = 0
        
        for col in numeric_df.columns:
            values = numeric_df[col].dropna()
            if len(values) > 10:  # Need sufficient data for outlier detection
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                outliers = ((values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr))).sum()
                outlier_count += outliers
                total_numeric_values += len(values)
        
        outlier_pct = outlier_count / total_numeric_values if total_numeric_values > 0 else 0
        
        # Calculate overall quality score
        quality_score = 100 * (1 - missing_pct) * (1 - infinite_pct) * (1 - min(outlier_pct, 0.5))
        
        logger.debug(f"Quality score calculation: missing={missing_pct:.3f}, infinite={infinite_pct:.3f}, outliers={outlier_pct:.3f}, score={quality_score:.1f}")
        
        return max(0.0, min(100.0, quality_score))
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the calculated features
        
        Args:
            features_df: Features DataFrame
        
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': len(features_df.columns),
            'total_rows': len(features_df),
            'missing_values': features_df.isna().sum().sum(),
            'missing_percentage': (features_df.isna().sum().sum() / features_df.size) * 100,
            'feature_types': {
                'numeric': len(features_df.select_dtypes(include=[np.number]).columns),
                'categorical': len(features_df.select_dtypes(include=['object', 'category']).columns)
            },
            'date_range': {
                'start': features_df.index.min(),
                'end': features_df.index.max()
            }
        }
        
        # Feature category breakdown
        categories = {
            'trend': ['sma', 'ema', 'macd', 'ichimoku'],
            'momentum': ['rsi', 'stoch', 'roc', 'williams'],
            'volatility': ['bb_', 'atr', 'volatility'],
            'volume': ['obv', 'vpt', 'ad_'],
            'basic': ['price_', 'return_', 'body_', 'gap_'],
            'future_targets': ['future_']
        }
        
        category_counts = {}
        for category, keywords in categories.items():
            count = len([col for col in features_df.columns 
                        if any(keyword in col.lower() for keyword in keywords)])
            category_counts[category] = count
        
        summary['feature_categories'] = category_counts
        
        return summary 