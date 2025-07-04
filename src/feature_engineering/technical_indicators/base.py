"""
Base classes for technical indicator calculations

This module provides the foundation classes and interfaces for all
technical indicator implementations in the feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime

from src.utils.logger import get_logger
from src.data_collector.config import feature_config

logger = get_logger(__name__, utility='feature_engineering')

@dataclass
class IndicatorResult:
    """
    Standardized result container for technical indicator calculations
    
    Attributes:
        data: DataFrame containing the calculated indicator values
        metadata: Dictionary containing calculation metadata
        quality_score: Quality score of the calculation (0-100)
        warnings: List of warnings generated during calculation
        calculation_time: Time taken for calculation in seconds
    """
    data: pd.DataFrame
    metadata: Dict[str, Any]
    quality_score: float
    warnings: List[str]
    calculation_time: float
    
    def __post_init__(self):
        """Validate the result after initialization"""
        if self.data.empty:
            logger.warning("IndicatorResult created with empty DataFrame")
        
        if not 0 <= self.quality_score <= 100:
            raise ValueError(f"Quality score must be between 0-100, got {self.quality_score}")

class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators
    
    This class provides the common interface and validation logic
    that all technical indicators must implement.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        """
        Initialize the indicator with data and parameters
        
        Args:
            data: OHLCV DataFrame with required columns
            **params: Indicator-specific parameters
        """
        self.data = data.copy()
        self.params = params
        self.config = feature_config
        
        # Validate input data
        self.validate_data()
        
        # Standardize column names
        self.standardize_columns()
        
        logger.debug(f"Initialized {self.__class__.__name__} with {len(self.data)} data points")
    
    def validate_data(self) -> None:
        """
        Validate that the input data meets requirements
        
        Raises:
            ValueError: If data validation fails
        """
        if self.data.empty:
            raise ValueError("Input data cannot be empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for minimum data points
        min_points = self.config.MIN_DATA_POINTS
        if len(self.data) < min_points:
            raise ValueError(f"Insufficient data points. Need at least {min_points}, got {len(self.data)}")
        
        # Check for data quality
        for col in required_columns:
            if self.data[col].isna().sum() / len(self.data) > self.config.MAX_MISSING_PCT:
                raise ValueError(f"Too many missing values in column {col}")
        
        logger.debug("Data validation passed")
    
    def standardize_columns(self) -> None:
        """Standardize column names to lowercase"""
        column_mapping = {}
        for col in self.data.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col.lower()
        
        if column_mapping:
            self.data = self.data.rename(columns=column_mapping)
            logger.debug(f"Standardized columns: {column_mapping}")
    
    def check_minimum_periods(self, required_periods: int) -> bool:
        """
        Check if we have enough data for the calculation
        
        Args:
            required_periods: Minimum periods required
            
        Returns:
            True if sufficient data, False otherwise
        """
        return len(self.data) >= required_periods
    
    def calculate_quality_score(self, result: pd.DataFrame) -> float:
        """
        Calculate quality score for the indicator result
        
        Args:
            result: Calculated indicator DataFrame
            
        Returns:
            Quality score between 0-100
        """
        if result.empty:
            return 0.0
        
        # Calculate missing data percentage
        missing_pct = result.isna().sum().sum() / (len(result) * len(result.columns))
        
        # Calculate outlier percentage
        outlier_count = 0
        total_values = 0
        
        for col in result.select_dtypes(include=[np.number]).columns:
            values = result[col].dropna()
            if len(values) > 0:
                z_scores = np.abs((values - values.mean()) / values.std())
                outlier_count += (z_scores > self.config.OUTLIER_THRESHOLD).sum()
                total_values += len(values)
        
        outlier_pct = outlier_count / total_values if total_values > 0 else 0
        
        # Calculate quality score (higher is better)
        quality_score = 100 * (1 - missing_pct) * (1 - outlier_pct)
        
        return max(0.0, min(100.0, quality_score))
    
    @abstractmethod
    def calculate(self) -> IndicatorResult:
        """
        Calculate the technical indicator
        
        Returns:
            IndicatorResult containing the calculated values and metadata
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the indicator calculation
        
        Returns:
            Dictionary containing metadata
        """
        return {
            'indicator_name': self.__class__.__name__,
            'parameters': self.params,
            'data_points': len(self.data),
            'date_range': {
                'start': self.data.index.min().isoformat() if not self.data.empty else None,
                'end': self.data.index.max().isoformat() if not self.data.empty else None
            },
            'calculation_timestamp': datetime.now().isoformat()
        }

class IndicatorValidator:
    """
    Utility class for validating indicator calculations and results
    """
    
    @staticmethod
    def validate_result(result: IndicatorResult, min_quality_score: float = 50.0) -> bool:
        """
        Validate an indicator result
        
        Args:
            result: IndicatorResult to validate
            min_quality_score: Minimum acceptable quality score
            
        Returns:
            True if validation passes, False otherwise
        """
        if result.data.empty:
            logger.warning("Validation failed: Empty result data")
            return False
        
        if result.quality_score < min_quality_score:
            logger.warning(f"Validation failed: Quality score {result.quality_score} below threshold {min_quality_score}")
            return False
        
        # Check for infinite or NaN values in critical columns
        numeric_cols = result.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(result.data[col]).any():
                logger.warning(f"Validation failed: Infinite values found in column {col}")
                return False
        
        logger.debug("Indicator result validation passed")
        return True
    
    @staticmethod
    def check_data_continuity(data: pd.DataFrame, max_gap_days: int = 7) -> List[str]:
        """
        Check for gaps in time series data
        
        Args:
            data: DataFrame with datetime index
            max_gap_days: Maximum acceptable gap in days
            
        Returns:
            List of warnings about data gaps
        """
        warnings = []
        
        if len(data) < 2:
            return warnings
        
        # Calculate gaps between consecutive dates
        date_diffs = data.index.to_series().diff().dt.days
        large_gaps = date_diffs[date_diffs > max_gap_days]
        
        if not large_gaps.empty:
            for date, gap in large_gaps.items():
                warnings.append(f"Data gap of {gap} days found at {date}")
        
        return warnings

def create_indicator_result(data: pd.DataFrame, metadata: Dict[str, Any], 
                            warnings: List[str] = None, calculation_time: float = 0.0) -> IndicatorResult:
    """
    Convenience function to create an IndicatorResult with quality scoring
    
    Args:
        data: Calculated indicator data
        metadata: Calculation metadata
        warnings: List of warnings (optional)
        calculation_time: Time taken for calculation
        
    Returns:
        IndicatorResult with calculated quality score
    """
    if warnings is None:
        warnings = []
    
    # Calculate quality score
    if data.empty:
        quality_score = 0.0
    else:
        missing_pct = data.isna().sum().sum() / (len(data) * len(data.columns))
        quality_score = 100 * (1 - missing_pct)
    
    return IndicatorResult(
        data=data,
        metadata=metadata,
        quality_score=quality_score,
        warnings=warnings,
        calculation_time=calculation_time
    ) 