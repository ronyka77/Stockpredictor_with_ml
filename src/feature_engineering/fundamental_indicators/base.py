"""
Base Fundamental Calculator

This module provides the base class for all fundamental analysis calculators,
similar to the technical indicators base class but adapted for fundamental data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import date
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.feature_engineering.config import config
from src.data_collector.polygon_fundamentals.data_models import (
    FundamentalDataResponse
)

logger = get_logger(__name__)

@dataclass
class FundamentalCalculationResult:
    """Result of fundamental calculation"""
    ticker: str
    date: date
    values: Dict[str, Any]
    quality_score: float
    missing_data_count: int
    calculation_errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'ticker': self.ticker,
            'date': self.date,
            'quality_score': self.quality_score,
            'missing_data_count': self.missing_data_count,
            'calculation_errors': self.calculation_errors,
            **self.values
        }

class BaseFundamentalCalculator(ABC):
    """
    Base class for fundamental analysis calculators
    
    This class provides common functionality for all fundamental calculators
    including data validation, error handling, and quality scoring.
    """
    
    def __init__(self, fundamental_config: Optional[Any] = None):
        """
        Initialize the fundamental calculator
        
        Args:
            fundamental_config: Configuration for fundamental calculations
        """
        self.config = fundamental_config or config.fundamental
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Quality tracking
        self.calculation_errors = []
        self.missing_data_count = 0
        
    @abstractmethod
    def calculate(self, data: FundamentalDataResponse) -> FundamentalCalculationResult:
        """
        Calculate fundamental indicators from financial data
        
        Args:
            data: FundamentalDataResponse containing financial statements
            
        Returns:
            FundamentalCalculationResult with calculated values
        """
        pass
    
    @abstractmethod
    def get_required_fields(self) -> Dict[str, List[str]]:
        """
        Get required fields for calculation
        
        Returns:
            Dictionary mapping statement types to required field lists
        """
        pass
    
    def validate_data(self, data: FundamentalDataResponse) -> bool:
        """
        Validate that required data is available
        
        Args:
            data: FundamentalDataResponse to validate
            
        Returns:
            True if data is sufficient for calculation
        """
        required_fields = self.get_required_fields()
        
        # Check if we have minimum required statements
        if not any([data.income_statements, data.balance_sheets, data.cash_flow_statements]):
            self.logger.warning("No financial statements available")
            return False
        
        # Check for required fields in each statement type
        for stmt_type, fields in required_fields.items():
            if stmt_type == 'income_statement' and data.income_statements:
                latest_stmt = data.get_latest_income_statement()
                if not self._check_required_fields(latest_stmt, fields):
                    return False
            
            elif stmt_type == 'balance_sheet' and data.balance_sheets:
                latest_stmt = data.get_latest_balance_sheet()
                if not self._check_required_fields(latest_stmt, fields):
                    return False
            
            elif stmt_type == 'cash_flow' and data.cash_flow_statements:
                latest_stmt = data.get_latest_cash_flow()
                if not self._check_required_fields(latest_stmt, fields):
                    return False
        
        return True
    
    def _check_required_fields(self, statement: Any, required_fields: List[str]) -> bool:
        """Check if required fields are present in a statement"""
        if not statement:
            return False
        
        missing_fields = []
        for field in required_fields:
            if not self.has_financial_value(statement, field):
                missing_fields.append(field)
        
        if missing_fields:
            self.logger.warning(f"Missing required fields: {missing_fields}")
            self.missing_data_count += len(missing_fields)
            return False
        
        return True
    
    def has_financial_value(self, statement: Any, field_name: str) -> bool:
        """
        Check if a financial field has a valid value in the statement
        
        Args:
            statement: Financial statement object
            field_name: Name of the field to check
            
        Returns:
            True if field has a valid value
        """
        if not statement:
            return False
        
        # Try to get the field value
        value = getattr(statement, field_name, None)
        
        # Check if value exists and is not None
        if value is None:
            return False
        
        # Check if it's a FinancialValue object with a valid value
        if hasattr(value, 'value'):
            return value.value is not None
        
        # For direct numeric values
        return True
    
    def safe_divide(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """
        Safely divide two numbers with proper error handling
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            
        Returns:
            Division result or None if invalid
        """
        try:
            if numerator is None or denominator is None:
                return None
            
            if denominator == 0:
                self.calculation_errors.append("Division by zero")
                return None
            
            result = numerator / denominator
            
            # Check for extreme values using config
            max_ratio_value = getattr(self.config, 'MAX_RATIO_VALUE', 1e6)
            if abs(result) > max_ratio_value:
                self.calculation_errors.append(f"Extreme ratio value: {result}")
                return None
            
            return result
            
        except (TypeError, ValueError, ZeroDivisionError) as e:
            self.calculation_errors.append(f"Division error: {str(e)}")
            return None
    
    def get_financial_value(self, financial_value: Any) -> Optional[float]:
        """
        Extract numeric value from FinancialValue object
        
        Args:
            financial_value: FinancialValue object or numeric value
            
        Returns:
            Numeric value or None
        """
        if financial_value is None:
            return None
        
        # Handle FinancialValue objects
        if hasattr(financial_value, 'value'):
            return financial_value.value
        
        # Handle direct numeric values
        if isinstance(financial_value, (int, float)):
            return float(financial_value)
        
        return None
    
    def apply_outlier_capping(self, value: Optional[float], ratio_name: str) -> Optional[float]:
        """
        Apply outlier capping based on configuration
        
        Args:
            value: Value to cap
            ratio_name: Name of the ratio for configuration lookup
            
        Returns:
            Capped value or None
        """
        if value is None or not self.config.OUTLIER_CAPPING:
            return value
        
        # Get capping limits from configuration
        cap_attr = f"{ratio_name.upper()}_CAP"
        if hasattr(self.config, cap_attr):
            min_val, max_val = getattr(self.config, cap_attr)
            
            if value < min_val:
                self.logger.info(f"Capping {ratio_name} from {value} to {min_val}")
                return min_val
            elif value > max_val:
                self.logger.info(f"Capping {ratio_name} from {value} to {max_val}")
                return max_val
        
        return value
    
    def calculate_growth_rate(self, current: Optional[float], previous: Optional[float]) -> Optional[float]:
        """
        Calculate growth rate between two values
        
        Args:
            current: Current period value
            previous: Previous period value
            
        Returns:
            Growth rate as decimal (e.g., 0.1 for 10% growth)
        """
        if current is None or previous is None:
            return None
        
        if previous == 0:
            # Handle zero base case
            if current == 0:
                return 0.0
            else:
                # Infinite growth - return None or a large number
                return None
        
        return (current - previous) / abs(previous)
    
    def calculate_cagr(self, ending_value: Optional[float], beginning_value: Optional[float], periods: int) -> Optional[float]:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            ending_value: Final value
            beginning_value: Initial value
            periods: Number of periods
            
        Returns:
            CAGR as decimal
        """
        if ending_value is None or beginning_value is None or periods <= 0:
            return None
        
        if beginning_value <= 0:
            return None
        
        try:
            return (ending_value / beginning_value) ** (1 / periods) - 1
        except (ValueError, ZeroDivisionError, OverflowError):
            self.calculation_errors.append("CAGR calculation error")
            return None
    
    def calculate_quality_score(self) -> float:
        """
        Calculate data quality score based on missing data and errors
        
        Returns:
            Quality score between 0 and 100
        """
        base_score = 100.0
        
        # Deduct for missing data
        base_score -= self.missing_data_count * 5
        
        # Deduct for calculation errors
        base_score -= len(self.calculation_errors) * 10
        
        return max(0.0, min(100.0, base_score))
    
    def reset_quality_tracking(self):
        """Reset quality tracking for new calculation"""
        self.calculation_errors = []
        self.missing_data_count = 0
    
    def get_latest_statements(self, data: FundamentalDataResponse) -> Dict[str, Any]:
        """
        Get the latest statements from fundamental data
        
        Args:
            data: FundamentalDataResponse
            
        Returns:
            Dictionary with latest statements
        """
        return {
            'income_statement': data.get_latest_income_statement(),
            'balance_sheet': data.get_latest_balance_sheet(),
            'cash_flow': data.get_latest_cash_flow()
        }
    
    def get_statements_by_period(self, data: FundamentalDataResponse, periods: int = 4) -> List[Dict[str, Any]]:
        """
        Get statements for multiple periods for trend analysis
        
        Args:
            data: FundamentalDataResponse
            periods: Number of periods to retrieve
            
        Returns:
            List of statement dictionaries sorted by date (newest first)
        """
        # Collect all statements with dates
        all_statements = []
        
        # Add income statements
        for stmt in data.income_statements:
            if stmt.filing_date:
                all_statements.append({
                    'date': stmt.filing_date,
                    'fiscal_period': stmt.fiscal_period,
                    'fiscal_year': stmt.fiscal_year,
                    'income_statement': stmt,
                    'balance_sheet': None,
                    'cash_flow': None
                })
        
        # Match balance sheets and cash flows by period
        for stmt_data in all_statements:
            period = stmt_data['fiscal_period']
            year = stmt_data['fiscal_year']
            
            # Find matching balance sheet
            for bs in data.balance_sheets:
                if bs.fiscal_period == period and bs.fiscal_year == year:
                    stmt_data['balance_sheet'] = bs
                    break
            
            # Find matching cash flow
            for cf in data.cash_flow_statements:
                if cf.fiscal_period == period and cf.fiscal_year == year:
                    stmt_data['cash_flow'] = cf
                    break
        
        # Sort by date (newest first) and return requested number of periods
        all_statements.sort(key=lambda x: x['date'], reverse=True)
        return all_statements[:periods]
    
    def log_calculation_summary(self, result: FundamentalCalculationResult):
        """Log summary of calculation results"""
        self.logger.info(
            f"Calculated fundamental indicators for {result.ticker} "
            f"(Quality: {result.quality_score:.1f}, "
            f"Missing: {result.missing_data_count}, "
            f"Errors: {len(result.calculation_errors)}) "
            f"values: {list(result.values.keys())}"
        )
        
        if result.calculation_errors:
            self.logger.warning(f"Calculation errors: {result.calculation_errors}")

class FundamentalCalculatorRegistry:
    """Registry for fundamental calculators"""
    
    _calculators = {}
    
    @classmethod
    def register(cls, name: str, calculator_class: type = None):
        """
        Register a calculator class
        
        Can be used as:
        1. Direct registration: register('name', CalculatorClass)
        2. Decorator: @register('name')
        """
        def decorator(calculator_class: type):
            """Decorator function for registration"""
            cls._calculators[name] = calculator_class
            logger.info(f"Registered fundamental calculator: {name} -> {calculator_class.__name__}")
            return calculator_class
        
        if calculator_class is not None:
            # Direct registration
            return decorator(calculator_class)
        else:
            # Decorator usage
            return decorator
    
    @classmethod
    def get_calculator(cls, name: str) -> Optional[type]:
        """Get a calculator class by name"""
        return cls._calculators.get(name)
    
    @classmethod
    def get_all_calculators(cls) -> Dict[str, type]:
        """Get all registered calculators"""
        return cls._calculators.copy()
    
    @classmethod
    def list_calculators(cls) -> List[str]:
        """List all registered calculator names"""
        return list(cls._calculators.keys()) 