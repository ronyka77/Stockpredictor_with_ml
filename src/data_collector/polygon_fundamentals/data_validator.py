"""
Data Validator for Fundamental Financial Data

This module provides validation and quality checking for fundamental
financial data retrieved from the Polygon API.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.data_collector.polygon_fundamentals.data_models import (
    FundamentalDataResponse,
    IncomeStatement,
    BalanceSheet,
    CashFlowStatement,
    FinancialValue
)
from src.data_collector.polygon_fundamentals.config import VALIDATION_RULES

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    errors: List[str]
    warnings: List[str]
    missing_fields: List[str]
    outliers: List[str]
    
    def __post_init__(self):
        """Calculate overall quality score"""
        if not hasattr(self, 'quality_score') or self.quality_score is None:
            self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """Calculate quality score based on validation results"""
        base_score = 100.0
        
        # Deduct for errors (major issues)
        base_score -= len(self.errors) * 20
        
        # Deduct for warnings (minor issues)
        base_score -= len(self.warnings) * 5
        
        # Deduct for missing fields
        base_score -= len(self.missing_fields) * 3
        
        # Deduct for outliers
        base_score -= len(self.outliers) * 2
        
        return max(0.0, min(100.0, base_score))

class FundamentalDataValidator:
    """Validator for fundamental financial data"""
    
    def __init__(self):
        self.validation_rules = VALIDATION_RULES
    
    def validate_response(self, response: FundamentalDataResponse) -> ValidationResult:
        """
        Validate a complete fundamental data response
        
        Args:
            response: FundamentalDataResponse to validate
        
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        missing_fields = []
        outliers = []
        
        # Check if we have any data at all
        if not any([response.income_statements, response.balance_sheets, response.cash_flow_statements]):
            errors.append("No financial statements found")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                errors=errors,
                warnings=warnings,
                missing_fields=missing_fields,
                outliers=outliers
            )
        
        # Validate each statement type
        for stmt in response.income_statements:
            result = self.validate_income_statement(stmt)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            missing_fields.extend(result.missing_fields)
            outliers.extend(result.outliers)
        
        for stmt in response.balance_sheets:
            result = self.validate_balance_sheet(stmt)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            missing_fields.extend(result.missing_fields)
            outliers.extend(result.outliers)
        
        for stmt in response.cash_flow_statements:
            result = self.validate_cash_flow_statement(stmt)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            missing_fields.extend(result.missing_fields)
            outliers.extend(result.outliers)
        
        # Cross-statement validation
        cross_validation = self._validate_cross_statements(response)
        errors.extend(cross_validation.errors)
        warnings.extend(cross_validation.warnings)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=0.0,  # Will be calculated in __post_init__
            errors=list(set(errors)),  # Remove duplicates
            warnings=list(set(warnings)),
            missing_fields=list(set(missing_fields)),
            outliers=list(set(outliers))
        )
    
    def validate_income_statement(self, stmt: IncomeStatement) -> ValidationResult:
        """Validate an income statement"""
        errors = []
        warnings = []
        missing_fields = []
        outliers = []
        
        # Check required fields
        required_fields = self.validation_rules['required_fields']['income_statement']
        for field in required_fields:
            value = getattr(stmt, field, None)
            if not value or (hasattr(value, 'value') and value.value is None):
                missing_fields.append(f"income_statement.{field}")
        
        # Check only essential fields required for fundamental calculations
        essential_fields = [
            'earnings_per_share_basic', 'weighted_average_shares_outstanding', 'operating_income_loss'
        ]
        for field in essential_fields:
            value = getattr(stmt, field, None)
            if not value or (hasattr(value, 'value') and value.value is None):
                missing_fields.append(f"income_statement.{field}")
        
        # Validate logical relationships
        revenues = self._get_value(stmt.revenues)
        # net_income = self._get_value(stmt.net_income_loss)
        gross_profit = self._get_value(stmt.gross_profit)
        cost_of_revenue = self._get_value(stmt.cost_of_revenue)
        
        # Revenue should be positive
        if revenues is not None and revenues < 0:
            warnings.append("Negative revenues in income statement")
        
        # Gross profit should equal revenues minus cost of revenue
        if all(v is not None for v in [revenues, cost_of_revenue, gross_profit]):
            expected_gross = revenues - cost_of_revenue
            if abs(gross_profit - expected_gross) > abs(expected_gross * 0.01):  # 1% tolerance
                warnings.append("Gross profit calculation inconsistency")
        
        # Check for extreme values
        if revenues is not None and abs(revenues) > 1e12:  # $1 trillion
            outliers.append("Extremely high revenues")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=0.0,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            outliers=outliers
        )
    
    def validate_balance_sheet(self, stmt: BalanceSheet) -> ValidationResult:
        """Validate a balance sheet"""
        errors = []
        warnings = []
        missing_fields = []
        outliers = []
        
        # Check required fields
        required_fields = self.validation_rules['required_fields']['balance_sheet']
        for field in required_fields:
            value = getattr(stmt, field, None)
            if not value or (hasattr(value, 'value') and value.value is None):
                missing_fields.append(f"balance_sheet.{field}")
        
        # Check only essential fields required for fundamental calculations
        essential_fields = [
            'current_assets', 'current_liabilities'
        ]
        for field in essential_fields:
            value = getattr(stmt, field, None)
            if not value or (hasattr(value, 'value') and value.value is None):
                missing_fields.append(f"balance_sheet.{field}")
        
        # Validate balance sheet equation: Assets = Liabilities + Equity
        assets = self._get_value(stmt.assets)
        liabilities = self._get_value(stmt.liabilities)
        equity = self._get_value(stmt.equity)
        
        if all(v is not None for v in [assets, liabilities, equity]):
            balance_diff = abs(assets - (liabilities + equity))
            tolerance = abs(assets * 0.01)  # 1% tolerance
            
            if balance_diff > tolerance:
                errors.append("Balance sheet equation does not balance")
        
        # Check for negative values where they shouldn't be
        if assets is not None and assets < 0:
            errors.append("Negative total assets")
        
        # Check current ratio reasonableness
        current_assets = self._get_value(stmt.current_assets)
        current_liabilities = self._get_value(stmt.current_liabilities)
        
        if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            min_ratio, max_ratio = self.validation_rules['logical_checks']['current_ratio_reasonable']
            
            if not (min_ratio <= current_ratio <= max_ratio):
                outliers.append(f"Unusual current ratio: {current_ratio:.2f}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=0.0,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            outliers=outliers
        )
    
    def validate_cash_flow_statement(self, stmt: CashFlowStatement) -> ValidationResult:
        """Validate a cash flow statement"""
        errors = []
        warnings = []
        missing_fields = []
        outliers = []
        
        # Check required fields
        required_fields = self.validation_rules['required_fields']['cash_flow']
        for field in required_fields:
            value = getattr(stmt, field, None)
            if not value or (hasattr(value, 'value') and value.value is None):
                missing_fields.append(f"cash_flow.{field}")
        
        
        # Validate cash flow equation
        operating_cf = self._get_value(stmt.net_cash_flow_from_operating_activities)
        investing_cf = self._get_value(stmt.net_cash_flow_from_investing_activities)
        financing_cf = self._get_value(stmt.net_cash_flow_from_financing_activities)
        net_cf = self._get_value(stmt.net_cash_flow)
        
        if all(v is not None for v in [operating_cf, investing_cf, financing_cf, net_cf]):
            calculated_net = operating_cf + investing_cf + financing_cf
            if abs(net_cf - calculated_net) > abs(calculated_net * 0.05):  # 5% tolerance
                warnings.append("Cash flow components don't sum to net cash flow")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=0.0,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            outliers=outliers
        )
    
    def _validate_cross_statements(self, response: FundamentalDataResponse) -> ValidationResult:
        """Validate consistency across different statement types"""
        errors = []
        warnings = []
        missing_fields = []
        outliers = []
        
        # Get latest statements for comparison
        latest_income = response.get_latest_income_statement()
        latest_balance = response.get_latest_balance_sheet()
        latest_cash_flow = response.get_latest_cash_flow()
        
        if not all([latest_income, latest_balance, latest_cash_flow]):
            warnings.append("Missing statements for cross-validation")
            return ValidationResult(
                is_valid=True,
                quality_score=0.0,
                errors=errors,
                warnings=warnings,
                missing_fields=missing_fields,
                outliers=outliers
            )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=0.0,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            outliers=outliers
        )
    
    def _get_value(self, financial_value: Optional[FinancialValue]) -> Optional[float]:
        """Extract numeric value from FinancialValue object"""
        if financial_value is None:
            return None
        if hasattr(financial_value, 'value'):
            return financial_value.value
        return financial_value