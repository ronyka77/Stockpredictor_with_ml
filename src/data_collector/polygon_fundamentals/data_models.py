"""
Pydantic Data Models for Polygon Fundamentals API

This module defines data models for parsing and validating responses
from the Polygon fundamentals API endpoints.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FinancialValue(BaseModel):
    """Model for financial values with units and labels"""
    value: Optional[float] = None
    unit: Optional[str] = None
    label: Optional[str] = None
    order: Optional[int] = None
    source: Optional[str] = None  # Add this line to track the data source

class FinancialStatement(BaseModel):
    """Base model for financial statement data"""
    
    # Statement metadata
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    filing_date: Optional[date] = None
    period_of_report_date: Optional[date] = None
    timeframe: Optional[str] = None  # quarterly, annual
    fiscal_period: Optional[str] = None  # Q1, Q2, Q3, Q4, FY
    fiscal_year: Optional[str] = None
    
    # Company information
    cik: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    
    # Source information
    source_filing_url: Optional[str] = None
    source_filing_file_url: Optional[str] = None
    
    model_config = ConfigDict()

    @field_validator('start_date', 'end_date', 'filing_date', 'period_of_report_date', mode='before')
    @classmethod
    def parse_date(cls, v):
        """Parse date strings into date objects"""
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                return None
        return v

class IncomeStatement(FinancialStatement):
    """Income statement specific data model"""
    
    # Revenue
    revenues: Optional[FinancialValue] = None
    cost_of_revenue: Optional[FinancialValue] = None
    gross_profit: Optional[FinancialValue] = None
    
    # Operating expenses
    operating_expenses: Optional[FinancialValue] = None
    selling_general_and_administrative_expenses: Optional[FinancialValue] = None
    research_and_development: Optional[FinancialValue] = None
    
    # Operating income
    operating_income_loss: Optional[FinancialValue] = None
    
    # Non-operating items
    interest_expense: Optional[FinancialValue] = None
    interest_income: Optional[FinancialValue] = None
    other_income_expense: Optional[FinancialValue] = None
    
    # Pre-tax and tax
    income_loss_before_income_tax_expense_benefit: Optional[FinancialValue] = None
    income_tax_expense_benefit: Optional[FinancialValue] = None
    
    # Net income
    net_income_loss: Optional[FinancialValue] = None
    net_income_loss_attributable_to_parent: Optional[FinancialValue] = None
    
    # Per share data
    earnings_per_share_basic: Optional[FinancialValue] = None
    earnings_per_share_diluted: Optional[FinancialValue] = None
    weighted_average_shares_outstanding: Optional[FinancialValue] = None
    weighted_average_shares_outstanding_diluted: Optional[FinancialValue] = None
    
    # Additional items
    comprehensive_income_loss: Optional[FinancialValue] = None
    comprehensive_income_loss_attributable_to_parent: Optional[FinancialValue] = None

class BalanceSheet(FinancialStatement):
    """Balance sheet specific data model"""
    
    # Assets
    assets: Optional[FinancialValue] = None
    current_assets: Optional[FinancialValue] = None
    noncurrent_assets: Optional[FinancialValue] = None
    
    # Current assets detail
    cash_and_cash_equivalents_at_carrying_value: Optional[FinancialValue] = None
    short_term_investments: Optional[FinancialValue] = None
    accounts_receivable_net_current: Optional[FinancialValue] = None
    inventory_net: Optional[FinancialValue] = None
    prepaid_expenses_current: Optional[FinancialValue] = None
    other_assets_current: Optional[FinancialValue] = None
    
    # Non-current assets
    property_plant_equipment_net: Optional[FinancialValue] = None
    goodwill: Optional[FinancialValue] = None
    intangible_assets_net_excluding_goodwill: Optional[FinancialValue] = None
    long_term_investments: Optional[FinancialValue] = None
    other_assets_noncurrent: Optional[FinancialValue] = None
    
    # Liabilities
    liabilities: Optional[FinancialValue] = None
    current_liabilities: Optional[FinancialValue] = None
    noncurrent_liabilities: Optional[FinancialValue] = None
    
    # Current liabilities detail
    accounts_payable_current: Optional[FinancialValue] = None
    accrued_liabilities_current: Optional[FinancialValue] = None
    short_term_debt: Optional[FinancialValue] = None
    other_liabilities_current: Optional[FinancialValue] = None
    
    # Non-current liabilities
    long_term_debt_noncurrent: Optional[FinancialValue] = None
    other_liabilities_noncurrent: Optional[FinancialValue] = None
    
    # Equity
    equity: Optional[FinancialValue] = None
    equity_attributable_to_parent: Optional[FinancialValue] = None
    stockholders_equity: Optional[FinancialValue] = None
    
    # Equity detail
    common_stock_shares_outstanding: Optional[FinancialValue] = None
    common_stock_par_or_stated_value_per_share: Optional[FinancialValue] = None
    additional_paid_in_capital: Optional[FinancialValue] = None
    retained_earnings_accumulated_deficit: Optional[FinancialValue] = None
    accumulated_other_comprehensive_income_loss: Optional[FinancialValue] = None
    treasury_stock_value: Optional[FinancialValue] = None

class CashFlowStatement(FinancialStatement):
    """Cash flow statement specific data model"""
    
    # Operating activities
    net_cash_flow_from_operating_activities: Optional[FinancialValue] = None
    net_income_loss: Optional[FinancialValue] = None
    depreciation_depletion_and_amortization: Optional[FinancialValue] = None
    stock_based_compensation: Optional[FinancialValue] = None
    
    # Working capital changes
    increase_decrease_in_accounts_receivable: Optional[FinancialValue] = None
    increase_decrease_in_inventory: Optional[FinancialValue] = None
    increase_decrease_in_accounts_payable: Optional[FinancialValue] = None
    increase_decrease_in_other_operating_assets_liabilities: Optional[FinancialValue] = None
    
    # Investing activities
    net_cash_flow_from_investing_activities: Optional[FinancialValue] = None
    payments_to_acquire_property_plant_and_equipment: Optional[FinancialValue] = None
    payments_to_acquire_businesses_net_of_cash_acquired: Optional[FinancialValue] = None
    payments_to_acquire_investments: Optional[FinancialValue] = None
    proceeds_from_sale_of_investments: Optional[FinancialValue] = None
    
    # Financing activities
    net_cash_flow_from_financing_activities: Optional[FinancialValue] = None
    proceeds_from_issuance_of_debt: Optional[FinancialValue] = None
    repayments_of_debt: Optional[FinancialValue] = None
    proceeds_from_issuance_of_common_stock: Optional[FinancialValue] = None
    payments_for_repurchase_of_common_stock: Optional[FinancialValue] = None
    payments_of_dividends: Optional[FinancialValue] = None
    
    # Net change in cash
    cash_and_cash_equivalents_at_beginning_of_period: Optional[FinancialValue] = None
    cash_and_cash_equivalents_at_end_of_period: Optional[FinancialValue] = None
    net_cash_flow: Optional[FinancialValue] = None
    
    # Free cash flow (calculated)
    free_cash_flow: Optional[FinancialValue] = None

class CompanyDetails(BaseModel):
    """Company details and classification information"""
    
    # Basic information
    ticker: str
    name: Optional[str] = None
    market: Optional[str] = None
    locale: Optional[str] = None
    primary_exchange: Optional[str] = None
    type: Optional[str] = None
    active: Optional[bool] = None
    currency_name: Optional[str] = None
    cik: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    
    # Market data
    market_cap: Optional[float] = None
    phone_number: Optional[str] = None
    address: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    homepage_url: Optional[str] = None
    total_employees: Optional[int] = None
    list_date: Optional[date] = None
    
    # GICS classification
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    
    # Branding
    logo_url: Optional[str] = None
    icon_url: Optional[str] = None
    
    @field_validator('list_date', mode='before')
    @classmethod
    def parse_list_date(cls, v):
        """Parse list date string into date object"""
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                return None
        return v

class FundamentalDataResponse(BaseModel):
    """Complete response model for fundamental data"""
    
    # Response metadata
    status: str
    request_id: Optional[str] = None
    count: Optional[int] = None
    next_url: Optional[str] = None
    
    # Financial data
    results: Optional[List[Dict[str, Any]]] = None
    
    # Company information
    company_details: Optional[CompanyDetails] = None
    
    # Processed statements
    income_statements: List[IncomeStatement] = Field(default_factory=list)
    balance_sheets: List[BalanceSheet] = Field(default_factory=list)
    cash_flow_statements: List[CashFlowStatement] = Field(default_factory=list)
    
    # Data quality metrics
    data_quality_score: Optional[float] = None
    missing_data_count: Optional[int] = None
    total_fields_count: Optional[int] = None
    
    def get_latest_income_statement(self) -> Optional[IncomeStatement]:
        """Get the most recent income statement"""
        if not self.income_statements:
            return None
        return max(self.income_statements, key=lambda x: x.filing_date or date.min)
    
    def get_latest_balance_sheet(self) -> Optional[BalanceSheet]:
        """Get the most recent balance sheet"""
        if not self.balance_sheets:
            return None
        return max(self.balance_sheets, key=lambda x: x.filing_date or date.min)
    
    def get_latest_cash_flow(self) -> Optional[CashFlowStatement]:
        """Get the most recent cash flow statement"""
        if not self.cash_flow_statements:
            return None
        return max(self.cash_flow_statements, key=lambda x: x.filing_date or date.min)
    
    def get_statements_by_period(self, fiscal_period: str) -> Dict[str, Any]:
        """Get all statements for a specific fiscal period"""
        result = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow': None
        }
        
        for stmt in self.income_statements:
            if stmt.fiscal_period == fiscal_period:
                result['income_statement'] = stmt
                break
        
        for stmt in self.balance_sheets:
            if stmt.fiscal_period == fiscal_period:
                result['balance_sheet'] = stmt
                break
        
        for stmt in self.cash_flow_statements:
            if stmt.fiscal_period == fiscal_period:
                result['cash_flow'] = stmt
                break
        
        return result
    
    def calculate_data_quality(self) -> float:
        """Calculate overall data quality score based on essential fields only"""
        if not any([self.income_statements, self.balance_sheets, self.cash_flow_statements]):
            return 0.0
        
        # Define essential fields for fundamental calculations
        essential_fields = {
            'income_statements': [
                'revenues', 'net_income_loss', 'earnings_per_share_basic', 
                'weighted_average_shares_outstanding', 'operating_income_loss'
            ],
            'balance_sheets': [
                'assets', 'equity', 'current_assets', 'current_liabilities', 'liabilities'
            ],
            'cash_flow_statements': [
                'net_cash_flow_from_operating_activities'
            ]
        }
        
        total_essential_fields = 0
        missing_essential_fields = 0
        
        # Check only essential fields
        for stmt_type, field_list in essential_fields.items():
            stmt_list = getattr(self, stmt_type, [])
            for stmt in stmt_list:
                stmt_dict = stmt.dict()
                for field_name in field_list:
                    total_essential_fields += 1
                    value = stmt_dict.get(field_name)
                    if value is None or (isinstance(value, dict) and value.get('value') is None):
                        missing_essential_fields += 1
        
        if total_essential_fields == 0:
            return 0.0
        
        # Calculate quality based on essential fields only
        completeness = (total_essential_fields - missing_essential_fields) / total_essential_fields
        self.data_quality_score = round(completeness, 4)
        self.missing_data_count = missing_essential_fields
        self.total_fields_count = total_essential_fields
        
        return self.data_quality_score

# Utility functions for data extraction
def extract_financial_value(data: Dict[str, Any], field_name: str) -> Optional[float]:
    """Extract numeric value from financial data field"""
    if field_name not in data:
        return None
    
    field_data = data[field_name]
    if isinstance(field_data, dict):
        return field_data.get('value')
    elif isinstance(field_data, (int, float)):
        return float(field_data)
    
    return None

def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safely divide two numbers, handling None and zero values"""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator

def calculate_growth_rate(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Calculate growth rate between two values"""
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous)

def calculate_cagr(ending_value: Optional[float], beginning_value: Optional[float], periods: int) -> Optional[float]:
    """Calculate Compound Annual Growth Rate"""
    if ending_value is None or beginning_value is None or beginning_value <= 0 or periods <= 0:
        return None
    
    try:
        return (ending_value / beginning_value) ** (1 / periods) - 1
    except (ValueError, ZeroDivisionError):
        return None 