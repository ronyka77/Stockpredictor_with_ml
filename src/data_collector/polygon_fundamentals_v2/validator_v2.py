from dataclasses import dataclass
from typing import List, Optional, Any

from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse
from src.data_collector.polygon_fundamentals.data_validator import (
    FundamentalDataValidator,
    ValidationResult,
)
from src.utils.core.logger import get_logger
from src.utils.core.validation import (
    SecureBaseModel,
    SecureNumeric,
    SecureString,
    SecureDateTime,
    ValidationUtils,
    SecurityValidationError,
    validate_input_data,
)
from src.utils.qa.security_audit import (
    SecurityAuditLogger,
    log_input_validation_failure,
)

logger = get_logger(__name__)
security_logger = SecurityAuditLogger()


class SecureFinancialValue(SecureBaseModel):
    """Security-enhanced financial value model"""

    value: Optional[float] = None
    unit: Optional[str] = None
    label: Optional[str] = None
    order: Optional[int] = None
    source: Optional[str] = None

    def validate_security(self) -> None:
        """Perform security validation on financial value"""
        try:
            # Validate numeric value for injection attacks
            if self.value is not None:
                SecureNumeric.validate_positive_number(self.value, "financial_value")

            # Validate string fields for injection attacks
            if self.unit:
                SecureString.validate_secure_string(self.unit)
            if self.label:
                SecureString.validate_secure_string(self.label)
            if self.source:
                SecureString.validate_secure_string(self.source)

            # Validate order as reasonable integer
            if self.order is not None:
                if not isinstance(self.order, int) or self.order < 0 or self.order > 1000:
                    raise SecurityValidationError("Invalid order value", "order", self.order)

        except SecurityValidationError as e:
            log_input_validation_failure(
                resource="financial_value",
                validation_errors=[e.message],
                input_source="polygon_fundamentals_api",
                details={
                    "field": e.field,
                    "value": str(e.value)[:100],
                    "validation_type": "financial_value"
                }
            )
            raise

    @classmethod
    def from_financial_value(cls, fv: Any) -> 'SecureFinancialValue':
        """Create secure financial value from regular FinancialValue"""
        if hasattr(fv, '__dict__'):
            data = fv.__dict__
        elif isinstance(fv, dict):
            data = fv
        else:
            raise SecurityValidationError("Invalid financial value format")

        # Create instance first, then validate security
        instance = cls(**data)
        instance.validate_security()
        return instance


class SecureFinancialStatement(SecureBaseModel):
    """Security-enhanced financial statement model"""

    # Statement metadata
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filing_date: Optional[str] = None
    period_of_report_date: Optional[str] = None
    timeframe: Optional[str] = None
    fiscal_period: Optional[str] = None
    fiscal_year: Optional[str] = None

    # Company information
    cik: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None

    # Source information
    source_filing_url: Optional[str] = None
    source_filing_file_url: Optional[str] = None

    def validate_security(self) -> None:
        """Perform security validation on financial statement"""
        try:
            # Validate dates
            if self.start_date:
                SecureDateTime.validate_date(self.start_date, "start_date")
            if self.end_date:
                SecureDateTime.validate_date(self.end_date, "end_date")
            if self.filing_date:
                SecureDateTime.validate_date(self.filing_date, "filing_date")
            if self.period_of_report_date:
                SecureDateTime.validate_date(self.period_of_report_date, "period_of_report_date")

            # Validate string fields
            if self.timeframe:
                SecureString.validate_secure_string(self.timeframe)
            if self.fiscal_period:
                SecureString.validate_secure_string(self.fiscal_period)
            if self.fiscal_year:
                SecureString.validate_secure_string(self.fiscal_year)

            # Validate company information
            if self.cik:
                SecureString.validate_secure_string(self.cik)
            if self.company_name:
                SecureString.validate_secure_string(self.company_name)
            if self.ticker:
                ValidationUtils.validate_ticker_symbol(self.ticker, "ticker")

            # Validate URLs
            if self.source_filing_url:
                SecureString.validate_secure_string(self.source_filing_url)
            if self.source_filing_file_url:
                SecureString.validate_secure_string(self.source_filing_file_url)

        except SecurityValidationError as e:
            log_input_validation_failure(
                resource="financial_statement",
                validation_errors=[e.message],
                input_source="polygon_fundamentals_api",
                details={
                    "field": e.field,
                    "value": str(e.value)[:100],
                    "validation_type": "financial_statement"
                }
            )
            raise


@dataclass
class ExtendedValidation:
    """Extended validation result with cross-statement and security validation."""

    base: ValidationResult
    cross_errors: List[str]
    cross_warnings: List[str]
    security_errors: List[str]
    security_warnings: List[str]

    @property
    def is_valid(self) -> bool:
        """Check if the extended validation passes all criteria."""
        return self.base.is_valid and not self.cross_errors and not self.security_errors

    @property
    def quality_score(self) -> float:
        """Return the quality score from the base validation."""
        return self.base.quality_score


class FundamentalDataValidatorV2:
    """Wraps V1 validator and adds cross-statement integrity rules with security validation."""

    def __init__(self) -> None:
        self.base = FundamentalDataValidator()

    def validate(self, resp: FundamentalDataResponse) -> ExtendedValidation:
        """Validate fundamental data response with security checks"""
        base = self.base.validate_response(resp)
        cross_errors: List[str] = []
        cross_warnings: List[str] = []
        security_errors: List[str] = []
        security_warnings: List[str] = []

        try:
            # Perform security validation on the entire response
            security_validation = self._validate_security(resp)
            security_errors.extend(security_validation["errors"])
            security_warnings.extend(security_validation["warnings"])

            # Perform cross-statement validation with security checks
            cross_validation = self._validate_cross_statements_secure(resp)
            cross_errors.extend(cross_validation["errors"])
            cross_warnings.extend(cross_validation["warnings"])

        except SecurityValidationError as e:
            security_errors.append(f"Security validation failed: {e.message}")
            log_input_validation_failure(
                field=e.field,
                value=str(e.value)[:100],
                validation_type="fundamental_response",
                error_message=e.message
            )
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            cross_errors.append(f"Validation error: {str(e)}")

        return ExtendedValidation(
            base=base,
            cross_errors=cross_errors,
            cross_warnings=cross_warnings,
            security_errors=security_errors,
            security_warnings=security_warnings
        )

    def _validate_security(self, resp: FundamentalDataResponse) -> dict:
        """Perform comprehensive security validation on fundamental data"""
        errors = []
        warnings = []

        try:
            # Validate response structure first
            if not isinstance(resp, FundamentalDataResponse):
                errors.append("Invalid response type - not a FundamentalDataResponse")
                return {"errors": errors, "warnings": warnings}

            # Validate each statement type with security checks
            for stmt in getattr(resp, 'income_statements', []):
                stmt_security = self._validate_statement_security(stmt, "income_statement")
                errors.extend(stmt_security["errors"])
                warnings.extend(stmt_security["warnings"])

            for stmt in getattr(resp, 'balance_sheets', []):
                stmt_security = self._validate_statement_security(stmt, "balance_sheet")
                errors.extend(stmt_security["errors"])
                warnings.extend(stmt_security["warnings"])

            for stmt in getattr(resp, 'cash_flow_statements', []):
                stmt_security = self._validate_statement_security(stmt, "cash_flow_statement")
                errors.extend(stmt_security["errors"])
                warnings.extend(stmt_security["warnings"])

        except Exception as e:
            logger.error(f"Security validation error: {e}")
            errors.append(f"Security validation failed: {str(e)}")

        return {"errors": errors, "warnings": warnings}

    def _validate_statement_security(self, stmt: Any, stmt_type: str) -> dict:
        """Validate individual financial statement for security threats"""
        errors = []
        warnings = []

        try:
            # Create secure statement model and validate
            secure_stmt = SecureFinancialStatement(**stmt.__dict__ if hasattr(stmt, '__dict__') else stmt)
            secure_stmt.validate_security()

            # Validate all financial values in the statement
            for attr_name in dir(stmt):
                if not attr_name.startswith('_'):
                    attr_value = getattr(stmt, attr_name)
                    if hasattr(attr_value, 'value'):  # FinancialValue object
                        try:
                            SecureFinancialValue.from_financial_value(attr_value)
                        except SecurityValidationError as e:
                            errors.append(f"{stmt_type}.{attr_name}: {e.message}")

        except SecurityValidationError as e:
            errors.append(f"{stmt_type} security validation failed: {e.message}")
        except Exception as e:
            logger.error(f"Statement security validation error for {stmt_type}: {e}")
            errors.append(f"{stmt_type} validation error: {str(e)}")

        return {"errors": errors, "warnings": warnings}

    def _validate_cross_statements_secure(self, resp: FundamentalDataResponse) -> dict:
        """Perform cross-statement validation with security checks"""
        errors = []
        warnings = []

        try:
            inc = resp.get_latest_income_statement()
            bs = resp.get_latest_balance_sheet()
            cf = resp.get_latest_cash_flow()

            # Validate period alignments
            period_errors, period_warnings = self._validate_period_alignments(inc, bs, cf)
            errors.extend(period_errors)
            warnings.extend(period_warnings)

            # Validate balance sheet equation
            bs_errors, bs_warnings = self._validate_balance_sheet_equation(bs)
            errors.extend(bs_errors)
            warnings.extend(bs_warnings)

            # Validate cash flow components
            cf_errors, cf_warnings = self._validate_cash_flow_components(cf)
            errors.extend(cf_errors)
            warnings.extend(cf_warnings)

        except Exception as e:
            logger.error(f"Cross-statement validation error: {e}")
            errors.append(f"Cross-statement validation failed: {str(e)}")

        return {"errors": errors, "warnings": warnings}

    def _validate_period_alignments(self, inc: Any, bs: Any, cf: Any) -> tuple:
        """Validate period alignments between financial statements"""
        errors = []
        warnings = []

        # Validate income statement and balance sheet alignment
        bs_errors, bs_warnings = self._validate_income_balance_sheet_alignment(inc, bs)
        errors.extend(bs_errors)
        warnings.extend(bs_warnings)

        # Validate income statement and cash flow alignment
        cf_errors, cf_warnings = self._validate_income_cash_flow_alignment(inc, cf)
        errors.extend(cf_errors)
        warnings.extend(cf_warnings)

        return errors, warnings

    def _validate_income_balance_sheet_alignment(self, inc: Any, bs: Any) -> tuple:
        """Validate alignment between income statement and balance sheet periods"""
        errors = []
        warnings = []

        if not (inc and bs):
            return errors, warnings

        try:
            if hasattr(inc, 'end_date') and hasattr(bs, 'end_date'):
                SecureDateTime.validate_date(str(inc.end_date), "inc_end_date")
                SecureDateTime.validate_date(str(bs.end_date), "bs_end_date")

            if (inc.end_date != bs.end_date) or (inc.fiscal_period != bs.fiscal_period):
                warnings.append("Latest IS and BS period misalignment")
        except SecurityValidationError as e:
            errors.append(f"Date validation error in period alignment: {e.message}")

        return errors, warnings

    def _validate_income_cash_flow_alignment(self, inc: Any, cf: Any) -> tuple:
        """Validate alignment between income statement and cash flow periods"""
        errors = []
        warnings = []

        if not (inc and cf):
            return errors, warnings

        try:
            if hasattr(cf, 'end_date'):
                SecureDateTime.validate_date(str(cf.end_date), "cf_end_date")

            if (inc.end_date != cf.end_date) or (inc.fiscal_period != cf.fiscal_period):
                warnings.append("Latest IS and CF period misalignment")
        except SecurityValidationError as e:
            errors.append(f"Date validation error in cash flow alignment: {e.message}")

        return errors, warnings

    def _validate_balance_sheet_equation(self, bs: Any) -> tuple:
        """Validate balance sheet equation with security checks"""
        errors = []
        warnings = []

        try:
            if not (bs and hasattr(bs, 'assets') and hasattr(bs, 'liabilities') and hasattr(bs, 'equity')):
                return errors, warnings

            assets_val = getattr(bs.assets, 'value', None) if bs.assets else None
            liabilities_val = getattr(bs.liabilities, 'value', None) if bs.liabilities else None
            equity_val = getattr(bs.equity, 'value', None) if bs.equity else None

            if not all(v is not None for v in [assets_val, liabilities_val, equity_val]):
                return errors, warnings

            # Validate all values are secure numbers
            SecureNumeric.validate_currency_amount(assets_val, "assets")
            SecureNumeric.validate_currency_amount(liabilities_val, "liabilities")
            SecureNumeric.validate_currency_amount(equity_val, "equity")

            denom = abs(assets_val) if assets_val else 1.0
            if abs(assets_val - (liabilities_val + equity_val)) > 0.02 * denom:
                warnings.append("Balance sheet equation off by >2%")

        except SecurityValidationError as e:
            errors.append(f"Balance sheet equation validation error: {e.message}")
        except Exception as e:
            logger.error(f"Error in balance sheet equation: {e}")

        return errors, warnings

    def _validate_cash_flow_components(self, cf: Any) -> tuple:
        """Validate cash flow components sum with security checks"""
        errors = []
        warnings = []

        try:
            if not (cf and hasattr(cf, 'net_cash_flow_from_operating_activities') and
                    hasattr(cf, 'net_cash_flow_from_investing_activities') and
                    hasattr(cf, 'net_cash_flow_from_financing_activities') and
                    hasattr(cf, 'net_cash_flow')):
                return errors, warnings

            op_val = getattr(cf.net_cash_flow_from_operating_activities, 'value', None)
            inv_val = getattr(cf.net_cash_flow_from_investing_activities, 'value', None)
            fin_val = getattr(cf.net_cash_flow_from_financing_activities, 'value', None)
            net_val = getattr(cf.net_cash_flow, 'value', None)

            if not all(v is not None for v in [op_val, inv_val, fin_val, net_val]):
                return errors, warnings

            # Validate all cash flow values
            SecureNumeric.validate_currency_amount(op_val, "operating_cash_flow")
            SecureNumeric.validate_currency_amount(inv_val, "investing_cash_flow")
            SecureNumeric.validate_currency_amount(fin_val, "financing_cash_flow")
            SecureNumeric.validate_currency_amount(net_val, "net_cash_flow")

            total = op_val + inv_val + fin_val
            denom = abs(total) if total else 1.0
            if abs(net_val - total) > 0.05 * denom:
                warnings.append("Cash flow components don't sum within 5% tolerance")

        except SecurityValidationError as e:
            errors.append(f"Cash flow validation error: {e.message}")
        except Exception as e:
            logger.error(f"Error in cash flow components sum: {e}")

        return errors, warnings
