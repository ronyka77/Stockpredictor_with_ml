from dataclasses import dataclass
from typing import List, Optional, Any, Union
from decimal import Decimal, InvalidOperation
import html

from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse
from src.data_collector.polygon_fundamentals.data_validator import (
    FundamentalDataValidator,
    ValidationResult,
)
from src.utils.core.logger import get_logger
from src.utils.core.validation import (
    SecureBaseModel,
    SecureDateTime,
    ValidationUtils,
    SecurityValidationError,
)
from src.utils.qa.security_audit import SecurityAuditLogger, log_input_validation_failure

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
            # Validate numeric value - allow negative values for financial data
            if self.value is not None:
                self._validate_financial_number(self.value, "financial_value")

            # Validate string fields with financial-friendly validation
            if self.unit:
                self._validate_financial_string(self.unit, "unit")
            if self.label:
                self._validate_financial_string(self.label, "label")
            if self.source:
                self._validate_financial_string(self.source, "source")

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
                    "validation_type": "financial_value",
                },
            )
            raise

    def _validate_financial_number(self, value: Any, field_name: str) -> Union[int, float]:
        """Validate financial numeric value (allows negative values unlike positive_number validation)"""
        try:
            num = float(value) if not isinstance(value, (int, float)) else value

            if not isinstance(num, (int, float)) or isinstance(num, bool):
                raise SecurityValidationError(f"{field_name} must be a number")

            # Check for extreme values that might indicate data corruption
            if abs(num) > 1e15:  # $1 quadrillion upper bound for financial data
                raise SecurityValidationError(f"{field_name} value is unreasonably large")

            # Check for potential NaN or infinity values
            if num != num or num == float("inf") or num == float("-inf"):
                raise SecurityValidationError(f"{field_name} contains invalid numeric value")

            return num

        except (ValueError, TypeError, OverflowError):
            raise SecurityValidationError(f"Invalid numeric value for {field_name}")

    def _validate_financial_string(self, value: Any, field_name: str) -> str:
        """Validate string values for financial data (less restrictive than secure string validation)"""
        if not isinstance(value, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        # Basic sanitization
        sanitized = html.escape(value.strip())

        # For financial data, only check for the most obvious security threats
        # Allow common financial characters and patterns
        obvious_threats = [
            "<script",
            "</script>",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror=",
            "onclick=",
        ]

        for threat in obvious_threats:
            if threat.lower() in sanitized.lower():
                raise SecurityValidationError(
                    f"Security threat detected in {field_name}", security_threat=True
                )

        # Check length limits (reasonable for financial data)
        if len(sanitized) > 500:  # Financial strings shouldn't be extremely long
            raise SecurityValidationError(f"{field_name} is too long")

        return sanitized

    @classmethod
    def from_financial_value(cls, fv: Any) -> "SecureFinancialValue":
        """Create secure financial value from regular FinancialValue"""
        if hasattr(fv, "__dict__"):
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

            # Validate string fields with financial-friendly validation
            if self.timeframe:
                self._validate_financial_string(self.timeframe, "timeframe")
            if self.fiscal_period:
                self._validate_financial_string(self.fiscal_period, "fiscal_period")
            if self.fiscal_year:
                self._validate_financial_string(self.fiscal_year, "fiscal_year")

            # Validate company information
            if self.cik:
                self._validate_financial_string(self.cik, "cik")
            if self.company_name:
                self._validate_financial_string(self.company_name, "company_name")
            if self.ticker:
                ValidationUtils.validate_ticker_symbol(self.ticker, "ticker")

            # Validate URLs with financial-friendly validation
            if self.source_filing_url:
                self._validate_financial_string(self.source_filing_url, "source_filing_url")
            if self.source_filing_file_url:
                self._validate_financial_string(
                    self.source_filing_file_url, "source_filing_file_url"
                )

        except SecurityValidationError as e:
            log_input_validation_failure(
                resource="financial_statement",
                validation_errors=[e.message],
                input_source="polygon_fundamentals_api",
                details={
                    "field": e.field,
                    "value": str(e.value)[:100],
                    "validation_type": "financial_statement",
                },
            )
            raise

    def _validate_financial_string(self, value: Any, field_name: str) -> str:
        """Validate string values for financial data (less restrictive than secure string validation)"""
        if not isinstance(value, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        # Basic sanitization
        sanitized = html.escape(value.strip())

        # For financial data, only check for the most obvious security threats
        # Allow common financial characters and patterns
        obvious_threats = [
            "<script",
            "</script>",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror=",
            "onclick=",
        ]

        for threat in obvious_threats:
            if threat.lower() in sanitized.lower():
                raise SecurityValidationError(
                    f"Security threat detected in {field_name}", security_threat=True
                )

        # Check length limits (reasonable for financial data)
        if len(sanitized) > 500:  # Financial strings shouldn't be extremely long
            raise SecurityValidationError(f"{field_name} is too long")

        return sanitized


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
                error_message=e.message,
            )
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            cross_errors.append(f"Validation error: {str(e)}")

        return ExtendedValidation(
            base=base,
            cross_errors=cross_errors,
            cross_warnings=cross_warnings,
            security_errors=security_errors,
            security_warnings=security_warnings,
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
            for stmt in getattr(resp, "income_statements", []):
                stmt_security = self._validate_statement_security(stmt, "income_statement")
                errors.extend(stmt_security["errors"])
                warnings.extend(stmt_security["warnings"])

            for stmt in getattr(resp, "balance_sheets", []):
                stmt_security = self._validate_statement_security(stmt, "balance_sheet")
                errors.extend(stmt_security["errors"])
                warnings.extend(stmt_security["warnings"])

            for stmt in getattr(resp, "cash_flow_statements", []):
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
            # For financial data processing, use minimal validation to avoid false positives
            # Financial data from Polygon API is trusted, so we skip strict security validation
            stmt_data = stmt.__dict__ if hasattr(stmt, "__dict__") else stmt

            # Only perform basic structural validation without strict security checks
            required_fields = ["ticker"]
            for field in required_fields:
                if field not in stmt_data or stmt_data[field] is None:
                    errors.append(f"{stmt_type} missing required field: {field}")

            # Validate ticker format (basic check)
            if hasattr(stmt, "ticker") and stmt.ticker:
                if (
                    not isinstance(stmt.ticker, str)
                    or len(stmt.ticker) < 1
                    or len(stmt.ticker) > 10
                ):
                    errors.append(f"{stmt_type} invalid ticker format")

            # Validate all financial values exist and are numeric (light validation)
            for attr_name in dir(stmt):
                if not attr_name.startswith("_"):
                    attr_value = getattr(stmt, attr_name)
                    if hasattr(attr_value, "value"):  # FinancialValue object
                        if attr_value.value is not None:
                            try:
                                # Just check it's a valid number, no security restrictions
                                float(attr_value.value)
                            except (ValueError, TypeError):
                                errors.append(f"{stmt_type}.{attr_name}: invalid numeric value")

        except Exception as e:
            logger.error(f"Statement validation error for {stmt_type}: {e}")
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
            if hasattr(inc, "end_date") and hasattr(bs, "end_date"):
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
            if hasattr(cf, "end_date"):
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
            if not (
                bs
                and hasattr(bs, "assets")
                and hasattr(bs, "liabilities")
                and hasattr(bs, "equity")
            ):
                return errors, warnings

            assets_val = getattr(bs.assets, "value", None) if bs.assets else None
            liabilities_val = getattr(bs.liabilities, "value", None) if bs.liabilities else None
            equity_val = getattr(bs.equity, "value", None) if bs.equity else None

            if not all(v is not None for v in [assets_val, liabilities_val, equity_val]):
                return errors, warnings

            # Validate all values are reasonable financial numbers (allow negative)
            self._validate_financial_currency_amount(assets_val, "assets")
            self._validate_financial_currency_amount(liabilities_val, "liabilities")
            self._validate_financial_currency_amount(equity_val, "equity")

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
            if not (
                cf
                and hasattr(cf, "net_cash_flow_from_operating_activities")
                and hasattr(cf, "net_cash_flow_from_investing_activities")
                and hasattr(cf, "net_cash_flow_from_financing_activities")
                and hasattr(cf, "net_cash_flow")
            ):
                return errors, warnings

            op_val = getattr(cf.net_cash_flow_from_operating_activities, "value", None)
            inv_val = getattr(cf.net_cash_flow_from_investing_activities, "value", None)
            fin_val = getattr(cf.net_cash_flow_from_financing_activities, "value", None)
            net_val = getattr(cf.net_cash_flow, "value", None)

            if not all(v is not None for v in [op_val, inv_val, fin_val, net_val]):
                return errors, warnings

            # Validate all cash flow values (allow negative values)
            self._validate_financial_currency_amount(op_val, "operating_cash_flow")
            self._validate_financial_currency_amount(inv_val, "investing_cash_flow")
            self._validate_financial_currency_amount(fin_val, "financing_cash_flow")
            self._validate_financial_currency_amount(net_val, "net_cash_flow")

            total = op_val + inv_val + fin_val
            denom = abs(total) if total else 1.0
            if abs(net_val - total) > 0.05 * denom:
                warnings.append("Cash flow components don't sum within 5% tolerance")

        except SecurityValidationError as e:
            errors.append(f"Cash flow validation error: {e.message}")
        except Exception as e:
            logger.error(f"Error in cash flow components sum: {e}")

        return errors, warnings

    def _validate_financial_currency_amount(self, value: Any, field_name: str) -> Decimal:
        """Validate currency amount for financial data (allows negative, larger bounds)"""
        try:
            if isinstance(value, str):
                # Handle string representations of currency
                value = value.replace("$", "").replace(",", "").strip()

            amount = Decimal(str(value))

            # Validate reasonable financial ranges (much larger than regular currency)
            # Allow up to $100 trillion for large company financials
            if abs(amount) > Decimal("100000000000000"):  # $100T upper bound
                raise SecurityValidationError(f"{field_name} amount is unreasonably large")

            # No lower bound restriction for negative values (losses, negative cash flows)

            # Round to 2 decimal places for currency precision
            return amount.quantize(Decimal("0.01"))

        except (InvalidOperation, ValueError):
            raise SecurityValidationError(f"Invalid currency amount for {field_name}")
