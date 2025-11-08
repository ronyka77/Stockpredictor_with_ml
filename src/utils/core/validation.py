"""
Comprehensive Input Validation Framework

This module provides a security-focused input validation system that extends Pydantic
with additional validation capabilities to prevent injection attacks and ensure data integrity.

Key Features:
- Input sanitization and validation against injection attacks
- Secure type validation with strict constraints
- Comprehensive error messages for security events
- Integration with existing logging infrastructure
- Performance-optimized validation for high-throughput data processing
"""

import re
import html
from datetime import datetime, date
from typing import Any, Dict, Optional, Union, TypeVar
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse, urljoin
from pydantic import BaseModel, ConfigDict

from .logger import get_logger

logger = get_logger(__name__, utility="validation")

# Type variables for generic validation
T = TypeVar('T')
ValidationResult = Union[T, None]

# Security constants
MAX_STRING_LENGTH = 10000  # Maximum string length to prevent DoS
MAX_LIST_LENGTH = 1000     # Maximum list length
MAX_DICT_KEYS = 100        # Maximum dictionary keys

# Regex patterns for validation
SQL_INJECTION_PATTERNS = [
    r';\s*--',  # SQL comment injection
    r';\s*/\*',  # SQL block comment injection
    r'union\s+select',  # UNION SELECT injection
    r';\s*drop\s+',  # DROP statement injection
    r';\s*delete\s+',  # DELETE statement injection
    r';\s*update\s+',  # UPDATE statement injection
    r';\s*insert\s+',  # INSERT statement injection
]

XSS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script tags
    r'javascript:',  # JavaScript URLs
    r'on\w+\s*=',  # Event handlers
    r'<iframe[^>]*>.*?</iframe>',  # Iframe injection
    r'<object[^>]*>.*?</object>',  # Object injection
]

PATH_TRAVERSAL_PATTERNS = [
    r'\.\./',  # Directory traversal
    r'\.\.\\',  # Windows directory traversal
    r'~',  # Home directory access
    r'\.\.',  # Double dot patterns
]

# Compiled regex patterns for performance
SQL_INJECTION_REGEX = re.compile('|'.join(SQL_INJECTION_PATTERNS), re.IGNORECASE)
XSS_REGEX = re.compile('|'.join(XSS_PATTERNS), re.IGNORECASE | re.DOTALL)
PATH_TRAVERSAL_REGEX = re.compile('|'.join(PATH_TRAVERSAL_PATTERNS))


class SecurityValidationError(Exception):
    """Custom validation error with security context"""

    def __init__(self, message: str, field: str = None, value: Any = None, security_threat: bool = False):
        self.message = message
        self.field = field
        self.value = value
        self.security_threat = security_threat
        super().__init__(message)

        # Log security threats
        if security_threat:
            logger.warning(f"Security threat detected in field '{field}': {message}",
                          extra={"field": field, "value": str(value)[:100], "threat_type": "validation"})


class SecureBaseModel(BaseModel):
    """
    Base model with security-focused validation features

    Extends Pydantic BaseModel with additional security validations
    and comprehensive error handling.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        strict=True
    )

    @classmethod
    def validate_security(cls, data: Dict[str, Any]) -> 'SecureBaseModel':
        """
        Validate data with security checks

        Args:
            data: Input data dictionary

        Returns:
            Validated model instance

        Raises:
            SecurityValidationError: If validation fails
        """
        try:
            # Pre-validate for security threats
            cls._check_security_threats(data)

            # Standard Pydantic validation
            instance = cls(**data)
            return instance

        except SecurityValidationError:
            # Re-raise security validation errors as-is
            raise
        except Exception as e:
            # Convert Pydantic errors to security-aware errors
            if hasattr(e, 'errors'):
                for error in e.errors():
                    field = '.'.join(str(loc) for loc in error['loc'])
                    message = error['msg']
                    raise SecurityValidationError(message, field=field, security_threat=False)
            else:
                logger.error(f"Unexpected validation error: {e}")
                raise SecurityValidationError(f"Validation failed: {str(e)}")

    @classmethod
    def _check_security_threats(cls, data: Dict[str, Any]) -> None:
        """
        Check for common security threats in input data

        Args:
            data: Input data to check

        Raises:
            SecurityValidationError: If security threat detected
        """
        def _check_value(value: Any, field_path: str = "") -> None:
            if isinstance(value, str):
                # Check for SQL injection
                if SQL_INJECTION_REGEX.search(value):
                    raise SecurityValidationError(
                        "Potential SQL injection detected",
                        field=field_path,
                        value=value,
                        security_threat=True
                    )

                # Check for XSS
                if XSS_REGEX.search(value):
                    raise SecurityValidationError(
                        "Potential XSS attack detected",
                        field=field_path,
                        value=value,
                        security_threat=True
                    )

                # Check for path traversal
                if PATH_TRAVERSAL_REGEX.search(value):
                    raise SecurityValidationError(
                        "Potential path traversal attack detected",
                        field=field_path,
                        value=value,
                        security_threat=True
                    )

                # Check string length limits
                if len(value) > MAX_STRING_LENGTH:
                    raise SecurityValidationError(
                        f"String length exceeds maximum allowed ({MAX_STRING_LENGTH})",
                        field=field_path,
                        value=f"length: {len(value)}"
                    )

            elif isinstance(value, (list, tuple)):
                # Check list length limits
                if len(value) > MAX_LIST_LENGTH:
                    raise SecurityValidationError(
                        f"List length exceeds maximum allowed ({MAX_LIST_LENGTH})",
                        field=field_path,
                        value=f"length: {len(value)}"
                    )

                # Recursively check list items
                for i, item in enumerate(value):
                    _check_value(item, f"{field_path}[{i}]")

            elif isinstance(value, dict):
                # Check dict key limits
                if len(value) > MAX_DICT_KEYS:
                    raise SecurityValidationError(
                        f"Dictionary has too many keys ({MAX_DICT_KEYS} max)",
                        field=field_path,
                        value=f"keys: {len(value)}"
                    )

                # Recursively check dict values
                for key, val in value.items():
                    if not isinstance(key, str) or len(key) > 100:
                        raise SecurityValidationError(
                            "Invalid dictionary key",
                            field=field_path,
                            value=f"key: {key}"
                        )
                    _check_value(val, f"{field_path}.{key}")

        # Check all fields
        for field, value in data.items():
            _check_value(value, field)


class SecureString(str):
    """
    Security-enhanced string type with built-in validation

    Provides string validation with injection protection and sanitization.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_secure_string

    @classmethod
    def validate_secure_string(cls, value: Any) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            raise SecurityValidationError(f"Expected string, got {type(value).__name__}")

        # Basic sanitization
        sanitized = html.escape(value.strip())

        # Security checks
        if SQL_INJECTION_REGEX.search(sanitized):
            raise SecurityValidationError("SQL injection pattern detected", security_threat=True)

        if XSS_REGEX.search(sanitized):
            raise SecurityValidationError("XSS pattern detected", security_threat=True)

        if PATH_TRAVERSAL_REGEX.search(sanitized):
            raise SecurityValidationError("Path traversal pattern detected", security_threat=True)

        return sanitized


class SecureNumeric:
    """Security-enhanced numeric validation utilities"""

    @staticmethod
    def validate_positive_number(value: Any, field_name: str = "value") -> Union[int, float]:
        """Validate positive numeric value"""
        try:
            num = float(value) if not isinstance(value, (int, float)) else value

            if not isinstance(num, (int, float)) or isinstance(num, bool):
                raise SecurityValidationError(f"{field_name} must be a number")

            if num <= 0:
                raise SecurityValidationError(f"{field_name} must be positive")

            # Check for overflow/underflow
            if abs(num) > 1e20:  # Reasonable upper bound
                raise SecurityValidationError(f"{field_name} value is too large")

            return num

        except (ValueError, TypeError):
            raise SecurityValidationError(f"Invalid numeric value for {field_name}")

    @staticmethod
    def validate_range(value: Any, min_val: Optional[float] = None,
                      max_val: Optional[float] = None, field_name: str = "value") -> Union[int, float]:
        """Validate numeric value within specified range"""
        num = SecureNumeric.validate_positive_number(value, field_name)

        if min_val is not None and num < min_val:
            raise SecurityValidationError(f"{field_name} must be >= {min_val}")

        if max_val is not None and num > max_val:
            raise SecurityValidationError(f"{field_name} must be <= {max_val}")

        return num

    @staticmethod
    def validate_currency_amount(value: Any, field_name: str = "amount") -> Decimal:
        """Validate currency amount with proper precision"""
        try:
            if isinstance(value, str):
                # Handle string representations of currency
                value = value.replace('$', '').replace(',', '').strip()

            amount = Decimal(str(value))

            # Validate reasonable currency ranges
            if abs(amount) > Decimal('999999999.99'):  # $999M upper bound
                raise SecurityValidationError(f"{field_name} amount is unreasonably large")

            if amount < Decimal('-999999999.99'):  # Large negative bound
                raise SecurityValidationError(f"{field_name} amount is unreasonably small")

            # Round to 2 decimal places for currency
            return amount.quantize(Decimal('0.01'))

        except (InvalidOperation, ValueError):
            raise SecurityValidationError(f"Invalid currency amount for {field_name}")


class SecureURL:
    """Security-enhanced URL validation"""

    ALLOWED_SCHEMES = {'http', 'https'}
    MAX_URL_LENGTH = 2048

    @staticmethod
    def validate_url(value: Any, field_name: str = "url",
                    allowed_schemes: Optional[set] = None) -> str:
        """
        Validate URL with security checks

        Args:
            value: URL string to validate
            field_name: Field name for error messages
            allowed_schemes: Set of allowed URL schemes (defaults to http/https)

        Returns:
            Validated URL string

        Raises:
            SecurityValidationError: If URL is invalid or insecure
        """
        if not isinstance(value, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        url = value.strip()

        if not url:
            raise SecurityValidationError(f"{field_name} cannot be empty")

        if len(url) > SecureURL.MAX_URL_LENGTH:
            raise SecurityValidationError(f"{field_name} is too long")

        try:
            parsed = urlparse(url)
        except Exception:
            raise SecurityValidationError(f"Invalid URL format for {field_name}")

        # Validate scheme
        schemes = allowed_schemes or SecureURL.ALLOWED_SCHEMES
        if parsed.scheme not in schemes:
            raise SecurityValidationError(f"URL scheme not allowed for {field_name}: {parsed.scheme}")

        # Validate netloc exists for network URLs
        if not parsed.netloc and parsed.scheme in {'http', 'https'}:
            raise SecurityValidationError(f"URL must have a valid domain for {field_name}")

        # Check for suspicious patterns
        if '..' in url or url.startswith('//'):
            raise SecurityValidationError(f"Suspicious URL pattern detected in {field_name}")

        return url

    @staticmethod
    def validate_api_endpoint(value: Any, base_url: Optional[str] = None,
                             field_name: str = "endpoint") -> str:
        """
        Validate API endpoint URL

        Args:
            value: Endpoint path or full URL
            base_url: Base URL to join with if value is a path
            field_name: Field name for error messages

        Returns:
            Validated endpoint URL
        """
        if not isinstance(value, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        endpoint = value.strip()

        if not endpoint:
            raise SecurityValidationError(f"{field_name} cannot be empty")

        # If base_url provided and endpoint is a path, join them
        if base_url and not endpoint.startswith(('http://', 'https://')):
            try:
                full_url = urljoin(base_url.rstrip('/') + '/', endpoint.lstrip('/'))
                return SecureURL.validate_url(full_url, field_name)
            except Exception:
                raise SecurityValidationError(f"Invalid endpoint path for {field_name}")

        # Otherwise validate as full URL
        return SecureURL.validate_url(endpoint, field_name)


class SecureDateTime:
    """Security-enhanced date/time validation"""

    @staticmethod
    def validate_datetime(value: Any, field_name: str = "datetime",
                         future_allowed: bool = True) -> datetime:
        """
        Validate datetime with security checks

        Args:
            value: DateTime value to validate
            field_name: Field name for error messages
            future_allowed: Whether future dates are allowed

        Returns:
            Validated datetime object
        """
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                # Try ISO format first
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise SecurityValidationError(f"Invalid datetime format for {field_name}")
        else:
            raise SecurityValidationError(f"Invalid datetime type for {field_name}")

        # Check for reasonable date ranges
        if dt.year < 1900 or dt.year > 2100:
            raise SecurityValidationError(f"Datetime year out of reasonable range for {field_name}")

        # Check future dates if not allowed
        if not future_allowed and dt > datetime.now():
            raise SecurityValidationError(f"Future datetime not allowed for {field_name}")

        return dt

    @staticmethod
    def validate_date(value: Any, field_name: str = "date",
                     future_allowed: bool = True) -> date:
        """
        Validate date with security checks

        Args:
            value: Date value to validate
            field_name: Field name for error messages
            future_allowed: Whether future dates are allowed

        Returns:
            Validated date object
        """
        if isinstance(value, date) and not isinstance(value, datetime):
            d = value
        elif isinstance(value, datetime):
            d = value.date()
        elif isinstance(value, str):
            try:
                d = datetime.fromisoformat(value.replace('Z', '+00:00')).date()
            except ValueError:
                raise SecurityValidationError(f"Invalid date format for {field_name}")
        else:
            raise SecurityValidationError(f"Invalid date type for {field_name}")

        # Check for reasonable date ranges
        if d.year < 1900 or d.year > 2100:
            raise SecurityValidationError(f"Date year out of reasonable range for {field_name}")

        # Check future dates if not allowed
        if not future_allowed and d > date.today():
            raise SecurityValidationError(f"Future date not allowed for {field_name}")

        return d


class ValidationUtils:
    """
    Utility functions for common validation patterns

    Provides reusable validation functions that can be used across the application.
    """

    @staticmethod
    def validate_email(email: str, field_name: str = "email") -> str:
        """Validate email address format"""
        if not isinstance(email, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        email = email.strip().lower()

        # Basic email regex (RFC 5322 compliant)
        email_pattern = re.compile(
            r'^[a-zA-Z0-9](?:[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+)*'
            r'@(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        )

        if not email_pattern.match(email):
            raise SecurityValidationError(f"Invalid email format for {field_name}")

        # Length checks
        if len(email) > 254:  # RFC 5321 limit
            raise SecurityValidationError(f"{field_name} is too long")

        return email

    @staticmethod
    def validate_ticker_symbol(ticker: str, field_name: str = "ticker") -> str:
        """Validate stock ticker symbol"""
        if not isinstance(ticker, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        ticker = ticker.strip().upper()

        # Ticker validation: alphanumeric, dots, hyphens, max 10 chars
        if not re.match(r'^[A-Z0-9.-]{1,10}$', ticker):
            raise SecurityValidationError(f"Invalid ticker symbol format for {field_name}")

        # Security check: prevent path traversal in ticker names
        if '..' in ticker or '/' in ticker or '\\' in ticker:
            raise SecurityValidationError(f"Invalid characters in {field_name}", security_threat=True)

        return ticker

    @staticmethod
    def validate_api_key(api_key: str, field_name: str = "api_key",
                        min_length: int = 20) -> str:
        """Validate API key format and security"""
        if not isinstance(api_key, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        api_key = api_key.strip()

        if len(api_key) < min_length:
            raise SecurityValidationError(f"{field_name} is too short (minimum {min_length} characters)")

        if len(api_key) > 128:  # Reasonable maximum
            raise SecurityValidationError(f"{field_name} is too long")

        # Check for suspicious patterns
        if any(char in api_key for char in ['<', '>', '&', '"', "'"]):
            raise SecurityValidationError(f"Invalid characters in {field_name}", security_threat=True)

        # Check for common weak patterns
        if api_key.lower() in ['password', 'admin', 'test', 'key', 'token']:
            raise SecurityValidationError(f"{field_name} appears to be a placeholder value", security_threat=True)

        return api_key

    @staticmethod
    def sanitize_filename(filename: str, field_name: str = "filename") -> str:
        """Sanitize filename to prevent path traversal and injection"""
        if not isinstance(filename, str):
            raise SecurityValidationError(f"{field_name} must be a string")

        # Remove path separators and dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '*', '?', '"', '|']
        sanitized = filename

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')

        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(' .')

        # Ensure non-empty result
        if not sanitized:
            raise SecurityValidationError(f"{field_name} becomes empty after sanitization")

        # Check length
        if len(sanitized) > 255:  # Reasonable filename length
            raise SecurityValidationError(f"{field_name} is too long after sanitization")

        return sanitized

    @staticmethod
    def validate_batch_size(batch_size: int, field_name: str = "batch_size",
                           min_size: int = 1, max_size: int = 10000) -> int:
        """Validate batch size parameters"""
        try:
            size = int(batch_size)
        except (ValueError, TypeError):
            raise SecurityValidationError(f"{field_name} must be an integer")

        if size < min_size:
            raise SecurityValidationError(f"{field_name} must be at least {min_size}")

        if size > max_size:
            raise SecurityValidationError(f"{field_name} cannot exceed {max_size}")

        return size


class ValidationMetrics:
    """
    Metrics collection for validation operations

    Tracks validation performance and security events for monitoring.
    """

    def __init__(self):
        self.validations_performed = 0
        self.security_threats_detected = 0
        self.validation_errors = 0
        self.average_validation_time = 0.0

    def record_validation(self, security_threat: bool = False, error: bool = False,
                         duration: float = 0.0) -> None:
        """Record a validation operation"""
        self.validations_performed += 1

        if security_threat:
            self.security_threats_detected += 1

        if error:
            self.validation_errors += 1

        # Update rolling average
        if duration > 0:
            self.average_validation_time = (
                (self.average_validation_time * (self.validations_performed - 1)) + duration
            ) / self.validations_performed

    def get_metrics(self) -> Dict[str, Any]:
        """Get current validation metrics"""
        return {
            "validations_performed": self.validations_performed,
            "security_threats_detected": self.security_threats_detected,
            "validation_errors": self.validation_errors,
            "threat_rate": (
                self.security_threats_detected / self.validations_performed
                if self.validations_performed > 0 else 0
            ),
            "error_rate": (
                self.validation_errors / self.validations_performed
                if self.validations_performed > 0 else 0
            ),
            "average_validation_time_ms": self.average_validation_time * 1000
        }


# Global metrics instance
validation_metrics = ValidationMetrics()


def validate_input_data(data: Any, schema: type = None, strict: bool = True) -> ValidationResult:
    """
    High-level input validation function

    Args:
        data: Data to validate
        schema: Pydantic model class for validation
        strict: Whether to use strict validation mode

    Returns:
        Validated data or None if validation fails
    """
    import time
    start_time = time.time()

    try:
        if schema and issubclass(schema, BaseModel):
            if hasattr(schema, 'validate_security'):
                result = schema.validate_security(data)
            else:
                result = schema(**data)
        else:
            # Basic type validation
            if strict:
                SecureBaseModel._check_security_threats({'data': data})
            result = data

        validation_metrics.record_validation(duration=time.time() - start_time)
        return result

    except SecurityValidationError as e:
        validation_metrics.record_validation(error=True, duration=time.time() - start_time)
        if e.security_threat:
            logger.warning(f"Security threat blocked: {e.message}")
        raise

    except Exception as e:
        validation_metrics.record_validation(error=True, duration=time.time() - start_time)
        logger.error(f"Validation failed: {e}")
        raise SecurityValidationError(f"Validation failed: {str(e)}")
