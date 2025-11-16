# Backward compatibility module for validation
# Re-exports from core.validation for backward compatibility

from .core.validation import (
    validate_input_data,
    SecurityValidationError,
    ValidationUtils,
    ValidationMetrics,
    SecureBaseModel,
    SecureString,
    SecureNumeric,
    SecureURL,
    SecureDateTime,
    SQL_INJECTION_REGEX,
    XSS_REGEX,
    PATH_TRAVERSAL_REGEX,
    MAX_STRING_LENGTH,
    MAX_LIST_LENGTH,
    MAX_DICT_KEYS,
)

__all__ = [
    "validate_input_data",
    "SecurityValidationError",
    "ValidationUtils",
    "ValidationMetrics",
    "SecureBaseModel",
    "SecureString",
    "SecureNumeric",
    "SecureURL",
    "SecureDateTime",
    "SQL_INJECTION_REGEX",
    "XSS_REGEX",
    "PATH_TRAVERSAL_REGEX",
    "MAX_STRING_LENGTH",
    "MAX_LIST_LENGTH",
    "MAX_DICT_KEYS",
]
