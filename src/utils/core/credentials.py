"""
Secure Credential Validation and Storage System

This module provides secure credential management with validation, storage,
and rotation capabilities. Prevents credential exposure in logs and errors.

Key Features:
- Secure API key validation and storage
- Credential masking and sanitization
- Integration with environment variables
- Secure credential rotation support
- No sensitive data in logs or error messages
"""

import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .logger import get_logger

logger = get_logger(__name__, utility="credentials")


class CredentialValidationError(Exception):
    """
    Custom exception for credential validation failures.
    Never exposes sensitive credential data in error messages.
    """

    def __init__(self, message: str, credential_type: str = "credential",
                 validation_error: bool = True):
        self.message = message
        self.credential_type = credential_type
        self.validation_error = validation_error
        super().__init__(message)

        # Log validation failures without sensitive data
        if validation_error:
            logger.warning(f"Credential validation failed for {credential_type}: {message}",
                          extra={"credential_type": credential_type, "validation_failed": True})


@dataclass
class CredentialMetadata:
    """Metadata for credential tracking and rotation"""

    credential_type: str
    created_at: datetime
    last_validated: datetime
    rotation_required: bool = False
    rotation_deadline: Optional[datetime] = None
    validation_count: int = 0
    last_validation_result: bool = True

    def mark_validated(self, success: bool = True) -> None:
        """Update validation metadata"""
        self.last_validated = datetime.now()
        self.validation_count += 1
        self.last_validation_result = success

    def requires_rotation(self) -> bool:
        """Check if credential rotation is required"""
        if self.rotation_deadline and datetime.now() > self.rotation_deadline:
            return True
        return self.rotation_required


class CredentialValidator(ABC):
    """Abstract base class for credential validators"""

    @abstractmethod
    def validate(self, credential: str) -> bool:
        """Validate a credential"""
        # Abstract method - implementation provided by subclasses
        pass

    @abstractmethod
    def get_validation_errors(self, credential: str) -> List[str]:
        """Get validation error messages (without exposing credential data)"""
        # Abstract method - implementation provided by subclasses
        pass

    @property
    @abstractmethod
    def credential_type(self) -> str:
        """Return the type of credential this validator handles"""
        pass


class APIKeyValidator(CredentialValidator):
    """Validator for API keys with security-focused validation"""

    # Common API key patterns (length and character requirements)
    MIN_LENGTH = 20
    MAX_LENGTH = 128

    # Characters that should not appear in API keys
    DANGEROUS_CHARS = ['<', '>', '&', '"', "'", '\n', '\r', '\t']

    # Common weak/placeholder values
    WEAK_VALUES = [
        'password', 'admin', 'test', 'key', 'token', 'api_key',
        'apikey', 'secret', 'credential', 'auth', 'bearer'
    ]

    def __init__(self, provider_name: str = "generic"):
        self.provider_name = provider_name

    @property
    def credential_type(self) -> str:
        return f"api_key_{self.provider_name}"

    def validate(self, credential: str) -> bool:
        """Validate API key with comprehensive security checks"""
        if not isinstance(credential, str):
            return False

        credential = credential.strip()

        # Length validation
        if len(credential) < self.MIN_LENGTH:
            return False

        if len(credential) > self.MAX_LENGTH:
            return False

        # Dangerous character check
        if any(char in credential for char in self.DANGEROUS_CHARS):
            return False

        # Weak value detection
        if credential.lower() in self.WEAK_VALUES:
            return False

        # Check for sequential patterns (potential security risk)
        if self._has_sequential_pattern(credential):
            return False

        # Check for repeated characters (potential security risk)
        if self._has_repeated_pattern(credential):
            return False

        return True

    def get_validation_errors(self, credential: str) -> List[str]:
        """Get validation error messages without exposing credential data"""
        errors = []

        if not isinstance(credential, str):
            errors.append("Credential must be a string")
            return errors

        credential = credential.strip()

        if len(credential) < self.MIN_LENGTH:
            errors.append(f"Credential too short (minimum {self.MIN_LENGTH} characters)")

        if len(credential) > self.MAX_LENGTH:
            errors.append(f"Credential too long (maximum {self.MAX_LENGTH} characters)")

        if any(char in credential for char in self.DANGEROUS_CHARS):
            errors.append("Credential contains invalid characters")

        if credential.lower() in self.WEAK_VALUES:
            errors.append("Credential appears to be a placeholder or weak value")

        if self._has_sequential_pattern(credential):
            errors.append("Credential has sequential pattern (security risk)")

        if self._has_repeated_pattern(credential):
            errors.append("Credential has repeated pattern (security risk)")

        return errors

    def _has_sequential_pattern(self, credential: str) -> bool:
        """Check for sequential character patterns"""
        # Check for sequential numbers
        for i in range(len(credential) - 2):
            try:
                if (int(credential[i+1]) == int(credential[i]) + 1 and
                    int(credential[i+2]) == int(credential[i]) + 2):
                    return True
            except ValueError:
                continue

        # Check for sequential letters
        for i in range(len(credential) - 2):
            if (credential[i+1] == chr(ord(credential[i]) + 1) and
                credential[i+2] == chr(ord(credential[i]) + 2)):
                return True

        return False

    def _has_repeated_pattern(self, credential: str) -> bool:
        """Check for repeated character patterns"""
        # More than 50% of characters are the same
        if len(credential) > 10:
            most_common = max(set(credential), key=credential.count)
            if credential.count(most_common) > len(credential) * 0.5:
                return True
        return False


class DatabasePasswordValidator(CredentialValidator):
    """Validator for database passwords"""

    MIN_LENGTH = 12
    MAX_LENGTH = 128

    @property
    def credential_type(self) -> str:
        return "database_password"

    def validate(self, credential: str) -> bool:
        """Validate database password"""
        if not isinstance(credential, str):
            return False

        credential = credential.strip()

        # Length validation
        if len(credential) < self.MIN_LENGTH:
            return False

        if len(credential) > self.MAX_LENGTH:
            return False

        # Must contain at least one uppercase, lowercase, digit
        has_upper = bool(re.search(r'[A-Z]', credential))
        has_lower = bool(re.search(r'[a-z]', credential))
        has_digit = bool(re.search(r'\d', credential))

        if not (has_upper and has_lower and has_digit):
            return False

        # Check for common weak passwords
        if credential.lower() in ['password', 'admin', 'root', 'postgres', 'database']:
            return False

        return True

    def get_validation_errors(self, credential: str) -> List[str]:
        """Get validation error messages"""
        errors = []

        if not isinstance(credential, str):
            errors.append("Password must be a string")
            return errors

        credential = credential.strip()

        if len(credential) < self.MIN_LENGTH:
            errors.append(f"Password too short (minimum {self.MIN_LENGTH} characters)")

        if len(credential) > self.MAX_LENGTH:
            errors.append(f"Password too long (maximum {self.MAX_LENGTH} characters)")

        has_upper = bool(re.search(r'[A-Z]', credential))
        has_lower = bool(re.search(r'[a-z]', credential))
        has_digit = bool(re.search(r'\d', credential))

        if not has_upper:
            errors.append("Password must contain at least one uppercase letter")
        if not has_lower:
            errors.append("Password must contain at least one lowercase letter")
        if not has_digit:
            errors.append("Password must contain at least one digit")

        if credential.lower() in ['password', 'admin', 'root', 'postgres', 'database']:
            errors.append("Password is too common or weak")

        return errors


@dataclass
class SecureCredential:
    """Secure credential wrapper that prevents accidental exposure"""

    _value: str = field(repr=False)
    _masked: str = field(init=False, repr=True)
    metadata: CredentialMetadata
    validator: CredentialValidator

    def __post_init__(self):
        """Initialize masked representation"""
        self._masked = self._mask_credential(self._value)

    @staticmethod
    def _mask_credential(value: str) -> str:
        """Create masked representation for safe display"""
        if not value:
            return ""

        length = len(value)
        if length <= 8:
            return "*" * length
        else:
            # Show first 4 and last 4 characters, mask the middle
            return f"{value[:4]}{'*' * (length - 8)}{value[-4:]}"

    @property
    def value(self) -> str:
        """Get the actual credential value (use with caution)"""
        logger.debug(f"Credential accessed: {self.metadata.credential_type}",
                    extra={"credential_type": self.metadata.credential_type, "access_time": datetime.now()})
        return self._value

    @property
    def masked(self) -> str:
        """Get masked representation for safe logging/display"""
        return self._masked

    def validate(self) -> bool:
        """Validate the credential and update metadata"""
        is_valid = self.validator.validate(self._value)
        self.metadata.mark_validated(is_valid)

        if not is_valid:
            errors = self.validator.get_validation_errors(self._value)
            logger.warning(f"Credential validation failed: {self.metadata.credential_type}",
                          extra={"credential_type": self.metadata.credential_type,
                                "errors": errors,
                                "validation_count": self.metadata.validation_count})

        return is_valid

    def requires_rotation(self) -> bool:
        """Check if credential needs rotation"""
        return self.metadata.requires_rotation()

    def rotate(self, new_value: str) -> bool:
        """Rotate credential to new value"""
        if not self.validator.validate(new_value):
            logger.error(f"Cannot rotate credential - new value invalid: {self.metadata.credential_type}")
            return False

        # Create hash of old value for audit trail (without storing the actual value)
        old_hash = hashlib.sha256(self._value.encode()).hexdigest()[:16]

        self._value = new_value
        self._masked = self._mask_credential(new_value)
        self.metadata.created_at = datetime.now()
        self.metadata.rotation_required = False
        self.metadata.rotation_deadline = None

        logger.info(f"Credential rotated successfully: {self.metadata.credential_type}",
                   extra={"credential_type": self.metadata.credential_type,
                         "old_hash_prefix": old_hash,
                         "rotation_time": datetime.now()})

        return True


class CredentialManager:
    """Central credential management system"""

    def __init__(self):
        self._credentials: Dict[str, SecureCredential] = {}
        self._validators = {
            'api_key': APIKeyValidator(),
            'polygon_api_key': APIKeyValidator('polygon'),
            'database_password': DatabasePasswordValidator(),
        }

    def add_credential(self, name: str, value: str, credential_type: str,
                      rotation_days: Optional[int] = None) -> SecureCredential:
        """
        Add a credential to the manager

        Args:
            name: Credential identifier
            value: Credential value
            credential_type: Type of credential (api_key, database_password, etc.)
            rotation_days: Days until rotation is required

        Returns:
            SecureCredential instance

        Raises:
            CredentialValidationError: If credential is invalid
        """
        if credential_type not in self._validators:
            raise CredentialValidationError(f"Unknown credential type: {credential_type}",
                                          credential_type=credential_type)

        validator = self._validators[credential_type]

        # Validate credential
        if not validator.validate(value):
            errors = validator.get_validation_errors(value)
            error_msg = f"Credential validation failed: {'; '.join(errors)}"
            raise CredentialValidationError(error_msg, credential_type=credential_type)

        # Create metadata
        metadata = CredentialMetadata(
            credential_type=credential_type,
            created_at=datetime.now(),
            last_validated=datetime.now(),
            validation_count=1,
            last_validation_result=True
        )

        if rotation_days:
            metadata.rotation_deadline = datetime.now() + timedelta(days=rotation_days)

        # Create secure credential
        credential = SecureCredential(
            _value=value,
            metadata=metadata,
            validator=validator
        )

        self._credentials[name] = credential

        logger.info(f"Credential added successfully: {name} ({credential_type})",
                   extra={"credential_name": name, "credential_type": credential_type})

        return credential

    def get_credential(self, name: str) -> Optional[SecureCredential]:
        """Get a credential by name"""
        credential = self._credentials.get(name)
        if credential:
            logger.debug(f"Credential retrieved: {name}",
                        extra={"credential_name": name, "credential_type": credential.metadata.credential_type})
        else:
            logger.warning(f"Credential not found: {name}",
                          extra={"credential_name": name})

        return credential

    def validate_all_credentials(self) -> Dict[str, bool]:
        """Validate all managed credentials"""
        results = {}
        for name, credential in self._credentials.items():
            results[name] = credential.validate()

        invalid_count = sum(1 for valid in results.values() if not valid)
        if invalid_count > 0:
            logger.warning(f"Credential validation completed: {invalid_count} invalid out of {len(results)} total",
                          extra={"total_credentials": len(results), "invalid_count": invalid_count})

        return results

    def get_rotation_candidates(self) -> List[str]:
        """Get list of credential names that require rotation"""
        candidates = [name for name, cred in self._credentials.items() if cred.requires_rotation()]
        if candidates:
            logger.info(f"Credentials requiring rotation: {candidates}",
                       extra={"rotation_candidates": candidates})
        return candidates

    def rotate_credential(self, name: str, new_value: str) -> bool:
        """Rotate a credential to a new value"""
        credential = self.get_credential(name)
        if not credential:
            logger.error(f"Cannot rotate non-existent credential: {name}")
            return False

        return credential.rotate(new_value)


class EnvironmentCredentialLoader:
    """Load credentials from environment variables securely"""

    # Mapping of environment variables to credential types
    ENV_MAPPING = {
        'POLYGON_API_KEY': 'polygon_api_key',
        'DB_PASSWORD': 'database_password',
        'POSTGRES_PASSWORD': 'database_password',
        'DATABASE_PASSWORD': 'database_password',
    }

    def __init__(self, credential_manager: CredentialManager):
        self.manager = credential_manager

    def load_from_environment(self, prefix: str = "", rotation_days: Optional[int] = 90) -> Dict[str, bool]:
        """
        Load credentials from environment variables

        Args:
            prefix: Environment variable prefix to filter by
            rotation_days: Default rotation period in days

        Returns:
            Dict mapping credential names to load success status
        """
        results = {}

        for env_var, cred_type in self.ENV_MAPPING.items():
            if prefix and not env_var.startswith(prefix):
                continue

            value = os.getenv(env_var)
            if value:
                try:
                    # Use environment variable name as credential name
                    cred_name = env_var.lower()
                    self.manager.add_credential(cred_name, value, cred_type, rotation_days)
                    results[cred_name] = True
                    logger.debug(f"Loaded credential from environment: {cred_name}")
                except CredentialValidationError as e:
                    logger.error(f"Failed to load credential {env_var}: {e.message}",
                               extra={"env_var": env_var, "error": e.message})
                    results[env_var.lower()] = False
            else:
                logger.debug(f"Environment variable not set: {env_var}")

        loaded_count = sum(1 for success in results.values() if success)
        logger.info(f"Environment credential loading completed: {loaded_count} loaded, {len(results) - loaded_count} failed",
                   extra={"loaded_count": loaded_count, "failed_count": len(results) - loaded_count})

        return results


# Global credential manager instance
credential_manager = CredentialManager()

# Environment loader instance
env_loader = EnvironmentCredentialLoader(credential_manager)


def validate_api_key(api_key: str, provider: str = "generic") -> bool:
    """
    Convenience function to validate an API key

    Args:
        api_key: The API key to validate
        provider: API provider name for context

    Returns:
        True if valid, False otherwise
    """
    validator = APIKeyValidator(provider)
    return validator.validate(api_key)


def validate_database_password(password: str) -> bool:
    """
    Convenience function to validate a database password

    Args:
        password: The password to validate

    Returns:
        True if valid, False otherwise
    """
    validator = DatabasePasswordValidator()
    return validator.validate(password)


def mask_credential(credential: str) -> str:
    """
    Mask a credential for safe display/logging

    Args:
        credential: The credential to mask

    Returns:
        Masked representation
    """
    if not credential:
        return ""

    length = len(credential)
    if length <= 8:
        return "*" * length
    else:
        return f"{credential[:4]}{'*' * (length - 8)}{credential[-4:]}"


def load_credentials_from_env(prefix: str = "", rotation_days: int = 90) -> Dict[str, bool]:
    """
    Load credentials from environment variables

    Args:
        prefix: Environment variable prefix filter
        rotation_days: Default credential rotation period

    Returns:
        Dict mapping credential names to success status
    """
    return env_loader.load_from_environment(prefix, rotation_days)
