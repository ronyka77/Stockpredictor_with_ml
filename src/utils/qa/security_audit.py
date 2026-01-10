"""
Security Audit Logging System for StockPredictor V1
This module provides specialized security event logging capabilities that extend
the centralized logging system with audit trails for security monitoring and
incident response.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.utils.core.logger import get_logger

# Initialize security logger
security_logger = get_logger(__name__, "security")

# Compact validation logging artifacts and helpers


@dataclass
class CompactValidationIssue:
    """Normalized representation of a single validation problem for compact logs."""

    field: str
    reason: str
    value: Optional[Union[str, int, float]] = None
    source: Optional[str] = None
    context: Dict[str, str] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce invariants
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("field must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        # Truncate and ASCII-normalize context values to keep logs compact
        for k, v in list(self.context.items()):
            s = _truncate_ascii(v, 120)
            self.context[k] = s


@dataclass
class CompactValidationLogConfig:
    """Configuration knobs for compact validation logging."""

    max_issues: int = 3
    max_value_len: int = 120
    max_reason_len: int = 200


def _truncate_ascii(value: object, max_len: int) -> str:
    """Return an ASCII-only, truncated string representation of value."""
    s = str(value)
    # Replace non-ascii chars with '?' to ensure ASCII-only logs
    s = s.encode("ascii", "replace").decode("ascii")
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_compact_issue(issue: CompactValidationIssue, cfg: CompactValidationLogConfig) -> str:
    """Format a single CompactValidationIssue into a short key=value fragment."""
    reason = _truncate_ascii(issue.reason, cfg.max_reason_len)
    parts: List[str] = [f"field={issue.field}", f'reason="{reason}"']
    if issue.value is not None:
        parts.append(f"value={_truncate_ascii(issue.value, cfg.max_value_len)}")
    if issue.source:
        parts.append(f"source={issue.source}")
    if issue.context:
        ctx = ",".join(f"{k}={_truncate_ascii(v, 60)}" for k, v in issue.context.items())
        parts.append(f"context={ctx}")
    return " ".join(parts)


def emit_compact_validation_log(
    issues: List[CompactValidationIssue],
    *,
    service: str = "unknown",
    cfg: Optional[CompactValidationLogConfig] = None,
) -> None:
    """Emit a single, compact validation log line summarizing the supplied issues.

    The message is intentionally single-line and ASCII-only to reduce noise in
    logging backends and to prevent generation of structured security audit
    events for routine validation failures.
    """
    cfg = cfg or CompactValidationLogConfig()
    logger = get_logger("validation")

    total = len(issues)
    shown = issues[: cfg.max_issues]
    fragments = [_format_compact_issue(i, cfg) for i in shown]
    msg = f"VALIDATION: {' | '.join(fragments)}"
    if total > cfg.max_issues:
        msg += f" ({total - cfg.max_issues} more issues suppressed)"

    # Use warning level for validation failures to make them visible but not
    # part of the security audit stream.
    logger.warning(msg)


def _make_issues_from_inputs(
    resource: str, validation_errors: List[str], input_source: str, details: Dict[str, Any]
) -> List[CompactValidationIssue]:
    """Create CompactValidationIssue objects from normalized inputs."""
    issues: List[CompactValidationIssue] = []
    field_name = details.get("field") or resource or "unknown"
    value = details.get("value")

    for err in validation_errors:
        issue = CompactValidationIssue(
            field=str(field_name),
            reason=str(err),
            value=value,
            source=input_source,
            context={k: str(v) for k, v in details.items() if k not in ("value", "field")},
        )
        issues.append(issue)
    return issues


class SecurityEventType(Enum):
    """Enumeration of security event types for audit logging."""

    # Authentication & Authorization Events
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TOKEN_VALIDATION = "token_validation"

    # Input Validation Events
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    MALICIOUS_INPUT_DETECTED = "malicious_input_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"

    # Data Access Events
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    DATA_EXPORT_ATTEMPT = "data_export_attempt"
    CONFIGURATION_CHANGE = "configuration_change"

    # System Security Events
    FILE_ACCESS_VIOLATION = "file_access_violation"
    NETWORK_ACCESS_VIOLATION = "network_access_violation"
    PROCESS_EXECUTION_ATTEMPT = "process_execution_attempt"

    # Credential Events
    CREDENTIAL_VALIDATION = "credential_validation"
    CREDENTIAL_ROTATION = "credential_rotation"
    API_KEY_EXPOSURE = "api_key_exposure"

    # Compliance Events
    AUDIT_LOG_ACCESS = "audit_log_access"
    SECURITY_POLICY_VIOLATION = "security_policy_violation"


class SecurityEventSeverity(Enum):
    """Security event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(BaseModel):
    """Structured security event for audit logging."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    service: str
    severity: SecurityEventSeverity
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    outcome: str = "unknown"  # success, failure, blocked, allowed
    risk_score: Optional[int] = Field(None, ge=0, le=100)

    # Tamper-evident fields
    integrity_hash: Optional[str] = None

    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            SecurityEventType: lambda v: v.value,
            SecurityEventSeverity: lambda v: v.value,
        },
    )

    @field_validator("integrity_hash", mode="before")
    @classmethod
    def compute_integrity_hash(cls, v, info):
        """Compute tamper-evident hash of the event data."""
        if v is not None:
            return v  # Already set

        # Get field values from validation info
        data = info.data

        # Create hash of key event fields (excluding the hash itself)
        event_data = {
            "event_id": data.get("event_id", ""),
            "event_type": str(data.get("event_type", "")),
            "timestamp": data.get("timestamp", "").isoformat() if data.get("timestamp") else "",
            "service": data.get("service", ""),
            "severity": str(data.get("severity", "")),
            "user_id": data.get("user_id", ""),
            "session_id": data.get("session_id", ""),
            "resource": data.get("resource", ""),
            "action": data.get("action", ""),
            "outcome": data.get("outcome", ""),
            "details": json.dumps(data.get("details", {}), sort_keys=True),
        }

        # Create deterministic string representation
        data_string = json.dumps(event_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data_string.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with integrity verification."""
        data = self.model_dump()

        # Verify integrity hash by recomputing it
        timestamp = data.get("timestamp", "")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        event_data = {
            "event_id": data.get("event_id", ""),
            "event_type": str(data.get("event_type", "")),
            "timestamp": timestamp,
            "service": data.get("service", ""),
            "severity": str(data.get("severity", "")),
            "user_id": data.get("user_id", ""),
            "session_id": data.get("session_id", ""),
            "resource": data.get("resource", ""),
            "action": data.get("action", ""),
            "outcome": data.get("outcome", ""),
            "details": json.dumps(data.get("details", {}), sort_keys=True),
        }
        data_string = json.dumps(event_data, sort_keys=True, separators=(",", ":"))
        computed_hash = hashlib.sha256(data_string.encode("utf-8")).hexdigest()

        stored_hash = data.get("integrity_hash")
        if stored_hash and stored_hash != computed_hash:
            security_logger.warning(
                f"Security event integrity violation detected for event {self.event_id}"
            )
            # Don't expose the event if integrity is compromised
            return {"error": "integrity_violation", "event_id": self.event_id}
        return data


class SecurityAuditLogger:
    """Central security audit logging system."""

    def __init__(self, service_name: str = "datacollector"):
        self.service_name = service_name
        self.logger = get_logger(f"security_audit.{service_name}", "security")

    def log_security_event(
        self,
        event_type: Union[SecurityEventType, str],
        severity: SecurityEventSeverity = SecurityEventSeverity.MEDIUM,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "unknown",
        risk_score: Optional[int] = None,
    ) -> str:
        """
        Log a security event with full audit trail.

        Args:
            event_type: Type of security event
            severity: Event severity level
            user_id: User identifier (sanitized)
            session_id: Session identifier
            ip_address: Client IP address (sanitized)
            user_agent: User agent string (sanitized)
            resource: Resource being accessed
            action: Action being performed
            details: Additional event details (sensitive data will be sanitized)
            outcome: Event outcome (success, failure, blocked, allowed)
            risk_score: Risk score from 0-100

        Returns:
            Event ID for tracking
        """
        # Convert string event type to enum if needed
        if isinstance(event_type, str):
            try:
                event_type = SecurityEventType(event_type)
            except ValueError:
                security_logger.warning(f"Unknown security event type: {event_type}")
                event_type = SecurityEventType.SECURITY_POLICY_VIOLATION

        # Sanitize sensitive fields
        sanitized_details = self._sanitize_details(details or {})

        # Create security event
        event = SecurityEvent(
            event_type=event_type,
            service=self.service_name,
            severity=severity,
            user_id=self._sanitize_user_id(user_id),
            session_id=self._sanitize_session_id(session_id),
            ip_address=self._sanitize_ip_address(ip_address),
            user_agent=self._sanitize_user_agent(user_agent),
            resource=self._sanitize_resource(resource),
            action=self._sanitize_action(action),
            details=sanitized_details,
            outcome=outcome,
            risk_score=risk_score,
        )

        # Log the event
        self._log_event(event)

        return event.event_id

    def log_authentication_success(
        self,
        user_id: str,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log successful authentication event."""
        return self.log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            severity=SecurityEventSeverity.LOW,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            outcome="success",
            details=details,
        )

    def log_authentication_failure(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log authentication failure event."""
        event_details = {"failure_reason": failure_reason}
        if details:
            event_details.update(details)

        return self.log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            severity=SecurityEventSeverity.MEDIUM,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            outcome="failure",
            details=event_details,
            risk_score=self._calculate_risk_score("auth_failure", failure_reason),
        )

    def log_input_validation_failure(
        self,
        resource: str,
        validation_errors: List[str],
        input_source: str = "unknown",
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Emit a compact validation log instead of an audit security event.

        For backward compatibility we return a stable placeholder event id.
        This method intentionally does NOT call `log_security_event` or
        `_log_event` to avoid creating security audit noise for routine
        validation failures.
        """
        details = details or {}

        # Build compact issues and emit a single-line compact log
        issues = _make_issues_from_inputs(resource, validation_errors, input_source, details)
        emit_compact_validation_log(issues, service=self.service_name)

        # Return placeholder event id for callers that expect a string identifier
        return "suppressed_input_validation_failure"

    def log_credential_validation(
        self,
        service: str,
        validation_result: bool,
        credential_type: str = "api_key",
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log credential validation event."""
        event_details = {"credential_type": credential_type, "service": service}
        if details:
            event_details.update(details)

        severity = SecurityEventSeverity.LOW if validation_result else SecurityEventSeverity.MEDIUM

        return self.log_security_event(
            event_type=SecurityEventType.CREDENTIAL_VALIDATION,
            severity=severity,
            user_id=user_id,
            resource=service,
            action="validate_credential",
            outcome="success" if validation_result else "failure",
            details=event_details,
        )

    def _log_event(self, event: SecurityEvent) -> None:
        """Log the security event using structured logging."""
        event_dict = event.to_dict()

        # Create structured log message
        log_message = f"SECURITY EVENT [{event.event_type}] {event.service}: {event.outcome}"

        # Add context for high-severity events
        if event.severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]:
            log_message += f" - HIGH RISK EVENT (severity: {event.severity})"

        # Log with appropriate level
        if event.severity == SecurityEventSeverity.CRITICAL:
            self.logger.error(log_message, extra={"security_event": event_dict})
        elif event.severity == SecurityEventSeverity.HIGH:
            self.logger.warning(log_message, extra={"security_event": event_dict})
        elif event.severity == SecurityEventSeverity.MEDIUM:
            self.logger.warning(log_message, extra={"security_event": event_dict})
        else:
            self.logger.info(log_message, extra={"security_event": event_dict})

    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data from event details."""
        sanitized = {}
        sensitive_keys = {
            "password",
            "api_key",
            "secret",
            "token",
            "key",
            "credential",
            "auth_token",
            "access_token",
            "refresh_token",
            "bearer_token",
            "authorization",
            "cookie",
            "session_token",
        }

        for key, value in details.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = self._mask_sensitive_value(str(value))
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Sanitize user identifier."""
        if not user_id:
            return None
        # Hash user IDs to prevent exposure while maintaining uniqueness for correlation
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]

    def _sanitize_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Sanitize session identifier."""
        if not session_id:
            return None
        # Session IDs are typically safe but we'll truncate for consistency
        return session_id[:32] + "..." if len(session_id) > 32 else session_id

    def _sanitize_ip_address(self, ip_address: Optional[str]) -> Optional[str]:
        """Sanitize IP address."""
        if not ip_address:
            return None
        # For IPv4, mask last octet; for IPv6, mask last segment
        if ":" in ip_address:  # IPv6
            parts = ip_address.split(":")
            if len(parts) > 4:
                return ":".join(parts[:-2]) + "::masked"
        else:  # IPv4
            parts = ip_address.split(".")
            if len(parts) == 4:
                return ".".join(parts[:-1]) + ".***"
        return ip_address

    def _sanitize_user_agent(self, user_agent: Optional[str]) -> Optional[str]:
        """Sanitize user agent string."""
        if not user_agent:
            return None
        # Truncate very long user agents
        return user_agent[:200] + "..." if len(user_agent) > 200 else user_agent

    def _sanitize_resource(self, resource: Optional[str]) -> Optional[str]:
        """Sanitize resource identifier."""
        if not resource:
            return None
        # Remove potential sensitive path information
        if "password" in resource.lower() or "secret" in resource.lower():
            return "sensitive_resource"
        return resource

    def _sanitize_action(self, action: Optional[str]) -> Optional[str]:
        """Sanitize action description."""
        if not action:
            return None
        # Actions are generally safe, but ensure they're reasonable length
        return action[:100] + "..." if len(action) > 100 else action

    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive values for logging."""
        if len(value) <= 4:
            return "***"
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def _calculate_risk_score(self, event_category: str, context: Any) -> int:
        """Calculate risk score based on event type and context."""
        base_scores = {
            "auth_failure": 60,
            "validation_failure": 40,
            "malicious_input": 80,
            "credential_exposure": 95,
        }

        base_score = base_scores.get(event_category, 30)

        # Adjust based on context
        if event_category == "auth_failure":
            if isinstance(context, str) and "brute_force" in context.lower():
                base_score += 20
        elif event_category == "validation_failure":
            if isinstance(context, int) and context > 5:
                base_score += 15

        return min(100, max(0, base_score))


# Global audit logger instance
audit_logger = SecurityAuditLogger()


def log_security_event(
    event_type: Union[SecurityEventType, str],
    severity: SecurityEventSeverity = SecurityEventSeverity.MEDIUM,
    **kwargs,
) -> str:
    """
    Convenience function to log security events.

    Args:
        event_type: Type of security event
        severity: Event severity level
        **kwargs: Additional event parameters

    Returns:
        Event ID for tracking
    """
    return audit_logger.log_security_event(event_type, severity, **kwargs)


def log_authentication_success(**kwargs) -> str:
    """Log successful authentication event."""
    return audit_logger.log_authentication_success(**kwargs)


def log_authentication_failure(**kwargs) -> str:
    """Log authentication failure event."""
    return audit_logger.log_authentication_failure(**kwargs)


def log_input_validation_failure(**kwargs) -> str:
    """Log input validation failure using compact validation logs.

    This wrapper normalizes calling forms and emits compact validation logs
    instead of security audit events. It intentionally does NOT call into
    the security audit event pipeline to avoid emitting `SECURITY EVENT [`
    lines for routine validation failures.
    """
    data = dict(kwargs)  # copy so we can mutate safely

    # Gather and normalize details
    details = data.pop("details", {}) or {}

    if "field" in data:
        details.setdefault("field", data.pop("field"))
    if "value" in data:
        details.setdefault("value", data.pop("value"))

    # Support single-string validation_error or list of errors
    if "validation_error" in data:
        ve = data.pop("validation_error")
        validation_errors = ve if isinstance(ve, list) else [str(ve)]
    else:
        validation_errors = data.pop("validation_errors", None)

    # Map severity into details (not used for structured audit events, but kept)
    if "severity" in data:
        details.setdefault("severity", data.pop("severity"))

    # Extract canonical args
    resource = data.pop("resource", None) or details.get("field") or "unknown"
    input_source = data.pop("input_source", data.pop("input", "unknown"))
    data.pop("ip_address", None)
    data.pop("user_id", None)

    # Anything else leftover should be attached to details
    for k, v in list(data.items()):
        details.setdefault(k, v)

    if not validation_errors:
        validation_errors = ["validation_failed"]

    # Build compact issues and emit the compact log
    issues = _make_issues_from_inputs(resource, validation_errors, input_source, details)
    emit_compact_validation_log(issues)

    # Return the placeholder for backward compatibility
    return "suppressed_input_validation_failure"


def log_credential_validation(**kwargs) -> str:
    """Log credential validation event."""
    return audit_logger.log_credential_validation(**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    logger = SecurityAuditLogger("test_service")

    # Test authentication events
    event_id = logger.log_authentication_success(
        user_id="user123",
        session_id="session_abc123",
        ip_address="192.168.1.100",
        details={"method": "password"},
    )
    print(f"Logged auth success event: {event_id}")

    # Test validation failure
    event_id = logger.log_input_validation_failure(
        resource="api/stocks",
        validation_errors=["Invalid ticker symbol format", "Missing required field"],
        ip_address="10.0.0.1",
        details={"input_length": 1500},
    )
    print(f"Logged validation failure event: {event_id}")

    # Test credential validation
    event_id = logger.log_credential_validation(
        service="polygon_api",
        validation_result=False,
        credential_type="api_key",
        details={"reason": "invalid_format"},
    )
    print(f"Logged credential validation event: {event_id}")
