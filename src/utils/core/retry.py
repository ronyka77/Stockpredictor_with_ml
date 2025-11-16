"""
Comprehensive retry mechanism framework with exponential backoff and circuit breaker patterns.

This module provides robust error recovery utilities for distributed systems and API clients,
addressing reliability issues with configurable retry policies and fault tolerance patterns.
"""

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retryable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,  # Covers various network-related errors
    )
    non_retryable_exceptions: tuple[Type[Exception], ...] = ()
    retry_on_result: Optional[Callable[[Any], bool]] = (
        None  # Function to check if result should trigger retry
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes needed to close circuit
    timeout: float = 60.0  # Time before attempting half-open


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.state == CircuitBreakerState.OPEN
            and time.time() - self.last_failure_time >= self.config.timeout
        )

    def _record_success(self):
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success

    def _record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker."""
        self.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def _close_circuit(self):
        """Close the circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker closed - service recovered")

    def _half_open_circuit(self):
        """Move to half-open state to test recovery."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        logger.info("Circuit breaker half-open - testing service recovery")

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker (synchronous)."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._half_open_circuit()
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    async def acall(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._half_open_circuit()
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for exponential backoff with optional jitter."""
    delay = min(config.base_delay * (config.backoff_factor**attempt), config.max_delay)

    if config.jitter:
        # Add random jitter (±25% of delay)
        import random

        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def is_retryable_exception(exception: Exception, config: RetryConfig) -> bool:
    """Check if an exception should trigger a retry."""
    # Check non-retryable exceptions first
    if isinstance(exception, config.non_retryable_exceptions):
        return False

    # Check retryable exceptions
    if isinstance(exception, config.retryable_exceptions):
        return True

    # Special handling for exceptions with error_category attribute (like PolygonAPIError)
    if hasattr(exception, "error_category") and exception.error_category == "retryable":
        return True

    return False


def _execute_with_retry(
    func: Callable[..., T],
    config: RetryConfig,
    circuit_breaker: Optional[CircuitBreaker],
    *args,
    **kwargs,
) -> T:
    """Execute function with retry logic (helper function to reduce complexity)."""
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            if circuit_breaker:
                return circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if this exception should trigger retry
            if not is_retryable_exception(e, config):
                logger.debug(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed "
                    f"({type(e).__name__}). Retrying in {delay:.2f}s: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_attempts} attempts failed. "
                    f"Last error: {type(e).__name__}: {e}"
                )

    # All attempts exhausted
    raise RetryError(
        f"Operation failed after {config.max_attempts} attempts",
        last_exception,
        config.max_attempts,
    )


def retry(
    config: Optional[RetryConfig] = None, circuit_breaker: Optional[CircuitBreaker] = None
) -> Callable:
    """
    Decorator for synchronous functions with retry logic and circuit breaker.

    Args:
        config: Retry configuration. Uses defaults if None.
        circuit_breaker: Optional circuit breaker for fault tolerance.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry(config=RetryConfig(max_attempts=5, base_delay=2.0))
        def api_call():
            return make_request()
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Check if function is async (coroutine function)
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs) -> T:
                return await _execute_with_async_retry(
                    func, config, circuit_breaker, *args, **kwargs
                )

            return async_wrapper
        else:

            def wrapper(*args, **kwargs) -> T:
                return _execute_with_retry(func, config, circuit_breaker, *args, **kwargs)

            return wrapper

    return decorator


async def _execute_with_async_retry(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig,
    circuit_breaker: Optional[CircuitBreaker],
    *args,
    **kwargs,
) -> T:
    """Execute async function with retry logic (helper function to reduce complexity)."""
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            if circuit_breaker:
                return await circuit_breaker.acall(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if this exception should trigger retry
            if not is_retryable_exception(e, config):
                logger.debug(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed "
                    f"({type(e).__name__}). Retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_attempts} attempts failed. "
                    f"Last error: {type(e).__name__}: {e}"
                )

    # All attempts exhausted
    raise RetryError(
        f"Async operation failed after {config.max_attempts} attempts",
        last_exception,
        config.max_attempts,
    )


def async_retry(
    config: Optional[RetryConfig] = None, circuit_breaker: Optional[CircuitBreaker] = None
) -> Callable:
    """
    Decorator for asynchronous functions with retry logic and circuit breaker.

    Args:
        config: Retry configuration. Uses defaults if None.
        circuit_breaker: Optional circuit breaker for fault tolerance.

    Returns:
        Decorated async function with retry logic.

    Example:
        @async_retry(config=RetryConfig(max_attempts=5, base_delay=2.0))
        async def api_call():
            return await make_async_request()
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            return await _execute_with_async_retry(func, config, circuit_breaker, *args, **kwargs)

        return wrapper

    return decorator


# Convenience configurations for common use cases

# API retry configuration - handles network errors and rate limits
API_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    non_retryable_exceptions=(),  # API errors like 401, 403 might not be retryable
)

# Database retry configuration - handles temporary connection issues
DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    backoff_factor=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    non_retryable_exceptions=(),
)

# File operation retry configuration - handles temporary file system issues
FILE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.1,
    max_delay=1.0,
    backoff_factor=1.5,
    retryable_exceptions=(OSError, IOError),
    non_retryable_exceptions=(),
)


# Convenience circuit breaker configurations
API_CIRCUIT_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout=60.0)
)

DATABASE_CIRCUIT_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=3, success_threshold=1, timeout=30.0)
)
