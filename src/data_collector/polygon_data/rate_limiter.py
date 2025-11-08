"""
Rate limiting functionality for Polygon.io API requests
"""

import time
from dataclasses import dataclass

from src.utils.core.logger import get_logger
from src.data_collector.config import config

logger = get_logger(__name__, utility="data_collector")


@dataclass
class RateLimiter:
    """
    Rate limiter for API requests with sliding window approach

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        last_request_time: Timestamp of last request
        request_count: Number of requests in current window
        window_start: Start time of current rate limiting window
    """

    requests_per_minute: int = 5
    last_request_time: float = 0
    request_count: int = 0
    window_start: float = 0

    def __post_init__(self):
        """Initialize the rate limiter"""
        self.window_start = time.time()

    def wait_if_needed(self) -> None:
        """
        Check rate limits and wait if necessary before making a request

        This implements a sliding window rate limiter that ensures we don't
        exceed the API rate limits. For free tier, it proactively waits
        after hitting the request limit.
        """
        current_time = time.time()

        # Reset window if minute has passed
        if current_time - self.window_start >= 60:
            self.window_start = current_time
            self.request_count = 0
            logger.info("Rate limit window reset")

        # Check if we've hit the limit
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.window_start)
            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached ({self.request_count}/{self.requests_per_minute}). "
                    f"Sleeping for {sleep_time:.2f} seconds"
                )
                time.sleep(sleep_time)
                # Update window start using current time after sleep
                self.window_start = time.time()
                self.request_count = 0
                # Refresh current_time to reflect time after sleep
                current_time = time.time()

        self.request_count += 1
        self.last_request_time = current_time

        logger.info(f"Request {self.request_count}/{self.requests_per_minute} in current window")

    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window"""
        current_time = time.time()

        # If window has expired, we have full quota
        if current_time - self.window_start >= 60:
            return self.requests_per_minute

        return max(0, self.requests_per_minute - self.request_count)

    def get_time_until_reset(self) -> float:
        """Get time in seconds until rate limit window resets"""
        current_time = time.time()
        time_elapsed = current_time - self.window_start

        if time_elapsed >= 60:
            return 0

        return 60 - time_elapsed

    def reset(self) -> None:
        """Manually reset the rate limiter"""
        self.window_start = time.time()
        self.request_count = 0
        logger.info("Rate limiter manually reset")

    def __str__(self) -> str:
        """String representation of rate limiter status"""
        remaining = self.get_remaining_requests()
        time_until_reset = self.get_time_until_reset()

        return (
            f"RateLimiter(requests: {self.request_count}/{self.requests_per_minute}, "
            f"remaining: {remaining}, reset_in: {time_until_reset:.1f}s)"
        )


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that can adjust based on API responses

    This extends the basic rate limiter to handle cases where the API
    might return rate limit errors, allowing us to back off more aggressively.
    """

    def __init__(self, requests_per_minute: int = 5, backoff_factor: float = 0.8):
        super().__init__(requests_per_minute=requests_per_minute)
        self.original_limit = requests_per_minute
        self.backoff_factor = backoff_factor
        self.consecutive_errors = 0

    def handle_rate_limit_error(self) -> None:
        """
        Handle a rate limit error by reducing the request rate
        """
        self.consecutive_errors += 1
        old_limit = self.requests_per_minute

        # For free tier (5 requests/minute), be more conservative
        if old_limit <= 5:
            # Reduce to 3 requests/minute for free tier
            self.requests_per_minute = max(2, int(old_limit * 0.6))
        else:
            # Standard backoff for paid tiers
            self.requests_per_minute = max(1, int(self.requests_per_minute * self.backoff_factor))

        logger.warning(
            f"Rate limit error detected. Reducing limit from {old_limit} to "
            f"{self.requests_per_minute} requests/minute"
        )

        # Reset current window to apply new limit immediately
        self.reset()

        # Add extra wait time after rate limit error
        logger.info("Adding 30 second cooldown after rate limit error")
        time.sleep(30)

    def handle_successful_request(self) -> None:
        """
        Handle a successful request - gradually restore original rate if needed
        """
        if self.consecutive_errors > 0:
            self.consecutive_errors = max(0, self.consecutive_errors - 1)

            # If we've had several successful requests, try to restore rate
            if self.consecutive_errors == 0 and self.requests_per_minute < self.original_limit:
                old_limit = self.requests_per_minute
                self.requests_per_minute = min(
                    self.original_limit, int(self.requests_per_minute / self.backoff_factor)
                )

                logger.info(
                    f"Restoring rate limit from {old_limit} to "
                    f"{self.requests_per_minute} requests/minute"
                )

    def __str__(self) -> str:
        """String representation including adaptive information"""
        base_str = super().__str__()
        return (
            f"{base_str[:-1]}, errors: {self.consecutive_errors}, "
            f"original_limit: {self.original_limit})"
        )


class NoOpRateLimiter(RateLimiter):
    """
    No-op limiter that performs no waiting or backoff. Used when rate limiting is disabled.
    """

    def __init__(self):
        # Initialize with very high limits but effectively do nothing
        super().__init__(requests_per_minute=10_000_000)

    def wait_if_needed(self) -> None:
        """
        No-op implementation - performs no waiting or rate limiting.
        """
        # Do nothing
        return

    def get_remaining_requests(self) -> int:
        """
        Return a very high number to indicate unlimited requests available.

        Returns:
            int: A large number indicating unlimited remaining requests
        """
        return 10_000_000

    def get_time_until_reset(self) -> float:
        """
        Return 0.0 since rate limiting is disabled.

        Returns:
            float: Always returns 0.0 indicating no waiting needed
        """
        return 0.0

    def handle_successful_request(self) -> None:
        """
        No-op implementation - no rate limit tracking needed.
        """
        return

    def reset(self) -> None:
        """
        No-op implementation - no rate limit state to reset.
        """
        return


def get_rate_limiter(requests_per_minute: int) -> RateLimiter:
    """
    Factory to return the appropriate rate limiter depending on configuration.
    """
    if getattr(config, "DISABLE_RATE_LIMITING", False):
        return NoOpRateLimiter()
    return AdaptiveRateLimiter(requests_per_minute=requests_per_minute)
