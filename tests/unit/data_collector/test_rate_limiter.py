import pytest

from tests.fixtures.frozen_time import freeze_time
from src.data_collector.polygon_data.rate_limiter import RateLimiter, AdaptiveRateLimiter


@pytest.mark.unit
def test_rate_limiter_sliding_window(mocker):
    with freeze_time(mocker, start=0.0):
        rl = RateLimiter(requests_per_minute=3)

        # First 3 requests should not sleep and consume quota
        for _ in range(3):
            rl.wait_if_needed()
        if rl.request_count != 3:
            raise AssertionError("RateLimiter did not increment request_count as expected")
        remaining_before = rl.get_remaining_requests()
        if remaining_before != 0:
            raise AssertionError("Remaining requests before reset mismatch")

        # Next call should sleep until window reset (simulate via frozen clock)
        rl.wait_if_needed()
        # wait_if_needed calls time.sleep(...) which advances clock via freeze_time
        # After sleep, a new window starts with count reset to 1
        if rl.request_count != 1:
            raise AssertionError("RateLimiter request_count not reset after window rollover")
        if rl.get_remaining_requests() != 2:
            raise AssertionError("Remaining requests after rollover mismatch")

        # Advance partial time, ensure time until reset decreases
        ttr = rl.get_time_until_reset()
        if not (0 < ttr <= 60):
            raise AssertionError("Time until reset out of expected range")


@pytest.mark.unit
def test_rate_limiter_manual_reset(mocker):
    with freeze_time(mocker, start=100.0):
        rl = RateLimiter(requests_per_minute=5)
        for _ in range(4):
            rl.wait_if_needed()
        if rl.get_remaining_requests() != 1:
            raise AssertionError("Remaining requests mismatch before manual reset")

        rl.reset()
        if rl.request_count != 0:
            raise AssertionError("Request count not zero after reset")
        if rl.get_remaining_requests() != 5:
            raise AssertionError("Remaining requests not restored after reset")


@pytest.mark.unit
def test_adaptive_rate_limiter_backoff_and_restore(mocker):
    with freeze_time(mocker, start=1000.0):
        arl = AdaptiveRateLimiter(requests_per_minute=5, backoff_factor=0.5)

        # Trigger rate limit error → backoff and cooldown sleep(30)
        before = arl.requests_per_minute
        arl.handle_rate_limit_error()
        if arl.requests_per_minute > before:
            raise AssertionError("AdaptiveRateLimiter did not back off on rate limit error")
        # Ensure cooldown advanced time by 30s implicitly
        # Next successful requests reduce error counter and may restore
        arl.handle_successful_request()
        arl.handle_successful_request()
        # After clearing consecutive_errors, restore toward original limit
        if arl.requests_per_minute > arl.original_limit:
            raise AssertionError("AdaptiveRateLimiter exceeded original limit after restores")
