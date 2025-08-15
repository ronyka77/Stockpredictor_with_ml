import pytest

from tests._fixtures.frozen_time import freeze_time
from src.data_collector.polygon_data.rate_limiter import RateLimiter, AdaptiveRateLimiter


@pytest.mark.unit
def test_rate_limiter_sliding_window(monkeypatch):
    with freeze_time(monkeypatch, start=0.0):
        rl = RateLimiter(requests_per_minute=3)

        # First 3 requests should not sleep and consume quota
        for _ in range(3):
            rl.wait_if_needed()
        assert rl.request_count == 3
        remaining_before = rl.get_remaining_requests()
        assert remaining_before == 0

        # Next call should sleep until window reset (simulate via frozen clock)
        rl.wait_if_needed()
        # wait_if_needed calls time.sleep(...) which advances clock via freeze_time
        # After sleep, a new window starts with count reset to 1
        assert rl.request_count == 1
        assert rl.get_remaining_requests() == 2

        # Advance partial time, ensure time until reset decreases
        ttr = rl.get_time_until_reset()
        assert 0 < ttr <= 60


@pytest.mark.unit
def test_rate_limiter_manual_reset(monkeypatch):
    with freeze_time(monkeypatch, start=100.0):
        rl = RateLimiter(requests_per_minute=5)
        for _ in range(4):
            rl.wait_if_needed()
        assert rl.get_remaining_requests() == 1

        rl.reset()
        assert rl.request_count == 0
        assert rl.get_remaining_requests() == 5


@pytest.mark.unit
def test_adaptive_rate_limiter_backoff_and_restore(monkeypatch):
    with freeze_time(monkeypatch, start=1000.0):
        arl = AdaptiveRateLimiter(requests_per_minute=5, backoff_factor=0.5)

        # Trigger rate limit error → backoff and cooldown sleep(30)
        before = arl.requests_per_minute
        arl.handle_rate_limit_error()
        assert arl.requests_per_minute <= before
        # Ensure cooldown advanced time by 30s implicitly
        # Next successful requests reduce error counter and may restore
        arl.handle_successful_request()
        arl.handle_successful_request()
        # After clearing consecutive_errors, restore toward original limit
        assert arl.requests_per_minute <= arl.original_limit


