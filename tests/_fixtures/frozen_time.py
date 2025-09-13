from contextlib import contextmanager
import pytest


class FrozenClock:
    def __init__(self, start: float = 0.0):
        self._now = float(start)

    def time(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)


@pytest.fixture
def frozen_time(mocker):
    """Provide a factory that freezes time using pytest-mock's mocker.

    Usage in tests:
        def test_x(frozen_time):
            clock = frozen_time(start=0.0)
            # calls to time.time() and time.sleep() use the frozen clock

    The mocker.patch calls are automatically undone at test end.
    """

    def _freeze(start: float = 0.0):
        clock = FrozenClock(start=start)
        mocker.patch("time.time", clock.time)
        mocker.patch("time.sleep", clock.sleep)
        return clock

    return _freeze


# Backwards-compatible contextmanager for legacy tests that still call
# `with freeze_time(mocker, start=...)`.
@contextmanager
def freeze_time(mocker, start: float = 0.0):
    clock = FrozenClock(start=start)
    # Use mocker.patch to avoid direct global assignment
    patch_time = mocker.patch("time.time", clock.time)
    patch_sleep = mocker.patch("time.sleep", clock.sleep)
    try:
        yield clock
    finally:
        # mocker will restore on test teardown; explicitly stop patches now
        try:
            patch_time.stop()
        except Exception:
            pass
        try:
            patch_sleep.stop()
        except Exception:
            pass
