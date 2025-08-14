import time
from contextlib import contextmanager


class FrozenClock:
    def __init__(self, start: float = 0.0):
        self._now = float(start)

    def time(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)

    def elapse(self, seconds: float) -> None:
        self._now += float(seconds)


@contextmanager
def freeze_time(monkeypatch, start: float = 0.0):
    clock = FrozenClock(start=start)
    monkeypatch.setattr(time, "time", clock.time)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    try:
        yield clock
    finally:
        # monkeypatch will restore automatically after test scope
        pass


