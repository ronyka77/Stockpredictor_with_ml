import time
from contextlib import contextmanager


class FrozenClock:
    def __init__(self, start: float = 0.0):
        self._now = float(start)

    def time(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)


@contextmanager
def freeze_time(mocker, start: float = 0.0):
    clock = FrozenClock(start=start)
    with (
        mocker.patch.object(time, "time", new=clock.time),
        mocker.patch.object(time, "sleep", new=clock.sleep),
    ):
        yield clock
