"""Centralized canned API responses for tests.

Provide simple FakeResponse and factory helpers so tests can patch network
calls and return deterministic payloads. Use `unittest.mock.patch` on the
module-qualified client methods and return these FakeResponse objects.
"""

from typing import Any, Dict, Optional


SAMPLE_GROUPED_DAILY = [{"T": "TST", "c": 1.5, "o": 1.0, "h": 2.0, "l": 0.5, "v": 100}]


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Optional[Dict[str, Any]] = None,
        raise_on_json: bool = False,
    ):
        self.status_code = status
        self._payload = payload or {"results": []}
        self._raise = raise_on_json

    def json(self) -> Dict[str, Any]:
        if self._raise:
            raise ValueError("malformed json")
        return self._payload

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")


def canned_api_factory(
    kind: str = "grouped_daily", status: int = 200, raise_on_json: bool = False
) -> FakeResponse:
    """Return a FakeResponse for the requested canned kind.

    Kinds supported:
    - "grouped_daily": returns a standard grouped-daily payload
    - "empty": returns an empty results payload
    - fallback returns empty results
    """
    if kind == "grouped_daily":
        payload = {"results": SAMPLE_GROUPED_DAILY, "next_url": None}
    elif kind == "empty":
        payload = {"results": [], "next_url": None}
    else:
        payload = {"results": []}

    return FakeResponse(status=status, payload=payload, raise_on_json=raise_on_json)
