from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional

from faker import Faker
from polyfactory.factories.pydantic_factory import ModelFactory

from src.data_collector.polygon_data.data_validator import OHLCVRecord, TickerInfo
from src.database import fundamental_models as fm

# Central Faker instance used by factories (seed via `set_factory_seed`)
faker = Faker()


def set_factory_seed(seed: int = 42) -> None:
    """Seed Faker and other RNGs for deterministic test output."""
    try:
        faker.seed_instance(seed)
    except Exception:
        pass
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def build_fundamental_ratios(
    ticker: str = "AAPL", date: date = date(2025, 1, 1), **kwargs
) -> fm.FundamentalRatios:
    obj = fm.FundamentalRatios()
    obj.ticker = ticker
    obj.date = date
    obj.pe_ratio = None
    obj.pb_ratio = None
    obj.ps_ratio = None
    return obj


def build_fundamental_growth_metrics(
    ticker: str = "AAPL", date: date = date(2025, 1, 1), **kwargs
) -> fm.FundamentalGrowthMetrics:
    obj = fm.FundamentalGrowthMetrics()
    obj.ticker = ticker
    obj.date = date
    obj.revenue_growth_1y = None
    obj.revenue_growth_3y = None
    return obj


def build_fundamental_scores(
    ticker: str = "AAPL", date: date = date(2025, 1, 1), **kwargs
) -> fm.FundamentalScores:
    obj = fm.FundamentalScores()
    obj.ticker = ticker
    obj.date = date
    obj.altman_z_score = None
    obj.piotroski_f_score = None
    return obj


def build_fundamental_sector(
    ticker: str = "AAPL",
    date: date = date(2025, 1, 1),
    gics_sector: str = "Tech",
    **kwargs,
) -> fm.FundamentalSectorAnalysis:
    obj = fm.FundamentalSectorAnalysis()
    obj.ticker = ticker
    obj.date = date
    obj.gics_sector = gics_sector
    return obj


# Backwards-compatible factory wrappers for tests that expect `.build()` methods
class FundamentalRatiosFactory:
    @classmethod
    def build(cls, **overrides):
        return build_fundamental_ratios(**overrides)


class FundamentalGrowthMetricsFactory:
    @classmethod
    def build(cls, **overrides):
        return build_fundamental_growth_metrics(**overrides)


class FundamentalScoresFactory:
    @classmethod
    def build(cls, **overrides):
        return build_fundamental_scores(**overrides)


class FundamentalSectorAnalysisFactory:
    @classmethod
    def build(cls, **overrides):
        return build_fundamental_sector(**overrides)


# Expose .build on the builder functions for callers that expect a factory-like API
build_fundamental_ratios.build = build_fundamental_ratios
build_fundamental_growth_metrics.build = build_fundamental_growth_metrics
build_fundamental_scores.build = build_fundamental_scores
build_fundamental_sector.build = build_fundamental_sector


class OHLCVRecordFactory(ModelFactory[OHLCVRecord]):
    __model__ = OHLCVRecord

    @classmethod
    def build(cls, **overrides):
        """Create a coherent OHLCVRecord where high >= max(open, close) and low <= min(open, close)."""
        # Base price
        base = round(faker.pyfloat(min_value=1.0, max_value=500.0, right_digits=4), 4)

        # Random spreads above/below base
        up = round(
            faker.pyfloat(
                min_value=0.0, max_value=max(1.0, base * 0.2), right_digits=4
            ),
            4,
        )
        down = round(
            faker.pyfloat(
                min_value=0.0, max_value=max(1.0, base * 0.2), right_digits=4
            ),
            4,
        )

        high = round(base + up, 4)
        low = round(max(0.0001, base - down), 4)

        # Open and close within low..high
        open_p = round(faker.pyfloat(min_value=low, max_value=high, right_digits=4), 4)
        close_p = round(faker.pyfloat(min_value=low, max_value=high, right_digits=4), 4)

        volume = faker.pyint(min_value=0, max_value=10_000_000)

        ticker = (
            overrides.get("ticker") or faker.pystr(min_chars=1, max_chars=4).upper()
        )
        timestamp = (
            overrides.get("timestamp")
            or (
                datetime.now(timezone.utc).date()
                - timedelta(days=faker.pyint(min_value=0, max_value=365))
            ).isoformat()
        )

        payload = {
            "ticker": ticker,
            "timestamp": timestamp,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close_p,
            "volume": volume,
            "vwap": None,
            "adjusted_close": None,
        }

        payload.update(overrides)

        return super().build(**payload)


class TickerInfoFactory(ModelFactory[TickerInfo]):
    __model__ = TickerInfo

    @staticmethod
    def ticker() -> str:
        return faker.pystr(min_chars=1, max_chars=4).upper()

    @staticmethod
    def name() -> str:
        return faker.company()[:24]

    market = "stocks"
    locale = "us"
    primary_exchange = "NYSE"
    currency_name = "USD"
    active = True


class APIResponseFactory:
    """Lightweight factory to create paginated/canned API-like responses.

    Provides `.json()`-style dict outputs compatible with `canned_api_factory` usage in tests.
    """

    def __init__(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        status: str = "OK",
        next_url: Optional[str] = None,
    ):
        self._payload = {
            "status": status,
            "results": results or [],
            "next_url": next_url,
        }

    def json(self) -> Dict[str, Any]:
        return self._payload


# Adapter so existing `canned_api_factory` callers can use these factories
def ohlcv_payload_dict(**overrides) -> Dict[str, Any]:
    model = OHLCVRecordFactory.build()
    payload = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    payload.update(overrides)
    return payload


def polygon_payload_dict(**overrides) -> Dict[str, Any]:
    """Return a Polygon-style raw record with keys 't','o','h','l','c','v','T'.

    Converts an OHLCVRecord model into the polygon raw format used by
    Polygon.io endpoints (milliseconds timestamp and single-letter ticker key 'T').
    """
    model = OHLCVRecordFactory.build()
    payload = model.model_dump() if hasattr(model, "model_dump") else model.dict()

    # Ensure timestamp is a date/datetime and convert to milliseconds since epoch
    ts = payload.get("timestamp")
    if isinstance(ts, str):
        try:
            from datetime import datetime

            ts_dt = datetime.fromisoformat(ts)
        except Exception:
            ts_dt = datetime.now(timezone.utc)
    elif hasattr(ts, "timestamp"):
        ts_dt = ts
    else:
        from datetime import datetime

        ts_dt = datetime.now(timezone.utc)

    try:
        millis = int(ts_dt.timestamp() * 1000)
    except Exception:
        millis = int(datetime.now(timezone.utc).timestamp() * 1000)

    poly = {
        "T": payload.get("ticker") or payload.get("ticker", "TST"),
        "t": millis,
        "o": float(payload.get("open", 0)),
        "h": float(payload.get("high", 0)),
        "l": float(payload.get("low", 0)),
        "c": float(payload.get("close", 0)),
        "v": int(payload.get("volume", 0)),
    }

    poly.update(overrides)
    return poly
