"""Data validation schemas and validator for Polygon OHLCV data (compact)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_serializer, field_validator

from src.utils.core.logger import get_logger
from src.utils.core.validation import (
    SecureBaseModel,
    SecureNumeric,
    SecurityValidationError,
    ValidationUtils,
    validate_input_data,
)

logger = get_logger(__name__)


def _today() -> date:
    return datetime.now().date()


def _is_polygon_record(record: Dict[str, Any]) -> bool:
    return "t" in record or any(k in record for k in ("o", "h", "l", "c", "v", "vw", "T"))


def _polygon_timestamp(ms: Any) -> datetime:
    if not isinstance(ms, (int, float)):
        raise SecurityValidationError(f"Invalid timestamp type: {type(ms)}", security_threat=False)
    if not (0 < ms < 2147483647000):
        raise SecurityValidationError(f"Invalid timestamp value: {ms}", security_threat=True)
    return datetime.fromtimestamp(ms / 1000)


def _validated_ticker(raw: str, field: str) -> str:
    return ValidationUtils.validate_ticker_symbol(raw, field)


def _num_or_zero(value: Any, *, field: str, kind: str, log) -> float:
    if value is None:
        return 0.0
    try:
        SecureNumeric.validate_positive_number(value, field_name=field)
        return float(value)
    except SecurityValidationError:
        log.warning(f"Invalid {kind} in API response: {value}")
        return 0.0


def _int_or_zero(value: Any, *, field: str, kind: str, log) -> int:
    if value is None:
        return 0
    try:
        SecureNumeric.validate_range(value, min_val=0, max_val=1e12, field_name=field)
        return int(value)
    except SecurityValidationError:
        log.warning(f"Invalid {kind} in API response: {value}")
        return 0


class OHLCVRecord(SecureBaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    timestamp: Union[datetime, date]
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(..., ge=0)
    vwap: Optional[float] = Field(None, ge=0)

    @field_serializer("timestamp")
    def _serialize_timestamp(self, v, _info):
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        return v

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        try:
            return _validated_ticker(v, "ticker")
        except SecurityValidationError as e:
            raise ValueError(f"Ticker validation failed: {e.message}") from e

    @field_validator("open", "high", "low", "close", mode="before")
    @classmethod
    def validate_price_fields(cls, v, info):
        if v is None:
            return v
        try:
            return SecureNumeric.validate_positive_number(v, field_name=info.field_name)
        except SecurityValidationError as e:
            raise ValueError(f"Price validation failed for {info.field_name}: {e.message}") from e

    @field_validator("high", mode="after")
    @classmethod
    def validate_high_price(cls, v, info):
        values = info.data or {}
        for other in ("low", "open", "close"):
            if other in values and v < values[other]:
                raise ValueError(
                    f"High price ({v}) must be >= {other.title()} price ({values[other]})"
                )
        return v

    @field_validator("low", mode="after")
    @classmethod
    def validate_low_price(cls, v, info):
        values = info.data or {}
        for other in ("open", "close"):
            if other in values and v > values[other]:
                raise ValueError(
                    f"Low price ({v}) must be <= {other.title()} price ({values[other]})"
                )
        return v

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v):
        # Allow 0 volume (no trading activity)
        if v == 0:
            return v
        try:
            return SecureNumeric.validate_range(v, min_val=0, max_val=1e12, field_name="volume")
        except SecurityValidationError as e:
            raise ValueError(f"Volume validation failed: {e.message}") from e

    @field_validator("vwap", mode="before")
    @classmethod
    def validate_vwap(cls, v, info):
        values = info.data or {}
        if v is None:
            return v

        if v <= 0:
            if all(key in values for key in ["high", "low", "close"]):
                volume = values.get("volume", None)
                fallback_vwap = cls._calculate_fallback_vwap(
                    values["high"], values["low"], values["close"], volume
                )
                logger.warning(f"Invalid VWAP {v}, using fallback: {fallback_vwap}")
                return fallback_vwap
            else:
                raise ValueError("VWAP must be positive")

        if "high" in values and "low" in values:
            tolerance = 0.20
            low_bound = values["low"] * (1 - tolerance)
            high_bound = values["high"] * (1 + tolerance)

            if not (low_bound <= v <= high_bound):
                volume = values.get("volume", None)
                fallback_vwap = cls._calculate_fallback_vwap(
                    values["high"], values["low"], values["close"], volume
                )
                logger.warning(f"VWAP {v} outside tolerance, using fallback: {fallback_vwap}")
                return fallback_vwap

        return v

    @staticmethod
    def _calculate_fallback_vwap(
        high: float, low: float, close: float, volume: Optional[int] = None
    ) -> float:
        if volume and volume > 0:
            vwap = (high + low + 2 * close) / 4
        else:
            vwap = (high + low + close) / 3

        return round(vwap, 4)

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v):
        if isinstance(v, str):
            try:
                if "T" in v:
                    parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
                else:
                    parsed = datetime.strptime(v, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {v}")

            # assign parsed value back to v so subsequent future-date checks run
            v = parsed

        today = _today()
        if isinstance(v, datetime) and v.date() > today:
            raise ValueError(f"Timestamp cannot be in the future: {v}")
        if isinstance(v, date) and not isinstance(v, datetime) and v > today:
            raise ValueError(f"Date cannot be in the future: {v}")

        return v

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        if isinstance(data["timestamp"], datetime):
            data["date"] = data["timestamp"].date()
        else:
            data["date"] = data["timestamp"]
        data.pop("timestamp", None)

        return data


class TickerInfo(BaseModel):
    ticker: str
    name: str
    market: str = "stocks"
    locale: str = "us"
    primary_exchange: Optional[str] = None
    currency_name: Optional[str] = None
    active: bool = True

    @field_validator("ticker")
    @classmethod
    def validate_ticker_format(cls, v):
        """Validate ticker format"""
        return v.strip().upper()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate company/asset name"""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class DataQualityMetrics(BaseModel):
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    validation_errors: List[str] = []
    data_gaps: List[str] = []
    outliers: List[str] = []

    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100

    def add_error(self, error: str) -> None:
        self.validation_errors.append(error)
        self.invalid_records += 1


class DataValidator:
    def __init__(self):
        self.log = get_logger(f"{__name__}.{self.__class__.__name__}")

    def validate_ohlcv_record(
        self, record: Dict[str, Any], ticker: Optional[str] = None
    ) -> Optional[OHLCVRecord]:
        try:
            validate_input_data(record, strict=True)
            normalized = self._normalize_record(record, ticker=ticker)
            return OHLCVRecord.validate_security(normalized)

        except SecurityValidationError as e:
            self.log.warning(f"Security validation failed for OHLCV record: {e.message}")
            return None
        except Exception as e:
            self.log.warning(f"Validation failed for OHLCV record: {e}")
            return None

    def validate_ohlcv_batch(
        self, records: List[Dict[str, Any]], ticker: Optional[str] = None
    ) -> Tuple[List[OHLCVRecord], DataQualityMetrics]:
        metrics = DataQualityMetrics(total_records=len(records))
        try:
            validate_input_data({"records": records, "ticker": ticker}, strict=True)
        except SecurityValidationError as e:
            self.log.warning(f"Security validation failed for OHLCV batch: {e.message}")
            metrics.add_error("Batch security validation failed")
            return [], metrics

        validated_records: List[OHLCVRecord] = []
        self.log.info(f"Validating {len(records)} records" + (f" for {ticker}" if ticker else ""))

        for i, record in enumerate(records):
            try:
                validated_record = self.validate_ohlcv_record(record, ticker=ticker)

                if validated_record:
                    validated_records.append(validated_record)
                    metrics.valid_records += 1
                else:
                    metrics.add_error(f"Record {i}: Validation failed")

            except SecurityValidationError as e:
                error_msg = f"Record {i}: Security validation failed - {e.message}"
                metrics.add_error(error_msg)
                self.log.warning(error_msg)
            except Exception as e:
                error_msg = f"Record {i}: {str(e)}"
                metrics.add_error(error_msg)
                self.log.info(error_msg)

        self.log.info(
            f"Validation complete: {metrics.valid_records}/{metrics.total_records} "
            f"records valid ({metrics.success_rate:.1f}%)"
        )

        return validated_records, metrics

    def _normalize_record(self, record: Dict[str, Any], *, ticker: Optional[str]) -> Dict[str, Any]:
        if _is_polygon_record(record):
            return self._normalize_polygon_record(record, ticker=ticker)

        d = dict(record)
        if ticker and "ticker" not in d:
            d["ticker"] = ticker
        return d

    def _normalize_polygon_record(
        self, r: Dict[str, Any], *, ticker: Optional[str]
    ) -> Dict[str, Any]:
        validate_input_data(r, strict=False)

        ts_raw = r.get("t")
        ts = _polygon_timestamp(ts_raw) if ts_raw else None

        raw_ticker = (r.get("T") or ticker or "").strip()
        tkr = _validated_ticker(raw_ticker, "polygon_ticker") if raw_ticker else ""

        d: Dict[str, Any] = {
            "ticker": tkr,
            "timestamp": ts,
            "open": _num_or_zero(
                r.get("o", 0), field="polygon_open", kind="open price", log=self.log
            ),
            "high": _num_or_zero(
                r.get("h", 0), field="polygon_high", kind="high price", log=self.log
            ),
            "low": _num_or_zero(r.get("l", 0), field="polygon_low", kind="low price", log=self.log),
            "close": _num_or_zero(
                r.get("c", 0), field="polygon_close", kind="close price", log=self.log
            ),
            "volume": _int_or_zero(
                r.get("v", 0), field="polygon_volume", kind="volume", log=self.log
            ),
        }

        vwap = r.get("vw")
        if vwap is not None:
            try:
                SecureNumeric.validate_positive_number(vwap, "polygon_vwap")
                d["vwap"] = float(vwap)
            except SecurityValidationError:
                self.log.warning(f"Invalid VWAP in API response: {vwap}")

        return {k: v for k, v in d.items() if v is not None}

    # Backwards-compatible alias for older internal call sites
    def _transform_polygon_record(
        self, polygon_record: Dict[str, Any], ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._normalize_polygon_record(polygon_record, ticker=ticker)

    def validate_ticker_info(self, ticker_data: Dict[str, Any]) -> Optional[TickerInfo]:
        try:
            validate_input_data(ticker_data, strict=True)

            if "ticker" in ticker_data:
                transformed_data = {
                    "ticker": ticker_data["ticker"],
                    "name": ticker_data.get("name", ""),
                    "market": ticker_data.get("market", "stocks"),
                    "locale": ticker_data.get("locale", "us"),
                    "primary_exchange": ticker_data.get("primary_exchange"),
                    "currency_name": ticker_data.get("currency_name"),
                    "active": ticker_data.get("active", True),
                }
            else:
                transformed_data = ticker_data
            if "ticker" in transformed_data:
                transformed_data["ticker"] = ValidationUtils.validate_ticker_symbol(
                    transformed_data["ticker"], "ticker"
                )
            if "name" in transformed_data and transformed_data["name"]:
                transformed_data["name"] = str(transformed_data["name"]).strip()

            return TickerInfo(**transformed_data)

        except SecurityValidationError as e:
            self.log.warning(f"Security validation failed for ticker info: {e.message}")
            return None
        except Exception as e:
            self.log.warning(f"Ticker validation failed: {e}")
            return None

    @staticmethod
    def calculate_batch_vwap(records: List[Dict[str, Any]]) -> float:
        if not records:
            return 0.0

        total_pv = 0.0
        total_v = 0
        for r in records:
            h = r.get("high", 0) or 0
            low = r.get("low", 0) or 0
            close = r.get("close", 0) or 0
            v = r.get("volume", 0) or 0
            if v > 0 and h > 0 and low > 0 and close > 0:
                total_pv += ((h + low + close) / 3) * v
                total_v += v

        return 0.0 if total_v == 0 else round(total_pv / total_v, 4)
