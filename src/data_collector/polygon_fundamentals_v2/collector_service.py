from typing import Any, Dict, Optional

from src.data_collector.polygon_fundamentals.cache_manager import (
    FundamentalCacheManager,
)
from src.data_collector.polygon_fundamentals_v2.raw_client import (
    RawPolygonFundamentalsClient,
)
from src.data_collector.polygon_fundamentals_v2.parser import FundamentalsParser
from src.data_collector.polygon_fundamentals_v2.validator_v2 import (
    FundamentalDataValidatorV2,
)
from src.data_collector.polygon_fundamentals_v2.repository import FundamentalsRepository
from src.data_collector.polygon_fundamentals_v2.extractor import FundamentalsExtractor
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FundamentalsCollectorService:
    """
    Orchestrates cache → raw client → parser → validator → repository.
    """

    def __init__(self) -> None:
        self.cache = FundamentalCacheManager()
        self.parser = FundamentalsParser()
        self.validator = FundamentalDataValidatorV2()
        self.repo = FundamentalsRepository()
        self.extractor = FundamentalsExtractor()

    async def process_ticker(self, ticker: str) -> Dict[str, Any]:
        self.repo.ensure_schema()
        ticker_id = self.repo.get_ticker_id(ticker)
        if ticker_id is None:
            logger.warning(f"Ticker not found in DB: {ticker}")
            return {"ticker": ticker, "success": False, "error": "unknown_ticker"}

        # cache-first
        raw: Optional[Dict[str, Any]] = self.cache.get_cached_data(ticker)
        used_cache = False
        if not raw:
            async with RawPolygonFundamentalsClient() as client:
                raw = await client.get_financials_raw(ticker)
                self.cache.save_cache(ticker, raw)
        else:
            used_cache = True

        if not raw or not raw.get("results"):
            if ticker_id is not None:
                self.repo.set_ticker_financials(
                    ticker_id=ticker_id, has_financials=False
                )
            return {"ticker": ticker, "success": False, "error": "no_results"}

        # persist raw payload (first result metadata anchors the period)
        first = raw["results"][0]
        self.repo.upsert_raw_payload(
            ticker_id=ticker_id,
            period_end=first.get("end_date"),
            timeframe=first.get("timeframe"),
            fiscal_period=first.get("fiscal_period"),
            fiscal_year=first.get("fiscal_year"),
            filing_date=first.get("filing_date"),
            source="cache" if used_cache else "api",
            payload=raw,
        )

        # extract and upsert structured facts
        try:
            self.extractor.extract_from_payload(ticker, raw)
        except Exception as ee:  # noqa: BLE001
            logger.warning(f"Extraction step failed for {ticker}: {ee}")

        # parse + validate
        dto = self.parser.parse(raw, ticker)
        v = self.validator.validate(dto)

        return {
            "ticker": ticker,
            "success": v.is_valid,
            "quality_score": v.quality_score,
            "warnings": sorted(set(v.base.warnings + v.cross_warnings)),
            "errors": sorted(set(v.base.errors + v.cross_errors)),
            "statement_counts": {
                "income": len(dto.income_statements),
                "balance": len(dto.balance_sheets),
                "cash": len(dto.cash_flow_statements),
            },
        }
