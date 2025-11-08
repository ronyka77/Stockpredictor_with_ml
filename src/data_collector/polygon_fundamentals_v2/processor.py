from typing import Dict, List

from src.data_collector.polygon_fundamentals_v2.collector_service import (
    FundamentalsCollectorService,
)
from src.database.connection import get_global_pool, fetch_all
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FundamentalsProcessor:
    """Processes fundamentals for active tickers, sequentially (safe for rate limits)."""

    def __init__(self) -> None:
        self.service = FundamentalsCollectorService()
        self.pool = get_global_pool()

    def _get_active_tickers(self) -> List[str]:
        try:
            rows = fetch_all("SELECT ticker FROM tickers WHERE active = true and type = 'CS'")
            return [r["ticker"] for r in (rows or [])]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load active tickers: {e}")
            return []

    async def process_all(self) -> Dict[str, bool]:
        """
        Process fundamentals data for all active tickers.

        Iterates through all active tickers, fetching and processing their
        fundamentals data through the collector service. Returns a dictionary
        mapping ticker symbols to success status.

        Returns:
            Dict mapping ticker symbols to boolean success status
        """
        results: Dict[str, bool] = {}
        tickers = self._get_active_tickers()
        for i, t in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Processing {t}")
            try:
                res = await self.service.process_ticker(t)
                results[t] = bool(res.get("success"))
            except Exception as e:  # noqa: BLE001
                logger.error(f"Exception while processing {t}: {e}")
                results[t] = False
        return results
