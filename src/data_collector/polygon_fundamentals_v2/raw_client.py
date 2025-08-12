from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import aiohttp

from src.data_collector.polygon_fundamentals.config import (
    PolygonFundamentalsConfig,
    polygon_fundamentals_config,
)
from src.data_collector.polygon_fundamentals.client import RateLimiter as BasicRateLimiter
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RawPolygonFundamentalsClient:
    """
    Minimal raw client that returns raw JSON for fundamentals endpoint.
    Keeps HTTP concerns isolated (session, retries, rate limiting).
    """

    def __init__(self, config: Optional[PolygonFundamentalsConfig] = None) -> None:
        self.config = config or polygon_fundamentals_config
        self.rate_limiter = BasicRateLimiter(self.config.REQUESTS_PER_MINUTE)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "RawPolygonFundamentalsClient":
        timeout = aiohttp.ClientTimeout(
            total=self.config.REQUEST_TIMEOUT, connect=self.config.CONNECTION_TIMEOUT
        )
        self.session = aiohttp.ClientSession(headers=self.config.headers, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session:
            await self.session.close()

    async def _get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        await self.rate_limiter.wait_if_needed()

        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status == 429:
                        backoff = self.config.RETRY_DELAY * (self.config.BACKOFF_FACTOR ** attempt)
                        logger.warning(f"429 rate limited. Sleeping {backoff:.2f}s (attempt {attempt+1})")
                        await asyncio.sleep(backoff)
                        continue
                    logger.error(f"HTTP {resp.status}: {await resp.text()}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt+1}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"HTTP failure attempt {attempt+1}: {e}")

            if attempt < self.config.RETRY_ATTEMPTS - 1:
                await asyncio.sleep(self.config.RETRY_DELAY * (self.config.BACKOFF_FACTOR ** attempt))

        return None

    async def get_financials_raw(self, ticker: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        url = self.config.get_financials_url(ticker)
        params = self.config.get_request_params(ticker, **kwargs)
        logger.info(f"Fetching fundamentals (raw) for {ticker}")
        return await self._get(url, params)


