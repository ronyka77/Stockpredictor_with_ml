"""
Fundamentals V2: Cleanly layered pipeline (client → parser → validator → repository → service).

This package does not modify existing V1 modules. It can run side-by-side.
"""

from .raw_client import RawPolygonFundamentalsClient
from .parser import FundamentalsParser
from .validator_v2 import FundamentalDataValidatorV2
from .repository import FundamentalsRepository
from .collector_service import FundamentalsCollectorService
from .processor import FundamentalsProcessor

__all__ = [
    "RawPolygonFundamentalsClient",
    "FundamentalsParser",
    "FundamentalDataValidatorV2",
    "FundamentalsRepository",
    "FundamentalsCollectorService",
    "FundamentalsProcessor",
]


