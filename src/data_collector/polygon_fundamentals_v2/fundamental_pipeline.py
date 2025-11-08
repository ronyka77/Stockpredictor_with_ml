import asyncio
from datetime import datetime
from typing import Any, Dict

from src.data_collector.polygon_fundamentals_v2.processor import FundamentalsProcessor
from src.utils.logger import get_logger


logger = get_logger(__name__)


async def main() -> Dict[str, Any]:
    logger.info("Starting Fundamentals V2 Pipeline")
    started = datetime.now()
    processor = FundamentalsProcessor()
    results = await processor.process_all()
    elapsed_s = (datetime.now() - started).total_seconds()
    total = len(results)
    ok = sum(1 for v in results.values() if v)
    logger.info(f"Fundamentals V2 complete: {ok}/{total} success in {elapsed_s / 60:.1f} minutes")
    return {"results": results, "success": ok, "total": total, "elapsed_s": elapsed_s}


if __name__ == "__main__":
    out = asyncio.run(main())
    logger.info(out)
