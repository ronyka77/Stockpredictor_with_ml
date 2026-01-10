"""
Runner to execute data, news and indicator pipelines and clear the cleaned data cache.

This module intentionally keeps a simple programmatic interface (no CLI parsing per
project guidelines) so it can be imported or executed as a script.
"""

from datetime import datetime
from typing import Dict, Any

from src.utils.core.logger import get_logger
from src.data_collector.polygon_data.data_pipeline import run_data_pipeline
from src.data_collector.polygon_news.news_pipeline import run_news_pipeline
from src.data_collector.indicator_pipeline.indicator_pipeline import run_indicator_pipeline
from src.utils.data.cleaned_data_cache import CleanedDataCache

logger = get_logger(__name__)


def run_all_pipelines() -> Dict[str, Any]:
    """Run data, news and indicator pipelines sequentially, then clear cache.

    Returns a dictionary with per-pipeline results and timings.
    """
    results: Dict[str, Any] = {}

    # 1) Run grouped daily data pipeline
    try:
        start = datetime.now()
        logger.info("Starting grouped daily data pipeline...")
        stats = run_data_pipeline(days_back=90, validate_data=True, save_stats=True)
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Grouped daily data pipeline finished in {duration:.1f}s")
        results["data_pipeline"] = {
            "success": True,
            "stats": getattr(stats, "to_dict", lambda: stats)(),
        }
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        results["data_pipeline"] = {"success": False, "error": str(e)}

    # 2) Run news pipeline (incremental/historical as implemented inside)
    try:
        start = datetime.now()
        logger.info("Starting news pipeline...")
        success = run_news_pipeline()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"News pipeline finished in {duration:.1f}s")
        results["news_pipeline"] = {"success": bool(success)}
    except Exception as e:
        logger.error(f"News pipeline failed: {e}")
        results["news_pipeline"] = {"success": False, "error": str(e)}

    # 3) Run indicator pipeline
    try:
        start = datetime.now()
        logger.info("Starting indicator pipeline...")
        ind_result = run_indicator_pipeline()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Indicator pipeline finished in {duration:.1f}s")
        results["indicator_pipeline"] = {"success": ind_result is not None, "result": ind_result}
    except Exception as e:
        logger.error(f"Indicator pipeline failed: {e}")
        results["indicator_pipeline"] = {"success": False, "error": str(e)}

    # 4) Clear cleaned data cache
    try:
        logger.info("Clearing cleaned data cache...")
        cache = CleanedDataCache()
        cache.clear_cache()
        results["clear_cache"] = {"success": True}
        logger.info("Cache cleared")
    except Exception as e:
        logger.error(f"Clearing cache failed: {e}")
        results["clear_cache"] = {"success": False, "error": str(e)}

    return results


def main() -> int:
    """Main entrypoint for running all pipelines. Returns exit code-like int.

    Per project policy, we don't parse CLI args in this module.
    """
    logger.info("Pipeline runner starting")
    results = run_all_pipelines()

    # Summarize
    successes = sum(1 for v in results.values() if v.get("success"))
    total = len(results)
    logger.info(f"Runner finished: {successes}/{total} succeeded")

    # Return non-zero on any failure to make script usable in automation
    return 0 if successes == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
