from pathlib import Path

from src.utils.core import logger as logger_mod


def test_get_logger_creates_file(tmp_path: Path) -> None:
    """Ensure the central logger writes a log file sink for a bound utility logger."""
    old_base = logger_mod.LOGS_BASE_DIR
    logger_mod.LOGS_BASE_DIR = tmp_path

    try:
        lg = logger_mod.get_logger(__name__, utility="testutil")
        if not hasattr(lg, "info"):
            raise AssertionError("bound logger is missing 'info' method")

        # Emit a log and flush queued sinks
        lg.info("unit test log entry")
        logger_mod.shutdown_logging()

        # Check for created log file (logger names files with utility prefix)
        util_dir = tmp_path / "testutil"
        files = list(util_dir.glob("testutil_*.log"))
        if not files:
            raise AssertionError(f"no log files were created in {util_dir}")
    finally:
        # restore module state
        logger_mod.LOGS_BASE_DIR = old_base
