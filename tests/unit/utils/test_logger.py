from pathlib import Path

from src.utils import logger as logger_mod


def test_get_logger_creates_file(tmp_path: Path) -> None:
    """Ensure the central Loguru-based logger creates a file sink for a utility.

    The test temporarily points the logger's base logs directory at `tmp_path`,
    obtains a bound logger, writes a message, shuts down sinks to flush queued
    messages, and asserts that at least one log file was created.
    """
    old_base = logger_mod.LOGS_BASE_DIR
    logger_mod.LOGS_BASE_DIR = tmp_path

    try:
        lg = logger_mod.get_logger(__name__, utility="testutil")
        assert hasattr(lg, "info"), "bound logger is missing 'info' method"

        # Emit a log and flush queued sinks
        lg.info("unit test log entry")
        logger_mod.shutdown_logging()

        # Check for created log file (logger names files with utility prefix)
        util_dir = tmp_path / "testutil"
        files = list(util_dir.glob("testutil_*.log"))
        assert files, f"no log files were created in {util_dir}"
    finally:
        # restore module state
        logger_mod.LOGS_BASE_DIR = old_base
