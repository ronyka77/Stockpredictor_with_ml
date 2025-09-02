from src.utils import logger as logger_mod


def test_get_logger_creates_file(tmp_path):
    # Temporarily override LOGS_BASE_DIR to tmp_path to avoid touching repo logs
    old = logger_mod.LOGS_BASE_DIR
    logger_mod.LOGS_BASE_DIR = tmp_path

    try:
        lg = logger_mod.get_logger("tests.unit.utils.test_logger", utility="testutil")
        # Loguru-bound logger exposes an `info` method; ensure API available
        assert hasattr(lg, "info")

        # write a log and ensure file was created
        lg.info("hello world - test")
        # flush queued messages by shutting down sinks and then check files
        logger_mod.shutdown_logging()
        files = list((tmp_path / "testutil").glob("*.log"))
        assert len(files) >= 1
    finally:
        # restore
        logger_mod.LOGS_BASE_DIR = old


