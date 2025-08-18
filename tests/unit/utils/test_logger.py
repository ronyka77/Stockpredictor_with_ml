import logging

from src.utils import logger as logger_mod


def test_setup_logging_config_and_get_logger(tmp_path):
    # Temporarily override LOGS_BASE_DIR to tmp_path to avoid touching repo logs
    old = logger_mod.LOGS_BASE_DIR
    logger_mod.LOGS_BASE_DIR = tmp_path

    try:
        cfg = logger_mod.setup_logging_config(utility="testutil")
        assert 'handlers' in cfg

        lg = logger_mod.get_logger("tests.unit.utils.test_logger", utility="testutil")
        assert isinstance(lg, logging.Logger)

        # write a log and ensure file was created
        lg.info("hello world - test")
        # find log files
        files = list((tmp_path / "testutil").glob("*.log"))
        assert len(files) >= 1
    finally:
        # restore
        logger_mod.LOGS_BASE_DIR = old


