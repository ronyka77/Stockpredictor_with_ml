"""
Centralized Logging System for StockPredictor V1
This module provides a unified logging configuration that can be imported
and used across all modules in the project.
"""

import logging
import atexit
import sys
import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from loguru import logger as _loguru_logger

# Run identifier for this process: combines process id and import-time timestamp.
# Using a module-level RUN_ID ensures each process run produces at most one
# log file per utility (the sink registry `_file_sink_ids` already prevents
# adding the same utility twice in one process).
RUN_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _ascii_patch(record: dict) -> None:
    """Mutate the record to ensure the message is ASCII-only before sinks write it.

    This will replace non-ASCII characters with the standard replacement char,
    preventing encoding issues in environments that require ASCII-only logs.
    """
    try:
        original = record.get("message", "")
        text = str(original)
        record["message"] = text.encode("ascii", errors="replace").decode("ascii")
    except Exception:
        # Fallback: ensure there's at least a string message.
        record["message"] = str(record.get("message", ""))


# Base logs directory - point to central logs folder at project root
LOGS_BASE_DIR = Path(__file__).parent.parent.parent / "logs"


class InterceptHandler(logging.Handler):
    """Intercepts stdlib logging and routes it to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Retrieve corresponding Loguru level if it exists
        try:
            level = _loguru_logger.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno

        # Find caller from where logging was called
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        # Ensure ASCII-safe message to avoid encoding issues in downstream sinks.
        try:
            message = record.getMessage()
            message = str(message).encode("ascii", errors="replace").decode("ascii")
        except Exception:
            message = str(record.getMessage())

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def _ensure_utility_dir(utility: str) -> Path:
    path = LOGS_BASE_DIR / utility
    path.mkdir(parents=True, exist_ok=True)
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    return path


def get_logger(name: str, utility: Optional[str] = None):
    """Return a Loguru-bound logger compatible with previous get_logger API.
    The returned object exposes `.info`, `.warning`, `.error`, `.debug`, and
    other Loguru methods via the bound logger.
    """
    # Auto-detect utility from module name if not provided
    if utility is None:
        lower_name = name.lower()
        if "polygon" in lower_name:
            utility = "polygon"
        elif "predictor" in lower_name:
            utility = "predictor"
        elif "data_collector" in lower_name:
            utility = "data_collector"
        elif "feature_engineering" in lower_name:
            utility = "feature_engineering"
        elif "mlp" in lower_name:
            utility = "mlp"
        elif "lightgbm" in lower_name:
            utility = "lightgbm"
        elif "xgboost" in lower_name:
            utility = "xgboost"
        elif "random_forest" in lower_name:
            utility = "random_forest"
        else:
            utility = "general"

    # Lazily initialize global sinks on first get_logger call
    _initialize_sinks_once()

    # Return a bound logger with metadata similar to previous behavior
    return _loguru_logger.bind(name=name, utility=utility)


def init_logging_structure():
    """Initialize the logging directory structure"""
    utilities = ["polygon", "predictor", "data_collector", "feature_engineering", "general"]
    for u in utilities:
        _ensure_utility_dir(u)


def _initialize_sinks_once() -> None:
    """Initialize console sink and stdlib intercept once per process."""
    global _sinks_initialized, _console_sink_id
    if _sinks_initialized:
        return

    # Remove default handlers
    _loguru_logger.remove()

    # Console sink: write to stdout, non-blocking queue enabled
    console_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message} | {function}:{line}"
    _console_sink_id = _loguru_logger.add(
        sys.stdout, level="INFO", enqueue=True, format=console_format
    )

    # Intercept stdlib logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    _sinks_initialized = True


def _ensure_file_sink_for_utility(utility: str, util_dir: Path) -> None:
    """Add a file sink for the given utility if not already added for this process run.
    File name includes timestamp to avoid cross-process collisions.
    """
    global _file_sink_ids
    if utility in _file_sink_ids:
        return
    # Use a deterministic run identifier (timestamp + pid) created on module import
    # so each process run produces at most one file per utility.
    log_file = util_dir / f"{utility}_{RUN_ID}.log"

    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message} | {function}:{line}"
    sink_id = _loguru_logger.add(
        str(log_file),
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        format=file_format,
    )

    _file_sink_ids[utility] = sink_id


def shutdown_logging() -> None:
    """Remove all Loguru sinks to flush queued messages. Safe to call multiple times."""
    global _sinks_initialized, _console_sink_id, _file_sink_ids
    # Attempt to remove file sinks; log failures at debug level with exception info
    for sid in list(_file_sink_ids.values()):
        try:
            _loguru_logger.remove(sid)
        except Exception as e:
            try:
                logger.debug(f"Error removing file sink id={sid}", exc_info=True)
            except Exception:
                # If logger is not available or logging fails, fallback to stderr
                sys.stderr.write(f"Error removing file sink id={sid}: {e}\n")
    # Clear file sink registry regardless of removal success
    _file_sink_ids.clear()

    # Attempt to remove console sink; ensure console sink id is reset regardless
    try:
        if _console_sink_id is not None:
            _loguru_logger.remove(_console_sink_id)
    except Exception as e:
        try:
            logger.debug(f"Error removing console sink id={_console_sink_id}", exc_info=True)
        except Exception:
            sys.stderr.write(f"Error removing console sink id={_console_sink_id}: {e}\n")
    finally:
        _console_sink_id = None

    # Ensure initialized flag is reset so state is consistent
    _sinks_initialized = False


# Initialize on import
init_logging_structure()

# --- New: sink management for enqueue and shutdown ---
_sinks_initialized = False
_console_sink_id: Optional[int] = None
_file_sink_ids: Dict[str, int] = {}

# Register shutdown to flush sinks on graceful exit
atexit.register(shutdown_logging)


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Removing log files under 2KB in 'logs' folder.")

    logs_root = LOGS_BASE_DIR
    num_removed = 0
    for dirpath, _, filenames in os.walk(logs_root):
        for fname in filenames:
            file_path = os.path.join(dirpath, fname)
            try:
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    if size < 8096:
                        os.remove(file_path)
                        logger.info(f"Removed log file: {file_path} ({size} bytes)")
                        num_removed += 1
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
    logger.info(f"Completed cleanup. Removed {num_removed} files under 2KB.")
