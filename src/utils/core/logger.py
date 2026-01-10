"""Centralized Logging System for StockPredictor V1.

This module provides a unified logging configuration that can be imported
and used across all modules in the project.

Refactor notes:
- Utility folders and utility-based file routing have been removed.
- A single timestamped log file is created per process run under /logs.
- Stdlib logging is intercepted and routed to Loguru.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import functools
import time
import asyncio

from loguru import logger as _loguru_logger

LOGS_BASE_DIR = Path(__file__).parent.parent.parent.parent / "logs"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class InterceptHandler(logging.Handler):
    """Intercept stdlib logging and forward it to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru_logger.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # ASCII-only logging to avoid UnicodeEncodeError issues downstream.
        try:
            message = record.getMessage()
            message = str(message).encode("ascii", errors="replace").decode("ascii")
        except Exception:
            message = str(record.getMessage())

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def _ensure_logs_dir() -> None:
    LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = LOGS_BASE_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def _unique_path(path: Path) -> Path:
    """Avoid collisions if multiple runs start within the same second."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix or ".log"
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


class _Sinks:
    """Internal sink lifecycle manager (single file + console)."""

    def __init__(self) -> None:
        self.initialized = False
        self.console_id: Optional[int] = None
        self.file_id: Optional[int] = None

    def ensure(self) -> None:
        if self.initialized:
            return

        _ensure_logs_dir()

        # Remove any pre-existing sinks configured elsewhere.
        _loguru_logger.remove()

        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message} | {function}:{line}"

        self.console_id = _loguru_logger.add(sys.stdout, level="INFO", enqueue=True, format=fmt)

        log_file = _unique_path(LOGS_BASE_DIR / f"run_{RUN_TIMESTAMP}.log")
        self.file_id = _loguru_logger.add(
            str(log_file),
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            enqueue=False,
            format=fmt,
        )

        # Intercept stdlib logging.
        logging.basicConfig(handlers=[InterceptHandler()], level=0)

        self.initialized = True

    def shutdown(self) -> None:
        # Remove file sink.
        try:
            if self.file_id is not None:
                _loguru_logger.remove(self.file_id)
        except Exception:
            sys.stderr.write("Error removing file sink\n")
        finally:
            self.file_id = None

        # Remove console sink.
        try:
            if self.console_id is not None:
                _loguru_logger.remove(self.console_id)
        except Exception:
            sys.stderr.write("Error removing console sink\n")
        finally:
            self.console_id = None

        self.initialized = False


_SINKS = _Sinks()


def init_logging_structure() -> None:
    """Ensure the logs directory exists."""
    _ensure_logs_dir()


def get_logger(name: str, utility: Optional[str] = None):
    """Return a Loguru-bound logger compatible with previous get_logger API.

    The returned object exposes `.info`, `.warning`, `.error`, `.debug`, and other
    Loguru methods via the bound logger.

    Note: `utility` is accepted for backwards compatibility but is ignored.
    """
    _SINKS.ensure()
    return _loguru_logger.bind(name=name)


def timeit(func):
    """Decorator that measures execution time of the wrapped function and logs it in seconds.

    Works with both synchronous and asynchronous functions.
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def _async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                # ASCII-only log message
                _loguru_logger.info(f"Execution time for {func.__name__}: {elapsed:.6f} seconds")
        return _async_wrapper

    @functools.wraps(func)
    def _sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            # ASCII-only log message
            _loguru_logger.info(f"Execution time for {func.__name__}: {elapsed:.6f} seconds")
    return _sync_wrapper


def shutdown_logging() -> None:
    """Remove all Loguru sinks to flush queued messages. Safe to call multiple times."""
    _SINKS.shutdown()


def merge_run_logs(
    run_timestamp: str = None, delete_originals: bool = True
) -> Optional[Path | list[Path]]:
    """Merge multiple log files with the same RUN_TIMESTAMP into one.

    This function finds all numbered variants of a log file (e.g., run_20260110_143000_1.log,
    run_20260110_143000_2.log) and appends them to the base file in order.

    Args:
        run_timestamp: The timestamp string (e.g., '20260110_143000'). If None, processes all timestamps.
        delete_originals: Whether to delete the numbered variants after merging

    Returns:
        Path to the merged log file, or None if only one file exists.
        If run_timestamp is None, returns list of all merged file paths.
    """
    if run_timestamp is None:
        # Process all timestamps in the logs folder
        return merge_all_run_logs(delete_originals)

    base_file = LOGS_BASE_DIR / f"run_{run_timestamp}.log"

    # Find all variants: run_20260110_143000_1.log, run_20260110_143000_2.log, etc.
    variants = sorted(
        LOGS_BASE_DIR.glob(f"run_{run_timestamp}_*.log"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )

    if not variants:
        _loguru_logger.info(f"No additional log files found for timestamp {run_timestamp}")
        return None

    _loguru_logger.info(f"Merging {len(variants)} log file(s) into {base_file.name}")

    # Append each variant to the base file
    with open(base_file, "a", encoding="utf-8") as merged:
        merged.write(f"\n{'=' * 80}\n")
        merged.write(f"MERGED LOG FILES: {len(variants)} additional file(s)\n")
        merged.write(f"{'=' * 80}\n\n")

        for variant_file in variants:
            merged.write(f"\n--- Content from {variant_file.name} ---\n\n")
            with open(variant_file, "r", encoding="utf-8") as src:
                merged.write(src.read())

            if delete_originals:
                variant_file.unlink()
                _loguru_logger.info(f"Deleted {variant_file.name}")

    _loguru_logger.info(f"Merge complete: {base_file}")
    return base_file


def merge_all_run_logs(delete_originals: bool = True) -> list[Path]:
    """Internal function to merge all run logs with variants in the logs folder.

    Args:
        delete_originals: Whether to delete the numbered variants after merging

    Returns:
        List of paths to all merged log files
    """
    import re

    # Pattern to extract timestamp from log files: run_20260110_143000.log or run_20260110_143000_1.log
    pattern = re.compile(r"^run_(\d{8}_\d{6})(?:_\d+)?\.log$")

    # Collect all unique timestamps
    timestamps = set()
    for log_file in LOGS_BASE_DIR.glob("run_*.log"):
        match = pattern.match(log_file.name)
        if match:
            timestamps.add(match.group(1))

    if not timestamps:
        _loguru_logger.info("No run log files found in logs folder")
        return []

    _loguru_logger.info(f"Found {len(timestamps)} unique timestamp(s) to check for merging")

    merged_files = []
    for timestamp in sorted(timestamps):
        result = merge_run_logs(timestamp, delete_originals)
        if result:
            merged_files.append(result)

    _loguru_logger.info(f"Completed merging. Total merged: {len(merged_files)} timestamp(s)")
    return merged_files


init_logging_structure()
atexit.register(shutdown_logging)


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Removing log files under 2KB in 'logs' folder.")

    merge_all_run_logs(delete_originals=True)

    num_removed = 0
    for dirpath, _, filenames in os.walk(LOGS_BASE_DIR):
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
