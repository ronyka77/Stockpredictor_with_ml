# Backward compatibility module for logger
# Re-exports from core.logger for backward compatibility

from .core.logger import get_logger, init_logging_structure, shutdown_logging

__all__ = ["get_logger", "init_logging_structure", "shutdown_logging"]
