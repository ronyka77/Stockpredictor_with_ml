"""
Centralized Logging System for StockPredictor V1

This module provides a unified logging configuration that can be imported
and used across all modules in the project.

Usage:
    from src.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("This is an info message")
    
    # For specific utilities
    logger = get_logger(__name__, utility="polygon")
    logger = get_logger(__name__, utility="predictor")
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional
from datetime import datetime

# Base logs directory
LOGS_BASE_DIR = Path(__file__).parent / "logs"

class MinuteFormatter(logging.Formatter):
    """Custom formatter that shows time only to the minute"""
    
    def formatTime(self, record, datefmt=None):
        """Override formatTime to show only to the minute"""
        ct = self.converter(record.created)
        if datefmt:
            s = datetime(*ct[:6]).strftime(datefmt)
        else:
            # Default format: YYYY-MM-DD HH:MM
            s = datetime(*ct[:6]).strftime('%Y-%m-%d %H:%M')
        return s

def setup_logging_config(utility: str = "general") -> dict:
    """
    Setup logging configuration for a specific utility
    
    Args:
        utility: Name of the utility (e.g., 'polygon', 'predictor', 'general')
        
    Returns:
        Logging configuration dictionary
    """
    # Create utility-specific log directory
    log_dir = LOGS_BASE_DIR / utility
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log file names with current date
    today = datetime.now().strftime('%Y%m%d')
    info_log_file = log_dir / f"{utility}_{today}.log"
    debug_log_file = log_dir / f"{utility}_debug_{today}.log"
    error_log_file = log_dir / f"{utility}_error_{today}.log"
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                '()': MinuteFormatter,
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                '()': MinuteFormatter,
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            },
            'console': {
                '()': MinuteFormatter,
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'formatter': 'console',
                'class': 'logging.StreamHandler',
            },
            'info_file': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': str(info_log_file),
                'mode': 'a',
                'encoding': 'utf-8',
            },
            'debug_file': {
                'level': 'DEBUG',
                'formatter': 'detailed',
                'class': 'logging.FileHandler',
                'filename': str(debug_log_file),
                'mode': 'a',
                'encoding': 'utf-8',
            },
            'error_file': {
                'level': 'ERROR',
                'formatter': 'detailed',
                'class': 'logging.FileHandler',
                'filename': str(error_log_file),
                'mode': 'a',
                'encoding': 'utf-8',
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'info_file', 'debug_file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    return config

def get_logger(name: str, utility: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name (typically __name__)
        utility: Utility name for organizing logs (auto-detected if None)
        
    Returns:
        Configured logger instance
    """
    # Auto-detect utility from module name if not provided
    if utility is None:
        if 'polygon' in name:
            utility = 'polygon'
        elif 'predictor' in name:
            utility = 'predictor'
        elif 'data_collector' in name:
            utility = 'data_collector'
        elif 'feature_engineering' in name:
            utility = 'feature_engineering'
        else:
            utility = 'general'
    
    # Setup logging configuration for this utility
    config = setup_logging_config(utility)
    
    # Configure logging (this is safe to call multiple times)
    logging.config.dictConfig(config)
    
    # Return logger
    return logging.getLogger(name)

def get_polygon_logger(name: str) -> logging.Logger:
    """Convenience function for polygon-specific logging"""
    return get_logger(name, utility='polygon')

def get_predictor_logger(name: str) -> logging.Logger:
    """Convenience function for predictor-specific logging"""
    return get_logger(name, utility='predictor')

def get_general_logger(name: str) -> logging.Logger:
    """Convenience function for general logging"""
    return get_logger(name, utility='general')

def cleanup_old_logs(utility: str, days_to_keep: int = 30) -> None:
    """
    Clean up old log files
    
    Args:
        utility: Utility name
        days_to_keep: Number of days of logs to keep
    """
    log_dir = LOGS_BASE_DIR / utility
    if not log_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    
    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
            except OSError as e:
                print(f"Failed to delete {log_file}: {e}")

# Initialize logs directory structure
def init_logging_structure():
    """Initialize the logging directory structure"""
    utilities = ['polygon', 'predictor', 'data_collector', 'feature_engineering', 'general']
    
    for utility in utilities:
        log_dir = LOGS_BASE_DIR / utility
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a .gitkeep file to ensure directories are tracked
        gitkeep = log_dir / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()

# Initialize on import
init_logging_structure()

# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    logger = get_logger(__name__)
    logger.info("Testing centralized logging system")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test polygon-specific logging
    polygon_logger = get_polygon_logger("test.polygon.module")
    polygon_logger.info("Testing polygon-specific logging")
    
    print(f"Logs are being written to: {LOGS_BASE_DIR}")
    print("Check the following directories:")
    for utility_dir in LOGS_BASE_DIR.iterdir():
        if utility_dir.is_dir():
            print(f"  - {utility_dir}")
            for log_file in utility_dir.glob("*.log"):
                print(f"    * {log_file.name}") 