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

# Base logs directory - point to central logs folder at project root
LOGS_BASE_DIR = Path(__file__).parent.parent.parent / "logs"


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
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            },
            'console': {
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
            },
            # Silence very chatty third-party libraries that log DEBUG by default
            'matplotlib': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False
            },
            'graphviz': {
                'handlers': ['console'],
                'level': 'WARNING',
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
        elif 'mlp' in name:
            utility = 'mlp'
        elif 'lstm' in name:
            utility = 'lstm'
        elif 'lightgbm' in name:
            utility = 'lightgbm'
        elif 'xgboost' in name:
            utility = 'xgboost'
        elif 'catboost' in name:
            utility = 'catboost'
        elif 'random_forest' in name:
            utility = 'random_forest'
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

def cleanup_old_logs(days_to_keep: int = 30, min_size_kb: int = 2) -> None:
    """
    Clean up old log files and small log files
    
    Args:
        utility: Utility name
        days_to_keep: Number of days of logs to keep
    """
    log_dir = LOGS_BASE_DIR 
    if not log_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    min_size_bytes = min_size_kb * 1024
    
    # Recursively check all subdirectories
    for log_file in log_dir.rglob("*.log"):
        try:
            file_stat = log_file.stat()
            # Delete old files or very small files
            if file_stat.st_mtime < cutoff_time or file_stat.st_size < min_size_bytes:
                log_file.unlink()
                logger.info(f"Deleted log file: {log_file} (size: {file_stat.st_size} bytes)")
        except OSError as e:
            logger.error(f"Failed to delete {log_file}: {e}")

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
    logger.info("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    cleanup_old_logs()