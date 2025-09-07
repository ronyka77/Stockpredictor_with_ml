"""
Database Package

This package provides database connectivity and management functionality
for the StockPredictor V1 system.
"""

from src.database.connection import (
    PostgresConnection,
    init_global_pool,
    get_global_pool,
    close_global_pool,
    fetch_all,
    fetch_one,
    execute,
    execute_values,
    run_in_transaction,
)

__all__ = [
    "PostgresConnection",
    "init_global_pool",
    "get_global_pool",
    "close_global_pool",
    "fetch_all",
    "fetch_one",
    "execute",
    "execute_values",
    "run_in_transaction",
]
