"""
Database Connection Pool for Fundamental Data Pipeline

This module provides a centralized connection pool for the fundamental data pipeline
to optimize database connection management.
"""

from src.database.connection import DatabaseConnectionPool
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global connection pool instance (will be lazily initialized)
_connection_pool = None

def get_connection_pool(min_connections: int = 2, max_connections: int = 10) -> DatabaseConnectionPool:
    """
    Get or create the connection pool singleton
    
    Args:
        min_connections: Minimum number of connections in the pool (default: 2)
        max_connections: Maximum number of connections in the pool (default: 10)
        
    Returns:
        DatabaseConnectionPool: Singleton connection pool instance
    """
    global _connection_pool
    
    if _connection_pool is None:
        logger.info(f"Initializing fundamental data connection pool ({min_connections}-{max_connections} connections)")
        _connection_pool = DatabaseConnectionPool(
            min_connections=min_connections,
            max_connections=max_connections
        )
        
    return _connection_pool

def close_connection_pool():
    """Close the global connection pool if it exists"""
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.close()
        _connection_pool = None
        logger.info("Fundamental data connection pool closed") 