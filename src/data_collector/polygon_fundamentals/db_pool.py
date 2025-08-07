"""
Database Connection Pool for Fundamental Data Pipeline

This module provides a centralized connection pool for the fundamental data pipeline
to optimize database connection management.
"""

import threading
from src.database.connection import DatabaseConnectionPool
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global connection pool instance (will be lazily initialized)
_connection_pool = None
# Thread lock for singleton initialization
_pool_lock = threading.Lock()

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
    
    # Double-checked locking pattern for thread-safe singleton
    if _connection_pool is None:
        with _pool_lock:
            # Check again inside the lock
            if _connection_pool is None:
                logger.info(f"Initializing fundamental data connection pool ({min_connections}-{max_connections} connections)")
                _connection_pool = DatabaseConnectionPool(
                    min_connections=min_connections,
                    max_connections=max_connections
                )
                # Store the initialization parameters for comparison
                _connection_pool._init_min_connections = min_connections
                _connection_pool._init_max_connections = max_connections
    else:
        # Check if parameters have changed and reinitialize if needed
        with _pool_lock:
            current_min = getattr(_connection_pool, '_init_min_connections', 2)
            current_max = getattr(_connection_pool, '_init_max_connections', 10)
            
            if current_min != min_connections or current_max != max_connections:
                logger.info(f"Connection pool parameters changed from ({current_min}-{current_max}) to ({min_connections}-{max_connections}). Reinitializing...")
                
                # Close existing pool
                _connection_pool.close()
                
                # Create new pool with updated parameters
                _connection_pool = DatabaseConnectionPool(
                    min_connections=min_connections,
                    max_connections=max_connections
                )
                # Store the new initialization parameters
                _connection_pool._init_min_connections = min_connections
                _connection_pool._init_max_connections = max_connections
        
    return _connection_pool

def close_connection_pool():
    """Close the global connection pool if it exists"""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close()
            _connection_pool = None
            logger.info("Fundamental data connection pool closed") 