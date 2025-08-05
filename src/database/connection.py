"""
Database Connection Module

This module provides database connection functionality for the StockPredictor V1 system.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
import os

from src.utils.logger import get_logger

logger = get_logger(__name__, utility='database')

class DatabaseConnection:
    """
    Database connection manager for PostgreSQL
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database connection
        
        Args:
            config: Database configuration dictionary
        """
        if config:
            self.config = config
        else:
            # Default configuration from environment variables
            self.config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'stock_data'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
        
        logger.debug(f"Initialized DatabaseConnection for {self.config['host']}:{self.config['port']}/{self.config['database']}")
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a database connection as a context manager
        
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_connection_string(self) -> str:
        """
        Get a connection string for logging (without password)
        
        Returns:
            Connection string without sensitive information
        """
        return f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
    
    def test_connection(self) -> bool:
        """
        Test the database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


class DatabaseConnectionPool:
    """
    Database connection pool manager for PostgreSQL
    
    Uses ThreadedConnectionPool for thread-safe connection pooling to improve
    performance by reusing database connections rather than creating new ones
    for each operation.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, 
                 min_connections: int = 1, 
                 max_connections: int = 10):
        """
        Initialize database connection pool
        
        Args:
            config: Database configuration dictionary
            min_connections: Minimum number of connections in the pool
            max_connections: Maximum number of connections in the pool
        """
        if config:
            self.config = config
        else:
            # Default configuration from environment variables
            self.config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'stock_data'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
        
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool = None
        self._initialize_pool()
        
        logger.info(f"Initialized DatabaseConnectionPool ({min_connections}-{max_connections} connections) "
                    f"for {self.config['host']}:{self.config['port']}/{self.config['database']}")
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            logger.debug("Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a connection from the pool as a context manager
        
        Yields:
            Database connection from the pool
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database pool connection error: {str(e)}")
            raise
        finally:
            if conn:
                # Clean the connection before returning it to the pool
                if hasattr(conn, 'rollback') and callable(conn.rollback):
                    conn.rollback()
                self.pool.putconn(conn)
    
    def get_connection_string(self) -> str:
        """
        Get a connection string for logging (without password)
        
        Returns:
            Connection string without sensitive information
        """
        return f"{self.config['host']}:{self.config['port']}/{self.config['database']} (pooled)"
    
    def test_connection(self) -> bool:
        """
        Test a connection from the pool
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"Pool connection test failed: {str(e)}")
            return False
    
    def close(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Connection pool closed") 