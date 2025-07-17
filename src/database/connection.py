"""
Database Connection Module

This module provides database connection functionality for the StockPredictor V1 system.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
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
            self.test_connection()
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