#!/usr/bin/env python3
"""
Database Setup for Fundamental Data Collection

This script sets up the database schema for optimized fundamental data collection.
"""

import logging
import sys
from pathlib import Path

from src.database.connection import DatabaseConnection, DatabaseConnectionPool
from src.utils.logger import get_logger

logger = get_logger(__name__)

def setup_fundamental_database():
    """Set up database tables for fundamental data"""
    # For setup operations, it's better to use a direct connection rather than a pool
    conn = None
    try:
        db_connection = DatabaseConnection()
        with db_connection.get_connection() as conn:
            with conn.cursor() as cursor:
                # Read the SQL schema file
                sql_file = Path(__file__).parent.parent.parent.parent / "sql" / "fundamental_data_schema.sql"
                
                if not sql_file.exists():
                    logger.error(f"SQL schema file not found: {sql_file}")
                    return False
                
                with open(sql_file, 'r') as f:
                    sql_schema = f.read()
                
                # Execute the schema
                # Split and execute SQL statements
                statements = sql_schema.split(';')
                
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        try:
                            cursor.execute(statement)
                            logger.debug(f"Executed: {statement[:50]}...")
                        except Exception as e:
                            logger.warning(f"Statement failed (may already exist): {e}")
                            logger.debug(f"Failed statement: {statement}")
                
                conn.commit()
                logger.info("✅ Fundamental database setup completed successfully")
                
                # Verify the table exists
                result = cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'raw_fundamental_data'
                    )
                """).scalar()
                
                if result:
                    logger.info("✅ raw_fundamental_data table created successfully")
                    
                    # Check indexes
                    indexes = cursor.execute("""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename = 'raw_fundamental_data'
                    """).fetchall()
                    
                    logger.info(f"📊 Created {len(indexes)} indexes:")
                    for index in indexes:
                        logger.info(f"   - {index[0]}")
                    
                    return True
                else:
                    logger.error("❌ raw_fundamental_data table not found after setup")
                    return False
                
    except Exception as e:
        logger.error(f"❌ Failed to set up fundamental database: {e}")
        if conn:
            conn.rollback()
        return False

def verify_database_setup():
    """Verify that the database is properly set up"""
    try:
        db_connection = DatabaseConnection()
        
        with db_connection.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if table exists
                table_exists = cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'raw_fundamental_data'
                    )
                """).scalar()
                
                if not table_exists:
                    logger.error("❌ raw_fundamental_data table does not exist")
                    return False
                
                # Check columns
                columns = cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'raw_fundamental_data'
                    ORDER BY ordinal_position
                """).fetchall()
                
                logger.info(f"📋 Table has {len(columns)} columns:")
                for column in columns:
                    logger.info(f"   - {column[0]}: {column[1]}")
                
                # Check constraints
                constraints = cursor.execute("""
                    SELECT constraint_name, constraint_type
                    FROM information_schema.table_constraints
                    WHERE table_name = 'raw_fundamental_data'
                """).fetchall()
                
                logger.info(f"🔒 Table has {len(constraints)} constraints:")
                for constraint in constraints:
                    logger.info(f"   - {constraint[0]}: {constraint[1]}")
                
                # Check indexes
                indexes = cursor.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes 
                    WHERE tablename = 'raw_fundamental_data'
                """).fetchall()
                
                logger.info(f"📊 Table has {len(indexes)} indexes:")
                for index in indexes:
                    logger.info(f"   - {index[0]}")
                
                logger.info("✅ Database setup verification complete!")
                return True
                
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Database Setup for Fundamental Data Collection"
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing setup, do not create'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.verify_only:
        logger.info("Verifying database setup...")
        success = verify_database_setup()
    else:
        logger.info("Setting up fundamental data collection database...")
        success = setup_fundamental_database()
        
        if success:
            logger.info("Verifying setup...")
            success = verify_database_setup()
    
    if success:
        logger.info("🎉 Database setup/verification successful!")
        sys.exit(0)
    else:
        logger.error("❌ Database setup/verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    main() 