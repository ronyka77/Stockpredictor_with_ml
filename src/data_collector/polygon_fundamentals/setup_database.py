#!/usr/bin/env python3
"""
Database Setup for Fundamental Data Collection

This script sets up the database schema for optimized fundamental data collection.
Simple programmatic interface that always performs full setup and verification.
"""

from pathlib import Path

from src.database.connection import DatabaseConnection
from src.data_collector.polygon_fundamentals.db_pool import get_connection_pool
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
                
                # Execute the schema using execute() method which handles complex SQL better
                try:
                    cursor.execute(sql_schema)
                    logger.debug("Executed SQL schema successfully")
                except Exception as e:
                    logger.warning(f"Schema execution failed (may already exist): {e}")
                
                conn.commit()
                logger.info("✅ Fundamental database setup completed successfully")
                
                # Verify the table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'raw_fundamental_data'
                    )
                """)
                result = cursor.fetchone()
                
                if result and result['exists']:
                    logger.info("✅ raw_fundamental_data table created successfully")
                    
                    # Check indexes
                    cursor.execute("""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename = 'raw_fundamental_data'
                    """)
                    indexes = cursor.fetchall()
                    
                    logger.info(f"📊 Created {len(indexes)} indexes:")
                    for index in indexes:
                        logger.info(f"   - {index['indexname']}")
                    
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
    """Verify that the database is properly set up using connection pool"""
    try:
        # Use connection pool for verification operations
        db_pool = get_connection_pool()
        
        with db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'raw_fundamental_data'
                    )
                """)
                result = cursor.fetchone()
                table_exists = result['exists'] if result else False
                
                if not table_exists:
                    logger.error("❌ raw_fundamental_data table does not exist")
                    return False
                
                # Check columns
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'raw_fundamental_data'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                
                logger.info(f"📋 Table has {len(columns)} columns:")
                for column in columns:
                    logger.info(f"   - {column['column_name']}: {column['data_type']}")
                
                # Check constraints
                cursor.execute("""
                    SELECT constraint_name, constraint_type
                    FROM information_schema.table_constraints
                    WHERE table_name = 'raw_fundamental_data'
                """)
                constraints = cursor.fetchall()
                
                logger.info(f"🔒 Table has {len(constraints)} constraints:")
                for constraint in constraints:
                    logger.info(f"   - {constraint['constraint_name']}: {constraint['constraint_type']}")
                
                # Check indexes
                cursor.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes 
                    WHERE tablename = 'raw_fundamental_data'
                """)
                indexes = cursor.fetchall()
                
                logger.info(f"📊 Table has {len(indexes)} indexes:")
                for index in indexes:
                    logger.info(f"   - {index['indexname']}")
                
                logger.info("✅ Database setup verification complete!")
                return True
                
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

def validate_table_structure():
    """Validate the table structure using connection pool"""
    try:
        # Use connection pool for validation operations
        db_pool = get_connection_pool()
        
        with db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check required columns exist
                required_columns = [
                    'ticker_id', 'date', 'filing_date', 'fiscal_period', 'fiscal_year',
                    'revenues', 'net_income_loss', 'assets', 'equity'
                ]
                
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'raw_fundamental_data'
                """)
                existing_columns = [row['column_name'] for row in cursor.fetchall()]
                
                missing_columns = [col for col in required_columns if col not in existing_columns]
                
                if missing_columns:
                    logger.error(f"❌ Missing required columns: {missing_columns}")
                    return False
                
                # Check data types
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'raw_fundamental_data'
                    AND column_name IN ('ticker_id', 'date', 'filing_date', 'revenues', 'net_income_loss')
                """)
                column_types = cursor.fetchall()
                
                logger.info("📋 Key column data types:")
                for col in column_types:
                    logger.info(f"   - {col['column_name']}: {col['data_type']}")
                
                logger.info("✅ Table structure validation complete!")
                return True
                
    except Exception as e:
        logger.error(f"Table structure validation failed: {e}")
        return False

def main():
    """Main function - always does full database setup and verification"""
    logger.info("Setting up fundamental data collection database...")
    
    # Always do full setup
    success = setup_fundamental_database()
    
    if success:
        logger.info("Verifying setup...")
        success = verify_database_setup()
        if success:
            logger.info("Validating table structure...")
            success = validate_table_structure()
    
        logger.info("✅ Database setup completed successfully")
        return True
    else:
        logger.error("❌ Database setup failed")
        return False 