#!/usr/bin/env python3
"""
Database Setup for Fundamental Data Collection

This script sets up the database schema for optimized fundamental data collection.
Simple programmatic interface that always performs full setup and verification.
"""

from pathlib import Path

from src.database.connection import init_global_pool, execute, fetch_one, fetch_all
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


def setup_fundamental_database() -> bool:
    """Set up database tables for fundamental data"""
    # For setup operations, prefer a pooled connection but allow direct use via pool
    conn = None
    try:
        # Ensure global pool is initialized (will validate credentials)
        init_global_pool()
        # Read the SQL schema file
        sql_file = Path(__file__).parent.parent.parent.parent / "sql" / "fundamentals_v2_schema.sql"

        if not sql_file.exists():
            logger.error(f"SQL schema file not found: {sql_file}")
            return False

        with open(sql_file, "r") as f:
            sql_schema = f.read()

        try:
            execute(sql_schema)
            logger.info("Executed SQL schema successfully")
        except Exception as e:
            logger.warning(f"Schema execution failed (may already exist): {e}")

        logger.info("Fundamental database setup completed successfully")

        # Verify the table exists using helpers
        result = fetch_one(
            """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'raw_fundamental_data'
                )
            """
        )

        if result and result.get("exists"):
            logger.info("raw_fundamental_data table created successfully")

            # Check indexes
            indexes = fetch_all(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'raw_fundamental_data'"
            )
            logger.info(f"Created {len(indexes)} indexes:")
            for index in indexes or []:
                logger.info(f"- {index['indexname']}")

            return True
        else:
            logger.error("raw_fundamental_data table not found after setup")
            return False

    except Exception as e:
        logger.error(f"Failed to set up fundamental database: {e}")
        if conn:
            conn.rollback()
        return False


def verify_database_setup() -> bool:
    """Verify that the database is properly set up using connection pool"""
    try:
        # Use helper functions for verification
        result = fetch_one(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema='public' AND table_name='raw_fundamental_data')"
        )
        table_exists = result.get("exists") if result else False

        if not table_exists:
            logger.error("raw_fundamental_data table does not exist")
            return False

        columns = fetch_all(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'raw_fundamental_data' ORDER BY ordinal_position"
        )
        logger.info(f"Table has {len(columns)} columns:")
        for column in columns or []:
            logger.info(f"- {column['column_name']}: {column['data_type']}")

        constraints = fetch_all(
            "SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name = 'raw_fundamental_data'"
        )
        logger.info(f"🔒 Table has {len(constraints)} constraints:")
        for constraint in constraints or []:
            logger.info(f"- {constraint['constraint_name']}: {constraint['constraint_type']}")

        indexes = fetch_all(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'raw_fundamental_data'"
        )
        logger.info(f"Table has {len(indexes)} indexes:")
        for index in indexes or []:
            logger.info(f"- {index['indexname']}")

        logger.info("Database setup verification complete!")
        return True

    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


def validate_table_structure():
    """Validate the table structure using connection pool"""
    try:
        required_columns = [
            "ticker_id",
            "date",
            "filing_date",
            "fiscal_period",
            "fiscal_year",
            "revenues",
            "net_income_loss",
            "assets",
            "equity",
        ]

        existing = fetch_all(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'raw_fundamental_data'"
        )
        existing_columns = [row["column_name"] for row in (existing or [])]

        missing_columns = [col for col in required_columns if col not in existing_columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        column_types = fetch_all(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'raw_fundamental_data' AND column_name IN ('ticker_id', 'date', 'filing_date', 'revenues', 'net_income_loss')"
        )

        logger.info("Key column data types:")
        for col in column_types or []:
            logger.info(f"- {col['column_name']}: {col['data_type']}")

        logger.info("Table structure validation complete!")
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

        logger.info("Database setup completed successfully")
        return True
    else:
        logger.error("Database setup failed")
        return False
