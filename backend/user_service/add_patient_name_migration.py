#!/usr/bin/env python3
"""
Migration script to add patient_name column to scans table
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from utils.database import get_db_url

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Add patient_name column to scans table"""
    try:
        # Get database URL
        db_url = get_db_url()
        logger.info(f"Connecting to database...")
        
        # Create engine
        engine = create_engine(db_url)
        
        # Check if column already exists
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'scans' 
                AND COLUMN_NAME = 'patient_name'
            """))
            
            if result.fetchone():
                logger.info("patient_name column already exists. Migration skipped.")
                return
            
            # Add the column
            logger.info("Adding patient_name column to scans table...")
            connection.execute(text("""
                ALTER TABLE scans 
                ADD COLUMN patient_name VARCHAR(255) NULL
            """))
            connection.commit()
            
            logger.info("Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_migration()
