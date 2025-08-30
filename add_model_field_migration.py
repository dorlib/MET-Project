#!/usr/bin/env python3
"""
Database migration script to add model_name field to scans table
"""

import mysql.connector
import sys

def run_migration():
    try:
        # Database connection
        conn = mysql.connector.connect(
            host='localhost',
            port=13306,
            user='root',
            password='rootpassword',
            database='metastasis_db'
        )
        cursor = conn.cursor()
        
        print("🔄 Adding model_name column to scans table...")
        
        # Add the model_name column
        cursor.execute("""
            ALTER TABLE scans 
            ADD COLUMN model_name VARCHAR(255) NULL
            AFTER patient_name
        """)
        
        print("✅ Added model_name column successfully")
        
        # Update existing scans with default model name
        # Since we don't know which model was used, set a default
        cursor.execute("""
            UPDATE scans 
            SET model_name = 'brats_t1ce.pth' 
            WHERE model_name IS NULL
        """)
        
        rows_updated = cursor.rowcount
        print(f"✅ Updated {rows_updated} existing scans with default model 'brats_t1ce.pth'")
        
        # Commit changes
        conn.commit()
        print("✅ Migration completed successfully!")
        
    except mysql.connector.Error as e:
        if e.errno == 1060:  # Column already exists
            print("ℹ️ Column 'model_name' already exists in scans table")
        else:
            print(f"❌ Database error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    run_migration()
