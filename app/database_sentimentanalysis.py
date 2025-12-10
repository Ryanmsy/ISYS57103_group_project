import pandas as pd
import sqlite3
import os

def migrate_excel_to_sqlite(excel_path: str, db_path: str, table_name: str = "reviews"):
    """
    Simulates an ETL (Extract, Transform, Load) process.
    Reads from a raw file (Excel) and loads into a Data Warehouse (SQLite).
    """
    
    # 1. Extract
    if not os.path.exists(excel_path):
        # Create dummy data if file doesn't exist for demonstration
        print(f"File {excel_path} not found. Creating dummy data...")
        data = {
            'reviewText': ['Great product', 'Terrible', 'Okay', 'Amazing', 'Bad'],
            'rating': [5, 1, 3, 5, 2]
        }
        df = pd.DataFrame(data)
    else:
        print(f"Extracting data from {excel_path}...")
        df = pd.read_excel(excel_path)

    # 2. Transform (Basic standardization)
    # Ensure columns match what the model expects
    # The original code expected 'text', but dataset often has 'reviewText'. 
    # We standardize to 'text' here in the Warehouse.
    if 'reviewText' in df.columns:
        df = df.rename(columns={'reviewText': 'text'})
    
    # Filter for required columns
    df = df[['text', 'rating']]

    # 3. Load
    print(f"Loading data into SQLite database: {db_path}...")
    try:
        # Create connection
        conn = sqlite3.connect(db_path)
        
        # Write to SQL (replace if exists mimics a full refresh)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"Success! Loaded {len(df)} rows into table '{table_name}'.")
        
        # Verification query
        cursor = conn.cursor()
        cursor.execute(f"SELECT count(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Verification: Database contains {count} rows.")
        
    except Exception as e:
        print(f"ETL Failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # Point this to your actual Excel file location
    EXCEL_FILE = "amazon_test_2500.xlsx" 
    DB_FILE = "corporate_data_warehouse.db"
    
    migrate_excel_to_sqlite(EXCEL_FILE, DB_FILE)