from src.database import get_duckdb_connection

CENSO_MERGE_IES_COURSE_PATH_IFMG = "data/raw/BQ_IFMG_2020_2024_CURSOS_IES.csv"

def ingest_censo_merge_ies_course_ifmg():
    """
    Ingests the CENSO MERGE IES COURSE IFMG dataset into DuckDB.

    This function reads the dataset from a CSV file and inserts it into a DuckDB table.
    If the table already exists, it will be replaced.

    Returns:
        None
    """
    # Establish a connection to DuckDB
    conn = get_duckdb_connection()

    # Create the table if it doesn't exist and insert data from the CSV file
    conn.execute(f"""
        CREATE OR REPLACE TABLE censo_merge_ies_course_ifmg AS
        SELECT * FROM read_csv_auto('{CENSO_MERGE_IES_COURSE_PATH_IFMG}')
    """)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    ingest_censo_merge_ies_course_ifmg()