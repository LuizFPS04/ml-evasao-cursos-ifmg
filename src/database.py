import duckdb

def get_duckdb_connection(database_path: str = "data/analytics.duckdb"):
    """
    Establishes a connection to a DuckDB database.

    Args:
        database_path (str): The path to the DuckDB database file. Defaults to "data/analytics.duckdb".

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    try:
        conn = duckdb.connect(database=database_path)
        return conn
    except Exception as e:
        print(f"Error connecting to DuckDB database: {e}")
        raise 