from sqlalchemy import create_engine
import os
import pandas as pd

def get_engine():
    db_host = os.getenv("DB_HOST", "localhost")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "root")
    db_name = os.getenv("DB_NAME", "nyc_taxi_data")

    # SQLAlchemy connection string
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)
    return engine

def example_query():
    """
    Fetches data from the 'trips' table and prints it as a DataFrame.
    """
    try:
        engine = get_engine()
        query = "SELECT * FROM trips;"
        
        # Using Pandas to run the query and loading the result into a DataFrame
        df = pd.read_sql(query, engine)
        print("Query result as DataFrame:\n")
        print(df)
    except Exception as e:
        print("Error during query execution:", e)
