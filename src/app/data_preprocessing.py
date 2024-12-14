from pyspark.sql import SparkSession # type: ignore

def get_spark_session(app_name: str = "TaxiTripDurationPrediction"):
    """
    Create and return a SparkSession.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, file_path: str):
    """
    Load a CSV file into a Spark DataFrame.
    """
    return spark.read.csv(file_path, header=True, inferSchema=True)

def clean_data(df):
    """
    Perform basic data cleaning: remove outliers, handle missing values.
    """
    df = df.filter((df.trip_duration > 60) & (df.trip_duration < 7200))
    df = df.filter((df.passenger_count > 0) & (df.passenger_count <= 6))
    return df
