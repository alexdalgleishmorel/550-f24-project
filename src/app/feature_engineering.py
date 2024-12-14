from pyspark.sql.functions import hour, dayofweek, month, col, radians, sin, cos, sqrt, atan2 # type: ignore
from pyspark.sql.functions import udf # type: ignore
from pyspark.ml.clustering import KMeans # type: ignore
from pyspark.ml.feature import VectorAssembler # type: ignore
from pyspark.sql import DataFrame # type: ignore


def add_temporal_features(df):
    """
    Add temporal features like hour, day of week, and month.
    """
    return df.withColumn("pickup_hour", hour("pickup_datetime")) \
             .withColumn("pickup_dayofweek", dayofweek("pickup_datetime")) \
             .withColumn("pickup_month", month("pickup_datetime"))

def calculate_distance(df):
    """
    Add a column for Haversine distance between pickup and dropoff points.
    """
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude differences to radians
    dlat = radians(col("dropoff_latitude") - col("pickup_latitude"))
    dlon = radians(col("dropoff_longitude") - col("pickup_longitude"))

    # Apply the Haversine formula
    a = sin(dlat / 2) ** 2 + cos(radians(col("pickup_latitude"))) * cos(radians(col("dropoff_latitude"))) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate distance and add it as a new column
    return df.withColumn("trip_distance", R * c)

def apply_kmeans(df, feature_cols, prediction_col, k):
    """
    Apply K-Means clustering to specified features and return updated DataFrame.
    The temporary 'features' column is dropped after creating the cluster column.
    """
    print("\nBefore assembling features:")
    df.show(10)

    # Step 1: Assemble the feature columns into a vector
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = vector_assembler.transform(df)
    
    print("\nAfter assembling features:")
    df.show(10)

    # Step 2: Fit the K-Means model
    print("\nFitting K-Means model...")
    kmeans = KMeans(featuresCol="features", predictionCol=prediction_col, k=k)
    model = kmeans.fit(df)

    # Step 3: Transform the DataFrame to add the prediction column
    print("\nAfter fitting K-Means, adding predictions:")
    df = model.transform(df)
    df.show(10)

    # Step 4: Drop the temporary 'features' column
    print("\nDropping the temporary 'features' column...")
    df = df.drop("features")
    df.show(10)

    return df

def assemble_features(df: DataFrame, feature_cols: list, output_col: str = "features") -> DataFrame:
    """
    Assemble all feature columns into a single vector column. If the output column exists, drop it first.
    """
    if output_col in df.columns:
        df = df.drop(output_col)  # Drop the existing features column to avoid conflicts

    assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col)
    return assembler.transform(df)

