from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, unix_timestamp, abs, col, hour, dayofweek, when, month, quarter, lit, sin, cos, radians, sqrt, atan2
from pyspark.ml.feature import VectorAssembler, PolynomialExpansion, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Enhanced Linear Regression Model") \
    .getOrCreate()

# File paths
train_path = "../data/training.csv"
validation_path = "../data/validation.csv"
test_path = "../data/test.csv"

# Load data
def load_data(file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

train_df = load_data(train_path)
validation_df = load_data(validation_path)
test_df = load_data(test_path)

# ------------------------
# 1. Data Cleaning
# ------------------------
def clean_data(df):
    df = df.withColumn("pickup_datetime", to_timestamp("pickup_datetime", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
    df = df.filter((df.trip_duration > 60) & (df.trip_duration < 7200))  # Trip duration between 1 min and 2 hrs
    df = df.filter((df.pickup_longitude.between(-74.3, -73.7)) & (df.pickup_latitude.between(40.5, 41.0)))  # NYC bounds
    df = df.filter((df.dropoff_longitude.between(-74.3, -73.7)) & (df.dropoff_latitude.between(40.5, 41.0)))
    return df

train_df = clean_data(train_df)
validation_df = clean_data(validation_df)
test_df = clean_data(test_df)

# Add Unix timestamp for pickup_datetime
train_df = train_df.withColumn("pickup_datetime_numeric", unix_timestamp("pickup_datetime"))
validation_df = validation_df.withColumn("pickup_datetime_numeric", unix_timestamp("pickup_datetime"))
test_df = test_df.withColumn("pickup_datetime_numeric", unix_timestamp("pickup_datetime"))

# ------------------------
# 2. Feature Enhancement
# ------------------------
def enhance_features(df):
    R = 6371  # Earth radius in kilometers
    return (
        df.withColumn("hour_of_day", hour("pickup_datetime"))
        .withColumn("day_of_week", dayofweek("pickup_datetime"))
        .withColumn("is_weekend", when(col("day_of_week") > 5, 1).otherwise(0))
        .withColumn("month", month("pickup_datetime"))
        .withColumn("quarter", quarter("pickup_datetime"))
        .withColumn(
            "season",  # Map months to seasons
            when(col("month").isin(12, 1, 2), lit(1))  # Winter
            .when(col("month").isin(3, 4, 5), lit(2))  # Spring
            .when(col("month").isin(6, 7, 8), lit(3))  # Summer
            .otherwise(lit(4)),  # Fall
        )
        .withColumn("lat_diff", abs(col("pickup_latitude") - col("dropoff_latitude")))
        .withColumn("lon_diff", abs(col("pickup_longitude") - col("dropoff_longitude")))
        .withColumn(
            "distance",
            R
            * 2
            * atan2(
                sqrt(
                    sin((radians(col("dropoff_latitude") - col("pickup_latitude")) / 2)) ** 2
                    + cos(radians(col("pickup_latitude")))
                    * cos(radians(col("dropoff_latitude")))
                    * sin((radians(col("dropoff_longitude") - col("pickup_longitude")) / 2)) ** 2
                ),
                sqrt(
                    1
                    - (
                        sin((radians(col("dropoff_latitude") - col("pickup_latitude")) / 2)) ** 2
                        + cos(radians(col("pickup_latitude")))
                        * cos(radians(col("dropoff_latitude")))
                        * sin((radians(col("dropoff_longitude") - col("pickup_longitude")) / 2)) ** 2
                    )
                ),
            ),
        )
        .withColumn(
            "is_rush_hour",
            when(
                (col("hour_of_day").between(7, 9))
                | (col("hour_of_day").between(16, 19)),
                1,
            ).otherwise(0),
        )
    )


train_df = enhance_features(train_df)
validation_df = enhance_features(validation_df)
test_df = enhance_features(test_df)

# ------------------------
# 3. Polynomial Expansion (degree=3)
# ------------------------
# Define feature columns
base_feature_cols = [
    "distance", "lat_diff", "lon_diff", "hour_of_day", "day_of_week", 
    "is_weekend", "is_rush_hour", "month", "quarter", "season", "passenger_count"
]
assembler = VectorAssembler(inputCols=base_feature_cols, outputCol="raw_features")

# Polynomial expansion to add non-linear feature interactions
poly_expansion = PolynomialExpansion(degree=3, inputCol="raw_features", outputCol="expanded_features")

# Standardization of expanded features
scaler = StandardScaler(inputCol="expanded_features", outputCol="features", withStd=True, withMean=True)

# Transform train, validation, and test data
train_df = assembler.transform(train_df)
train_df = poly_expansion.transform(train_df)
train_df = scaler.fit(train_df).transform(train_df).select("features", "trip_duration")

validation_df = assembler.transform(validation_df)
validation_df = poly_expansion.transform(validation_df)
validation_df = scaler.fit(validation_df).transform(validation_df).select("features", "trip_duration")

test_df = assembler.transform(test_df)
test_df = poly_expansion.transform(test_df)
test_df = scaler.fit(test_df).transform(test_df).select("features", "trip_duration")

# ------------------------
# 4. Regularization - Linear Regression Model
# ------------------------
lr = LinearRegression(featuresCol="features", labelCol="trip_duration", maxIter=100, regParam=0.05, elasticNetParam=0.3)
model = lr.fit(train_df)

# ------------------------
# 5. Evaluation
# ------------------------
# Validate the model
validation_predictions = model.transform(validation_df)

# Evaluate using RMSE, MAE, and R²
def evaluate_model(predictions, label_col="trip_duration", prediction_col="prediction"):
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2")
    return {
        "RMSE": evaluator_rmse.evaluate(predictions),
        "MAE": evaluator_mae.evaluate(predictions),
        "R²": evaluator_r2.evaluate(predictions)
    }

validation_metrics = evaluate_model(validation_predictions, label_col="trip_duration", prediction_col="prediction")
print("Validation Metrics:")
for metric, value in validation_metrics.items():
    print(f"{metric}: {value}")

# Test the model
test_predictions = model.transform(test_df)
test_metrics = evaluate_model(test_predictions, label_col="trip_duration", prediction_col="prediction")
print("Test Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")

# Stop Spark Session
spark.stop()
