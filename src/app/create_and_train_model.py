from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, hour, dayofweek, log1p
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import math

def manhattan_distance(lat1, lon1, lat2, lon2):
    lat_dist = abs(lat2 - lat1) * 111
    lon_dist = abs(lon2 - lon1) * 85
    return lat_dist + lon_dist

manhattan_udf = udf(lambda lat1, lon1, lat2, lon2: manhattan_distance(lat1, lon1, lat2, lon2), DoubleType())

spark = SparkSession.builder \
    .appName("Enhanced Taxi Trip Duration Model") \
    .getOrCreate()

train_path = "../data/training.csv"
validation_path = "../data/validation.csv"
test_path = "../data/test.csv"

# Loading data
def load_data(file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

train_df = load_data(train_path)
validation_df = load_data(validation_path)
test_df = load_data(test_path)

# Adding features and Manhattan distance
def add_features(df):
    return df.withColumn("manhattan_distance", manhattan_udf(col("pickup_latitude"), col("pickup_longitude"),
                                                             col("dropoff_latitude"), col("dropoff_longitude"))) \
             .withColumn("pickup_hour", hour(col("pickup_datetime"))) \
             .withColumn("pickup_dayofweek", dayofweek(col("pickup_datetime")))

train_df = add_features(train_df)
validation_df = add_features(validation_df)
test_df = add_features(test_df)

# Removing outliers in training data based on trip_duration
quantiles = train_df.approxQuantile("trip_duration", [0.01, 0.99], 0)
train_df = train_df.filter((col("trip_duration") >= quantiles[0]) & (col("trip_duration") <= quantiles[1]))

# Log-transform trip_duration
train_df = train_df.withColumn("log_trip_duration", log1p(col("trip_duration")))
validation_df = validation_df.withColumn("log_trip_duration", log1p(col("trip_duration")))
test_df = test_df.withColumn("log_trip_duration", log1p(col("trip_duration")))

# Defining features
feature_cols = ["manhattan_distance", "pickup_hour", "pickup_dayofweek"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df).select("features", "log_trip_duration")
validation_df = assembler.transform(validation_df).select("features", "log_trip_duration")
test_df = assembler.transform(test_df).select("features", "log_trip_duration")

# Training Gradient Boosted Trees (GBT) regression model
gbt = GBTRegressor(featuresCol="features", labelCol="log_trip_duration", maxIter=50, maxDepth=5, stepSize=0.1)
model = gbt.fit(train_df)

# Validating model
validation_predictions = model.transform(validation_df)

# Evaluating model performance
def evaluate_model(predictions, label_col="log_trip_duration", prediction_col="prediction"):
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}

validation_metrics = evaluate_model(validation_predictions)
print("Validation Metrics:")
for metric, value in validation_metrics.items():
    print(f"{metric}: {value}")

# Testing model
test_predictions = model.transform(test_df)
test_metrics = evaluate_model(test_predictions)
print("Test Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")

print("Feature Importances:")
for feature, importance in zip(feature_cols, model.featureImportances):
    print(f"{feature}: {importance}")

# Stopping Spark session
spark.stop()
