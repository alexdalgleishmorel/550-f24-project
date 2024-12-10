from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Linear Regression Model Training") \
    .getOrCreate()

train_path = "../data/training.csv"
validation_path = "../data/validation.csv"
test_path = "../data/test.csv"

def load_data(file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

train_df = load_data(train_path)
validation_df = load_data(validation_path)
test_df = load_data(test_path)

# DEFINING FEATURES
feature_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", 
                "dropoff_latitude", "passenger_count", "pickup_hour", "pickup_day"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df).select("features", "trip_duration")
validation_df = assembler.transform(validation_df).select("features", "trip_duration")
test_df = assembler.transform(test_df).select("features", "trip_duration")

# TRAINING LINEAR REGRESSION MODEL
lr = LinearRegression(featuresCol="features", labelCol="trip_duration", maxIter=100, regParam=0.1, elasticNetParam=0.8)
model = lr.fit(train_df)

# VALIDATING MODEL
validation_predictions = model.transform(validation_df)
evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="rmse")
validation_rmse = evaluator.evaluate(validation_predictions)
print(f"Validation RMSE: {validation_rmse}")

# EVALUATE USING TEST DATA SET
test_predictions = model.transform(test_df)
test_rmse = evaluator.evaluate(test_predictions)
print(f"Test RMSE: {test_rmse}")

# OUTPUT METRICS ON MODEL
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")

spark.stop()
