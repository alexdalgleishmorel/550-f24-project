from pyspark.sql import SparkSession
from pyspark.sql import functions as Functions
from pyspark.sql.functions import col
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

def transform_data(df): 
    df = df.withColumn('displacement', Functions.sqrt((col("pickup_latitude") - col("dropoff_latitude"))**2 + (col("pickup_longitude") - col("dropoff_longitude"))**2))
    
    df =df.withColumn('dropoff_weekday', Functions.dayofweek(col('dropoff_datetime')))
    df =df.withColumn('dropoff_day', Functions.dayofmonth(col('dropoff_datetime')))
    df =df.withColumn('dropoff_month', Functions.month(col('dropoff_datetime')))

    df = df.withColumn('manhattan_distance', Functions.abs(Functions.col('pickup_latitude') - Functions.col('dropoff_latitude')) + Functions.abs(Functions.col('pickup_longitude') - Functions.col('dropoff_longitude')))
  
    df =df.withColumn('pickup_weekday', Functions.dayofweek(col('pickup_datetime')))
    df =df.withColumn('isWeekend', Functions.when(Functions.col('pickup_weekday').isin(1, 7), 1).otherwise(0))
    df =df.withColumn('pickup_day', Functions.dayofmonth(col('pickup_datetime')))
    df =df.withColumn('pickup_month', Functions.month(col('pickup_datetime')))
    df =df.withColumn('pickup_time', Functions.month(col('pickup_datetime')))
    df =df.withColumn('pickup_hour', Functions.hour(Functions.col('pickup_datetime'))) 
    df =df.withColumn('pickup_minute', Functions.minute(Functions.col('pickup_datetime'))) 
    df =df.withColumn('isRushhour',Functions.when((Functions.hour(Functions.col('pickup_datetime')).between(8, 9)) & (Functions.dayofweek(Functions.col('pickup_datetime')).between(2, 6)), 1).when((Functions.hour(Functions.col('pickup_datetime')).between(15, 19)) & (Functions.dayofweek(Functions.col('pickup_datetime')).between(2, 6)), 1).otherwise(0))
    return df


train_df = load_data(train_path)
train_df = transform_data(train_df)

validation_df = load_data(validation_path)
validation_df = transform_data(validation_df)

test_df = load_data(test_path)
test_df = transform_data(test_df)

# DEFINING FEATURES
feature_cols = ["manhattan_distance", "passenger_count", "pickup_day", "pickup_month", "isWeekend", "pickup_hour", "isRushhour"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df).select("features", "trip_duration")
validation_df = assembler.transform(validation_df).select("features", "trip_duration")
test_df = assembler.transform(test_df).select("features", "trip_duration")

# TRAINING LINEAR REGRESSION MODEL
lr = LinearRegression(featuresCol="features", labelCol="trip_duration", maxIter=100, regParam=0.05, elasticNetParam=0.8)
model = lr.fit(train_df)

# VALIDATING MODEL
validation_predictions = model.transform(validation_df)
evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="rmse")
validation_rmse = evaluator.evaluate(validation_predictions)
print(f"Validation RMSE: {validation_rmse}")

# EVALUATE USING TEST DATA SET
test_predictions = model.transform(test_df)
predictions = test_predictions.toPandas()
predictions.to_csv('predictions.csv', index=False)
test_rmse = evaluator.evaluate(test_predictions)
print(f"Test RMSE: {test_rmse}")

# OUTPUT METRICS ON MODEL
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")

spark.stop()
