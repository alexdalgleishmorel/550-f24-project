from pyspark.sql import SparkSession
from pyspark.sql import functions as Functions
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("Linear Regression Model Training") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.num.executors", "4") \
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
    df =df.withColumn('isRushhour',Functions.when((Functions.hour(Functions.col('pickup_datetime')).between(7, 9)) & (Functions.dayofweek(Functions.col('pickup_datetime')).between(2, 6)), 1).when((Functions.hour(Functions.col('pickup_datetime')).between(15, 19)) & (Functions.dayofweek(Functions.col('pickup_datetime')).between(2, 6)), 1).otherwise(0))
    return df

def evaluate_model(predictions, label_col="trip_duration", prediction_col="prediction"):
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}

# LOAD DATA INTO DATAFRAMES AND TRANSFORM
train_df = load_data(train_path)
train_df = transform_data(train_df)

validation_df = load_data(validation_path)
validation_df = transform_data(validation_df)

test_df = load_data(test_path)
test_df = transform_data(test_df)

# DEFINING FEATURES
feature_cols = ["manhattan_distance", "pickup_weekday", "passenger_count", "pickup_day", "pickup_month", "pickup_hour", "pickup_minute"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df).select("features", "trip_duration")
validation_df = assembler.transform(validation_df).select("features", "trip_duration")
test_df = assembler.transform(test_df).select("features", "trip_duration")

# TRAINING LINEAR REGRESSION MODEL
lr = RandomForestRegressor(featuresCol="features", labelCol="trip_duration", numTrees=100, maxDepth=10,minInstancesPerNode=2)
model = lr.fit(train_df)

# VALIDATING MODEL
validation_predictions = model.transform(validation_df)
validation_metrics = evaluate_model(validation_predictions)
print("Validation Metrics:")
for metric, value in validation_metrics.items():
    print(f"{metric}: {value}")

# EVALUATE USING TEST DATA SET
test_predictions = model.transform(test_df)
test_metrics = evaluate_model(test_predictions)
print("Test Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")

# OUTPUT METRICS ON MODEL
def plot_predictions_vs_actual(predictions_df, label_col="trip_duration", prediction_col="prediction"):
    actual = predictions_df.select(label_col).toPandas()[label_col]
    predicted = predictions_df.select(prediction_col).toPandas()[prediction_col]
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, predicted, alpha=0.6, color='b', label='Predicted vs Actual')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='r', label='Perfect Prediction')
    plt.xlabel('Actual Trip Duration')
    plt.ylabel('Predicted Trip Duration')
    plt.title('Prediction vs Actual Trip Duration (Validation Set)')
    plt.legend()
    plt.xlim(0, 6000)
    plt.ylim(0, 5000)
    plt.savefig('prediction_vs_actual.png')
    plt.show()

plot_predictions_vs_actual(validation_predictions)

print("Feature Importances:")
for feature, importance in zip(feature_cols, model.featureImportances):
    print(f"{feature}: {importance}")

spark.stop()
