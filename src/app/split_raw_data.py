import os
import sys
from pyspark.sql import SparkSession

# Setting JAVA_HOME and updating PATH
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"
os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:{os.environ['PATH']}"

raw_data_path = "../data/raw.csv"
if not os.path.exists(raw_data_path):
    print("\nraw.csv doesn't exist in /data directory.\n\nYou'll need to retrieve it from https://www.kaggle.com/c/nyc-taxi-trip-duration/data.\n\nIt is labelled as train.csv.\n\n")
    sys.exit(1)

# Creating Spark session
spark = SparkSession.builder \
    .appName("Split Dataset") \
    .getOrCreate()

# Loading raw.csv dataset
file_path = "../data/raw.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Randomly splitting the raw dataset into training, validation, and test sets
training_df, validation_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)

# Saving the splits to the separate CSV files
training_df.write.csv("../data/training.csv", header=True, mode="overwrite")
validation_df.write.csv("../data/validation.csv", header=True, mode="overwrite")
test_df.write.csv("../data/test.csv", header=True, mode="overwrite")

# Stoping Spark session
spark.stop()
