from data_preprocessing import get_spark_session, load_data, clean_data
from feature_engineering import add_temporal_features, calculate_distance, apply_kmeans, assemble_features
from model_pipeline import train_model, evaluate_model, test_model


def main():
    try:
        # Step 1: Initialize Spark session
        spark = get_spark_session()

        # Step 2: Load and clean data
        print("Loading training dataset...")
        train_df = load_data(spark, "../data/training.csv")
        print("Training data loaded successfully.")

        # Inspect schema and sample rows before cleaning
        print("Schema before cleaning:")
        train_df.printSchema()
        print("Top 10 rows before cleaning:")
        train_df.show(10)

        print("Cleaning training dataset...")
        train_df = clean_data(train_df)
        print("Training data cleaned successfully.")

        # Inspect schema and rows after cleaning
        print("Schema after cleaning:")
        train_df.printSchema()
        print("Top 10 rows after cleaning:")
        train_df.show(10)

        # Step 3: Feature engineering
        print("Starting feature engineering...")
        train_df = add_temporal_features(train_df)
        train_df = calculate_distance(train_df)

        # Inspect schema and rows after adding features
        print("Schema after adding temporal and distance features:")
        train_df.printSchema()
        print("Top 10 rows after adding temporal and distance features:")
        train_df.show(10)

        # Apply K-Means clustering for pickup and dropoff regions
        print("Applying K-Means clustering for pickup regions...")
        train_df = apply_kmeans(train_df, ["pickup_latitude", "pickup_longitude"], "pickup_cluster", 10)
        print("Applying K-Means clustering for dropoff regions...")
        train_df = apply_kmeans(train_df, ["dropoff_latitude", "dropoff_longitude"], "dropoff_cluster", 10)

        # Inspect schema and rows after clustering
        print("Schema after clustering:")
        train_df.printSchema()
        print("Top 10 rows after clustering:")
        train_df.show(10)

        # Assemble features
        feature_columns = ["pickup_hour", "pickup_dayofweek", "pickup_month", 
                           "trip_distance", "pickup_cluster", "dropoff_cluster"]
        print("Assembling features...")
        train_df = assemble_features(train_df, feature_columns)

        # Inspect schema and rows after assembling features
        print("Schema after assembling features:")
        train_df.printSchema()
        print("Top 10 rows after assembling features:")
        train_df.show(10)

        # Step 4: Train model
        print("Training the model...")
        model = train_model(train_df)

        # Step 5: Evaluate model
        print("Evaluating the model...")
        val_df = load_data(spark, "../data/validation.csv")

        # Apply the same feature engineering to the validation dataset
        print("\nApplying feature engineering to validation dataset...")
        val_df = add_temporal_features(val_df)
        val_df = calculate_distance(val_df)
        val_df = apply_kmeans(val_df, ["pickup_latitude", "pickup_longitude"], "pickup_cluster", 10)
        val_df = apply_kmeans(val_df, ["dropoff_latitude", "dropoff_longitude"], "dropoff_cluster", 10)

        # Assemble features for validation dataset
        print("\nAssembling features for validation dataset...")
        val_df = assemble_features(val_df, feature_columns)

        # Evaluate the model
        evaluation_results = evaluate_model(model, val_df)
        print(f"Validation results: {evaluation_results}")

        # Step 6: Test model
        print("Testing the model...")
        test_df = load_data(spark, "../data/test.csv")

        # Apply the same feature engineering to the test dataset
        print("\nApplying feature engineering to test dataset...")
        test_df = add_temporal_features(test_df)
        test_df = calculate_distance(test_df)
        test_df = apply_kmeans(test_df, ["pickup_latitude", "pickup_longitude"], "pickup_cluster", 100)
        test_df = apply_kmeans(test_df, ["dropoff_latitude", "dropoff_longitude"], "dropoff_cluster", 100)

        # Assemble features for test dataset
        print("\nAssembling features for test dataset...")
        test_df = assemble_features(test_df, feature_columns)

        # Generate predictions for the test dataset
        predictions = test_model(model, test_df)

        # Display the predictions
        print("\nPredictions on test dataset:")
        predictions.show(10)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
