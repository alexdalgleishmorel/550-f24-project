from pyspark.ml.regression import LinearRegression # type: ignore
from pyspark.ml.evaluation import RegressionEvaluator # type: ignore

def train_model(df, label_col="trip_duration", features_col="features"):
    """
    Train a regression model and return it.
    """
    lr = LinearRegression(featuresCol=features_col, labelCol=label_col)
    return lr.fit(df)

def evaluate_model(model, df, label_col="trip_duration", prediction_col="prediction"):
    """
    Evaluate the model using RMSE.
    """
    predictions = model.transform(df)
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}

def test_model(model, df, output_cols=["id", "trip_duration", "prediction"]):
    """
    Predict using the model on the test dataset and return the predictions DataFrame.
    """
    predictions = model.transform(df)
    return predictions.select(*output_cols)
