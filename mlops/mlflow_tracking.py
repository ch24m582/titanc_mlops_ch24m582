import mlflow
import mlflow.h2o
import h2o
from h2o.automl import H2OAutoML
import yaml
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models.signature import infer_signature

def load_config():
    with open("mlops/config.yaml") as f:
        return yaml.safe_load(f)

def load_data(path):
    spark = SparkSession.builder.appName("LoadParquet").getOrCreate()
    df = spark.read.parquet(path).toPandas()
    spark.stop()
    return df

def train_and_log(data_path):
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    h2o.init()
    df = load_data(data_path)

    # 🔍 Input example for schema
    input_example = df.drop(columns=["Survived"]).astype("float64")

    hf = h2o.H2OFrame(df)
    target = "Survived"
    features = [col for col in hf.columns if col != target]
    hf[target] = hf[target].asfactor()

    with mlflow.start_run():
        aml = H2OAutoML(max_models=10, seed=42)
        aml.train(x=features, y=target, training_frame=hf)

        model = aml.leader

        #  Infer signature from input and prediction
        signature = infer_signature(
            input_example,
            model.predict(h2o.H2OFrame(input_example)).as_data_frame()
        )
        print(" Run ID:", mlflow.active_run().info.run_id)
        mlflow.log_param("max_models", 10)
        mlflow.log_metric("auc", model.auc())
        mlflow.log_metric("accuracy", model.accuracy()[0][1])
    
        #  Log model with schema and input example
        model_uri = mlflow.h2o.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        print(" Logged model URI:", model_uri)

        # Log leaderboard
        aml.leaderboard.as_data_frame().to_csv("automl_leaderboard.csv", index=False)
        mlflow.log_artifact("automl_leaderboard.csv")

        print(" Model logged to MLflow with schema")

    h2o.shutdown()

if __name__ == "__main__":
    train_and_log("data/processed/train_processed")
