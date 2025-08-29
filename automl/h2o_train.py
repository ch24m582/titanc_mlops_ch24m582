import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from pyspark.sql import SparkSession

def load_parquet_as_pandas(path):
    spark = SparkSession.builder.appName("LoadParquet").getOrCreate()
    df = spark.read.parquet(path)
    pandas_df = df.toPandas()
    spark.stop()
    return pandas_df

def train_model(data_path):
    h2o.init()

    # Load processed data
    df = load_parquet_as_pandas(data_path)
    print(" Pandas columns before H2OFrame:", df.columns.tolist())
    # Convert to H2OFrame
    hf = h2o.H2OFrame(df)

    # Set target and features
    target = "Survived"
    features = [col for col in hf.columns if col != target]

    # Ensure classification task
    hf[target] = hf[target].asfactor()

    # Run AutoML
    aml = H2OAutoML(max_models=10, seed=42)
    aml.train(x=features, y=target, training_frame=hf)

    # Save best model
    model_path = h2o.save_model(model=aml.leader, path="models/", force=True)
    print(f" Best model saved to: {model_path}")
    print("Model input columns:", aml.leader._model_json['output']['names'])

    # Print expected input schema
    expected_columns = features
    expected_types = [hf.type(col) for col in features]
 
    print(" Expected columns:", expected_columns)
    print("Expected types:", expected_types)

    schema_df = pd.DataFrame({
        "column_name": expected_columns,
        "type": expected_types
    })
    schema_df.to_csv("models/expected_schema.csv", index=False)
    print(" Schema saved to models/expected_schema.csv")


    h2o.shutdown()

if __name__ == "__main__":
    train_model("data/processed/train_processed")



