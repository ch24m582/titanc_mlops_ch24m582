from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline

def preprocess_data(input_path, output_path):
    spark = SparkSession.builder \
        .appName("Titanic Preprocessing") \
        .getOrCreate()

    # Load dataset
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Drop irrelevant columns
    df = df.drop("Name", "Ticket", "Cabin")

    # Fill missing values
    df = df.fillna({
        "Embarked": "S",
        "Age": df.select("Age").dropna().agg({"Age": "mean"}).first()[0]
    })

    # Encode categorical columns
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in ["Sex", "Embarked"]]
    encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in ["Sex", "Embarked"]]

    # Assemble features
    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_vec", "Embarked_vec"],
        outputCol="features_raw"
    )

    # Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    # Build pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    model = pipeline.fit(df)
    processed_df = model.transform(df)

    # Flatten feature vector into individual columns
    processed_df = processed_df.withColumn("features_array", vector_to_array("features"))

    feature_names = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_vec", "Embarked_vec"]
    for i, name in enumerate(feature_names):
        processed_df = processed_df.withColumn(name, processed_df["features_array"][i])

    # Save final flattened dataset
    processed_df.select(feature_names + ["Survived"]).write.parquet(output_path, mode="overwrite")

    spark.stop()

if __name__ == "__main__":
    preprocess_data("data/raw/train.csv", "data/processed/train_processed")



# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when
# from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
# from pyspark.ml import Pipeline

# def preprocess_data(input_path, output_path):
    # spark = SparkSession.builder \
        # .appName("Titanic Preprocessing") \
        # .getOrCreate()

    # # Load dataset
    # df = spark.read.csv(input_path, header=True, inferSchema=True)

    # # Drop irrelevant columns
    # df = df.drop("Name", "Ticket", "Cabin")

    # # Fill missing values
    # df = df.fillna({"Embarked": "S", "Age": df.select("Age").dropna().agg({"Age": "mean"}).first()[0]})

    # # Encode categorical columns
    # indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in ["Sex", "Embarked"]]
    # encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_vec") for col in ["Sex", "Embarked"]]

    # # Assemble features
    # assembler = VectorAssembler(
        # inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_vec", "Embarked_vec"],
        # outputCol="features_raw"
    # )

    # # Scale features
    # scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    # # Final pipeline
    # pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    # model = pipeline.fit(df)
    # processed_df = model.transform(df)

    # # Save processed data
    # processed_df.select("features", "Survived").write.parquet(output_path, mode="overwrite")

    # spark.stop()

# if __name__ == "__main__":
    # preprocess_data("data/raw/train.csv", "data/processed/train_processed")
