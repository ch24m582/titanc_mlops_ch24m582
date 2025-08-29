from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model from MLflow Registry
model_name = "TitanicClassifier"
model_stage = "Production"
#model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

mlflow.set_tracking_uri("file:./mlruns")
model = mlflow.pyfunc.load_model("models:/TitanicClassifier@production")


# Define input schema
class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_vec: float
    Embarked_vec: float

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        input_df = pd.DataFrame([passenger.dict()])

        #  Cast all columns to float to match H2O's 'real' type
        input_df = input_df.astype("float64")

        print("Input columns:", input_df.columns.tolist())
        print("Model input schema:", model.metadata.get_input_schema())
        print("Input dtypes:\n", input_df.dtypes)

        prediction = model.predict(input_df)
        

        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
