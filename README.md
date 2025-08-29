Titanic MLOps Pipeline

An end-to-end machine learning pipeline for predicting Titanic survival, built with scalable, reproducible, and production-ready MLOps practices.

Features

  Distributed preprocessing with Apache Spark
  
  Automated model training via H2O AutoML
  
  Experiment tracking and model registry using MLflow
  
  RESTful prediction API with FastAPI
  
  Data versioning and pipeline reproducibility via DVC + Git
  
  Container-ready deployment with Docker
  
  Modular, testable, and CI/CD-friendly architecture


Sructure of titanc_mlops_ch24m582/

├── data/

│   ├── raw/                  # Original train/test CSVs (tracked via DVC)

│   └── processed/            # Cleaned and encoded datasets

├── spark_pipeline/           # Spark-based preprocessing scripts

├── automl/                   # H2O AutoML training and leaderboard

├── models/                   # Saved models and metadata

├── mlops/

│   ├── config.yaml           # Central config for model name, paths, etc.

│   ├── mlflow_register.py    # Registers model and assigns alias

│   └── predict_schema.csv    # Expected input schema for inference

├── deployment/

│   ├── app.py                # FastAPI app for serving predictions

│   └── Dockerfile            # Containerization setup

├── mlruns/                   # MLflow tracking artifacts

├── artifacts/                # Intermediate outputs (optional)

├── testing/                  # Unit tests and API validation

├── .dvc/                     # DVC metadata

├── .gitignore

└── README.md


Setup Instructions

1. Clone the repo

	git clone https://github.com/ch24m582/titanc_mlops_ch24m582.git
	
	cd titanc_mlops_ch24m582

2. Create and activate virtual environment

	python3 -m venv venv
	
	source venv/bin/activate
	
	pip install -r requirements.txt

3. Pull raw data via DVC

  	dvc pull

4. Preprocess data with Spark

	python spark_pipeline/preprocess.py

6. Train model with H2O AutoML

  	python automl/train_model.py
7. Register model with MLflow
   
  	python mlops/mlflow_register.py

9. Launch FastAPI server

  	uvicorn deployment.app:app --reload

DVC Data Tracking

dvc add data/raw/train.csv

dvc add data/raw/test.csv

git add data/raw/*.dvc .gitignore

git commit -m "Track raw data with DVC"

Sample Prediction

POST /predict

{

  "Pclass": 3,
  
  "Age": 22.0,
  
  "SibSp": 1,
  
  "Parch": 0,
  
  "Fare": 7.25,
  
  "Sex_vec": 1.0,
  
  "Embarked_vec": 0.0
}



Response

	{
	
	  0
	
	}
