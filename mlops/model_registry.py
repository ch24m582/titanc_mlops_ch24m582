import mlflow
from mlflow.tracking import MlflowClient
import yaml

def load_config():
    with open("mlops/config.yaml") as f:
        return yaml.safe_load(f)

def register_model():
    cfg = load_config()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg["experiment_name"])
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.auc DESC"],
        max_results=1
    )

    if not runs:
        print(" No runs found with AUC metric. Please check training logs.")
        return

    latest_run = runs[0]
    #run_id = latest_run.info.run_id
    run_id = "e2bd23799b7e4cbba5fa19e81f1d64c6"
    model_uri = f"runs:/{run_id}/model"

    # Register the model
    result = mlflow.register_model(model_uri, cfg["model_name"])
    print(f" Registered model version {result.version} for '{cfg['model_name']}'")

    # Assign alias 'production'
    client.set_registered_model_alias(
        name="TitanicClassifier",
        version=result.version,
        alias="production"
    )

    print(f" Alias 'production' assigned to version {result.version}")

if __name__ == "__main__":
    register_model()
