import os
import mlflow
import logging

from mlflow.client import MlflowClient
from mlflow.entities import ViewType


from ny_taxi.config.config import ProductionConfig

def list_experiments(config_production: ProductionConfig) -> None:
    client = MlflowClient(tracking_uri=config_production.mlflow_tracking_uri)
    logging.info("Top 5 experiments with best test rmse")
    runs = client.search_runs(
        experiment_ids="1",
        filter_string="", #"metrics.test_rmse < 6"
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.test_rmse ASC"]
    )
    for run in runs:
        logging.info(f"run_id:{run.info.run_id}, rmse: {run.data.metrics['test_rmse']:.4f}")
    return

def list_registered_models(config_production: ProductionConfig) -> None:
    client = MlflowClient(tracking_uri=config_production.mlflow_tracking_uri)
    logging.info("All the latest versions of the registered models")
    latest_versions = client.get_latest_versions(name="ny_taxi_duration_regressor")
    for version in latest_versions:
        logging.info(f"version: {version.version}, tags: {version.tags}, run_id: {version.run_id}")
    return
