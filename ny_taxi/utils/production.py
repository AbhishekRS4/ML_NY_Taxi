import os
import mlflow
import shutil
import logging

from typing import Tuple
from mlflow.entities import ViewType
from mlflow.client import MlflowClient


from ny_taxi.config.config import ProductionConfig


def list_experiments(config_production: ProductionConfig) -> None:
    client = MlflowClient(tracking_uri=config_production.mlflow_tracking_uri)
    logging.info("Top 5 experiments with best test rmse")
    runs = client.search_runs(
        experiment_ids="1",
        filter_string="",  # "metrics.test_rmse < 6"
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.test_rmse ASC"],
    )
    for run in runs:
        logging.info(
            f"run_id:{run.info.run_id}, rmse: {run.data.metrics['test_rmse']:.4f}"
        )
    return


def list_registered_models(config_production: ProductionConfig) -> None:
    client = MlflowClient(tracking_uri=config_production.mlflow_tracking_uri)
    logging.info("All the latest versions of the registered models")
    latest_versions = client.get_latest_versions(name=config_production.experiment_name)
    for version in latest_versions:
        logging.info(
            f"name: {version.name}, version: {version.version}, tags: {version.tags}, run_id: {version.run_id}"
        )
    return


def get_latest_registered_model(
    config_production: ProductionConfig,
) -> Tuple[str, str, int]:
    client = MlflowClient(tracking_uri=config_production.mlflow_tracking_uri)
    logging.info("All the latest versions of the registered models")
    latest_versions = client.get_latest_versions(name=config_production.experiment_name)
    logging.info(latest_versions[-1])
    model_run_id = latest_versions[-1].run_id
    model_name = latest_versions[-1].name
    model_version = latest_versions[-1].version
    list_files_for_prod = os.listdir(latest_versions[-1].source)
    for _file in list_files_for_prod:
        shutil.copy2(os.path.join(latest_versions[-1].source, _file), "model_for_prod")
    return model_name, model_run_id, model_version
