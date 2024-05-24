import os
import sys
import toml
import numpy as np


from prefect import flow, get_run_logger

from ny_taxi.trainer.train import train_pipeline
from ny_taxi.config.config import DataLoaderConfig, PipelineConfig, TrainerConfig


@flow(validate_parameters=False)
def trainer() -> None:
    logger = get_run_logger()

    config = toml.load("config.toml")
    logger.info(f"config - {config}")

    month = config["trainer"]["month"]
    train_year = config["trainer"]["train_year"]
    test_year = config["trainer"]["test_year"]
    taxi_type = config["trainer"]["taxi_type"]
    dir_dataset = config["trainer"]["dir_dataset"]
    regressor_type = config["trainer"]["regressor_type"]
    mlflow_tracking_uri = config["trainer"]["mlflow_tracking_uri"]
    experiment_name = config["trainer"]["experiment_name"]

    # set train data loader config
    config_train_loader = DataLoaderConfig(
        dir_dataset=dir_dataset,
        year=train_year,
        taxi_type=taxi_type,
        month=month,
    )

    # set test data loader config
    config_test_loader = DataLoaderConfig(
        dir_dataset=dir_dataset,
        year=test_year,
        taxi_type=taxi_type,
        month=month,
    )

    # set pipeline config
    config_pipeline = PipelineConfig(regressor_type=regressor_type)

    # set trainer config
    config_trainer = TrainerConfig(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
    )

    # train the model pipeline
    train_pipeline(
        config_train_loader, config_test_loader, config_pipeline, config_trainer
    )
    return


def main() -> None:
    trainer()
    return


if __name__ == "__main__":
    main()
