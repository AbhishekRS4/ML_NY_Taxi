import os
import logging
import argparse
import numpy as np


from ny_taxi.trainer.train import train_pipeline
from ny_taxi.config.config import DataLoaderConfig, PipelineConfig


def trainer(ARGS: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    # set train data loader config
    config_train_loader = DataLoaderConfig(
        dir_dataset=ARGS.dir_dataset,
        year=ARGS.train_year,
        taxi_type=ARGS.taxi_type,
        month=ARGS.month,
    )

    # set test data loader config
    config_test_loader = DataLoaderConfig(
        dir_dataset=ARGS.dir_dataset,
        year=ARGS.test_year,
        taxi_type=ARGS.taxi_type,
        month=ARGS.month,
    )

    # set pipeline config
    config_pipeline = PipelineConfig(regressor_type=ARGS.regressor_type)

    # train the model pipeline
    train_pipeline(config_train_loader, config_test_loader, config_pipeline)
    return


def main() -> None:
    month = 0
    train_year = 2021
    test_year = 2024
    taxi_type = "green"
    dir_dataset = "dataset_ny_taxi"
    regressor_type = "linear"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train_year",
        default=train_year,
        type=int,
        help="year indicating the dataset to be loaded for train set",
    )
    parser.add_argument(
        "--test_year",
        default=test_year,
        type=int,
        help="year indicating the dataset to be loaded for test set",
    )
    parser.add_argument(
        "--month",
        default=month,
        type=int,
        help="month of the dataset to be loaded (0 for all months data for the given year)",
    )
    parser.add_argument(
        "--taxi_type",
        default=taxi_type,
        type=str,
        help="taxi type of the dataset to be loaded",
    )
    parser.add_argument(
        "--dir_dataset",
        default=dir_dataset,
        type=str,
        help="dir where the dataset needs to be loaded",
    )
    parser.add_argument(
        "--regressor_type",
        default=regressor_type,
        type=str,
        choices=["linear", "ridge", "xgboost"],
        help="type of the regressor model to be used for the model pipeline",
    )
    ARGS, unparsed = parser.parse_known_args()
    trainer(ARGS)
    return


if __name__ == "__main__":
    main()
