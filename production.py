import os
import logging
import argparse


from ny_taxi.config.config import ProductionConfig
from ny_taxi.utils.production import list_experiments, list_registered_models


def production(ARGS: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    config_production = ProductionConfig(
        mlflow_tracking_uri=ARGS.mlflow_tracking_uri,
        experiment_name=ARGS.experiment_name,
    )
    list_experiments(config_production)
    list_registered_models(config_production)
    return


def main() -> None:
    mlflow_tracking_uri = "sqlite:///mlruns.db"
    experiment_name = "ny_taxi"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default=mlflow_tracking_uri,
        type=str,
        help="mlflow tracking uri",
    )
    parser.add_argument(
        "--experiment_name",
        default=experiment_name,
        type=str,
        help="mlflow experiment name",
    )
    ARGS, unparsed = parser.parse_known_args()
    production(ARGS)
    return


if __name__ == "__main__":
    main()
