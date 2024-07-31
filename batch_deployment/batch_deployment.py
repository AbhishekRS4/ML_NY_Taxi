import os
import toml
import uuid
import mlflow

import numpy as np
import pandas as pd

from typing import List
from sklearn.pipeline import Pipeline
from prefect import flow, task, get_run_logger


@task()
def create_target_dir(dir_target: str) -> None:
    if not os.path.isdir(dir_target):
        os.makedirs(dir_target)
    return


@task()
def get_ride_ids(num_rides: int) -> List:
    ride_ids = []
    for i in range(num_rides):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


@task(retries=3, retry_delay_seconds=2)
def load_model(model_for_inference: str) -> Pipeline:
    model = mlflow.sklearn.load_model(model_for_inference)
    return model


@task(retries=3, retry_delay_seconds=2)
def load_data(file_test_rides: str) -> pd.DataFrame:
    df_rides = pd.read_parquet(file_test_rides)
    return df_rides


@task()
def prepare_features(df_rides: pd.DataFrame) -> pd.DataFrame:
    # add duration
    df_rides["duration"] = (
        df_rides.lpep_dropoff_datetime - df_rides.lpep_pickup_datetime
    )

    # convert duration to mins
    df_rides.duration = df_rides.duration.apply(lambda td: td.total_seconds() / 60)

    # create feature for input
    df_rides["PU_DO"] = (
        df_rides["PULocationID"].astype(str)
        + "_"
        + df_rides["DOLocationID"].astype(str)
    )
    return df_rides


@task()
def get_prediction(model: Pipeline, ride_features: list) -> np.ndarray:
    ride_duration_preds = model.predict(ride_features)
    return ride_duration_preds


@task(retries=3, retry_delay_seconds=2)
def save_predictions(
    Y_test_preds: np.ndarray, file_target: str, df_rides: pd.DataFrame, ride_ids: List
) -> None:
    df_rides["predicted_duration"] = Y_test_preds
    df_rides["ride_ids"] = ride_ids
    logger = get_run_logger()
    logger.info(df_rides.head())
    df_rides.to_parquet(file_target, index=False)
    return


@flow(validate_parameters=False)
def batch_deployment(
    model_for_inference: str,
    dir_dataset: str,
    dir_target: str,
    year: int,
    month: int,
    taxi_type: str,
) -> None:
    logger = get_run_logger()
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    file_test_rides = f"{dir_dataset}/{taxi_type}/{year}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"
    dir_target = os.path.join(
        dir_target, f"taxi_type={taxi_type}", f"year={year}", f"month={month}"
    )
    file_target = os.path.join(dir_target, file_test_rides.split("/")[-1])

    if not dir_target.startswith("s3://"):
        create_target_dir(dir_target)
    model = load_model(model_for_inference)
    df_rides = load_data(file_test_rides)
    ride_ids = get_ride_ids(df_rides.shape[0])
    df_rides = prepare_features(df_rides)
    X_test_dicts = df_rides[categorical + numerical].to_dict(orient="records")
    Y_test_preds = get_prediction(model, X_test_dicts)
    save_predictions(Y_test_preds, file_target, df_rides, ride_ids)
    return


def main() -> None:
    file_toml = "batch_deployment_config.toml"
    config = toml.load(file_toml)
    model_for_inference = config["batch_deployment"]["model_for_inference"]
    dir_dataset = config["batch_deployment"]["dir_dataset"]
    dir_target = config["batch_deployment"]["dir_target"]
    year = config["batch_deployment"]["year"]
    month = config["batch_deployment"]["month"]
    taxi_type = config["batch_deployment"]["taxi_type"]
    batch_deployment(
        model_for_inference, dir_dataset, dir_target, year, month, taxi_type
    )
    return


if __name__ == "__main__":
    main()
