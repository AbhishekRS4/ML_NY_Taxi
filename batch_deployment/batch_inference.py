import os
import mlflow

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from prefect import flow, task, get_run_logger


@task()
def create_target_dir(dir_target: str) -> None:
    if not os.path.isdir(dir_target):
        os.makedirs(dir_target)
    return


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
    Y_test_preds: np.ndarray, file_target: str, df_rides: pd.DataFrame
) -> None:
    df_rides["predicted_duration"] = Y_test_preds
    logger = get_run_logger()
    logger.info(df_rides.head())
    df_rides.to_parquet(file_target, index=False)
    return


@flow(validate_parameters=False)
def batch_inference() -> None:
    logger = get_run_logger()
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    model_for_inference = "./model_for_prod/"
    file_test_rides = "../dataset_ny_taxi/green/2024/green_tripdata_2024-01.parquet"
    dir_target = "../output/"
    file_target = os.path.join(dir_target, file_test_rides.split("/")[-1])

    create_target_dir(dir_target)
    model = load_model(model_for_inference)
    df_rides = load_data(file_test_rides)
    df_rides = prepare_features(df_rides)
    X_test_dicts = df_rides[categorical + numerical].to_dict(orient="records")
    Y_test_preds = get_prediction(model, X_test_dicts)
    save_predictions(Y_test_preds, file_target, df_rides)
    return


if __name__ == "__main__":
    batch_inference()
