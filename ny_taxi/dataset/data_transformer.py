import os
import numpy as np
import pandas as pd

from ny_taxi.config.config import FeatureTargetConfig


def transform(df_ny_taxi: pd.DataFrame) -> pd.DataFrame:
    # add a new feature duration = dropoff time - pickup time
    df_ny_taxi["duration"] = (
        df_ny_taxi.lpep_dropoff_datetime - df_ny_taxi.lpep_pickup_datetime
    )
    # convert to mins
    df_ny_taxi.duration = df_ny_taxi.duration.apply(lambda td: td.total_seconds() / 60)
    # choose data that is appropriate
    df_ny_taxi = df_ny_taxi[(df_ny_taxi.duration >= 1) & (df_ny_taxi.duration <= 60)]
    # replace unknown payment type with 1000 to indicate unknown class
    df_ny_taxi["payment_type"] = df_ny_taxi["payment_type"].replace(np.NaN, 1000)
    # replace unknown trip type with 1000 to indicate unknown class
    df_ny_taxi["trip_type"] = df_ny_taxi["trip_type"].replace(np.NaN, 1000)

    feature_target_config = FeatureTargetConfig()

    # list of categorical features
    categorical = feature_target_config.categorical

    # list of numerical features
    numerical = feature_target_config.numerical

    # convert categorical features to strings
    df_ny_taxi[categorical] = df_ny_taxi[categorical].astype(str)

    return df_ny_taxi
