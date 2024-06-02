import os
import numpy as np
import pandas as pd

from typing import Tuple
from prefect import task, get_run_logger

from ny_taxi.config.config import DataLoaderConfig


@task(retries=3, retry_delay_seconds=2)
def data_loader(
    config_dataloader: DataLoaderConfig,
) -> Tuple[pd.DataFrame, DataLoaderConfig]:
    dir_dataset = os.path.join(
        config_dataloader.dir_dataset,
        config_dataloader.taxi_type,
        str(config_dataloader.year),
    )
    if config_dataloader.month == 0:
        # load all month data
        list_parquet_files = [
            f for f in sorted(os.listdir(dir_dataset)) if f.endswith(".parquet")
        ]
    else:
        list_parquet_files = [
            f
            for f in os.listdir(dir_dataset)
            if f.endswith(f"{config_dataloader.month:02d}.parquet")
        ]

    df = None
    all_files = ""

    for file_parq in list_parquet_files:
        if df is not None:
            df_temp = pd.read_parquet(os.path.join(dir_dataset, file_parq))
            df = pd.concat([df, df_temp], sort=False)
            all_files = all_files + "__" + file_parq
        else:
            df = pd.read_parquet(os.path.join(dir_dataset, file_parq))
            all_files = file_parq

    config_dataloader.all_files = all_files
    # return config_dataloader since it has been updated when using prefect worlflows
    return df, config_dataloader
