import os
import numpy as np
import urllib.request

from prefect import task, flow, get_run_logger

from ny_taxi.config.config import DataDownloaderConfig


@task
def create_dir(dir_dataset: str) -> None:
    logger = get_run_logger()
    if not os.path.isdir(dir_dataset):
        os.makedirs(dir_dataset)
        logger.info(f"created dataset directory: {dir_dataset}")
    return


@task(retries=3, retry_delay_seconds=2)
def download(config_downloader: DataDownloaderConfig) -> None:
    logger = get_run_logger()
    logger.info(f"NY Taxi dataset downloader config:{config_downloader}")

    all_months = np.arange(1, 13)

    for month in all_months:
        file_name = f"{config_downloader.taxi_type}_tripdata_{config_downloader.year}-{month:02d}.parquet"
        file_url = f"{config_downloader.file_base_url}/{file_name}"

        try:
            urllib.request.urlretrieve(
                file_url, os.path.join(config_downloader.dir_dataset, file_name)
            )
            logger.info(f"downloaded file: {file_url}")
        except:
            logger.info(f"file url not found: {file_url}")

    return


@flow(validate_parameters=False)
def downloader(config_downloader: DataDownloaderConfig) -> None:
    """
    for now the validate_parameters is set to False for the prefect flow
    since there is a bug in the pydantic package
    """

    # create dataset directory
    create_dir(config_downloader.dir_dataset)

    # download dataset parquet files
    download(config_downloader)
    return
