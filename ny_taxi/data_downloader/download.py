import os
import urllib.request
import logging
import numpy as np

from ny_taxi.config.data_downloader import DataDownloaderConfig


def download(config_downloader: DataDownloaderConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"NY Taxi dataset downloader config:{config_downloader}")

    if not os.path.isdir(config_downloader.dir_dataset):
        logging.info(f"created dataset directory: {config_downloader.dir_dataset}")
        os.makedirs(config_downloader.dir_dataset)

    all_months = np.arange(1, 13)

    for month in all_months:
        file_name = f"{config_downloader.taxi_type}_tripdata_{config_downloader.year}-{month:02d}.parquet"
        file_url = f"{config_downloader.file_base_url}/{file_name}"

        try:
            urllib.request.urlretrieve(
                file_url, os.path.join(config_downloader.dir_dataset, file_name)
            )
            logging.info(f"downloaded file: {file_url}")
        except:
            logging.info(f"file url not found: {url}")

    return
