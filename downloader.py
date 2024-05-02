import os
import argparse

from ny_taxi.data_downloader.download import download
from ny_taxi.config.config import DataDownloaderConfig


def main() -> None:
    year = 2021
    taxi_type = "green"
    dir_dataset = "dataset_ny_taxi"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--year",
        default=year,
        type=int,
        help="year of the dataset to be downloaded",
    )
    parser.add_argument(
        "--taxi_type",
        default=taxi_type,
        type=str,
        help="taxi type of the dataset to be downloaded",
    )
    parser.add_argument(
        "--dir_dataset",
        default=dir_dataset,
        type=str,
        help="dir where the dataset needs to be downloaded",
    )
    ARGS, unparsed = parser.parse_known_args()
    config_downloader = DataDownloaderConfig(
        dir_dataset=os.path.join(ARGS.dir_dataset, ARGS.taxi_type, str(ARGS.year)),
        year=ARGS.year,
        taxi_type=ARGS.taxi_type,
    )
    download(config_downloader)

    return


if __name__ == "__main__":
    main()
