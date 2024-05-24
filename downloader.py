import os
import toml

from ny_taxi.config.config import DataDownloaderConfig
from ny_taxi.data_downloader.download import downloader


def main() -> None:
    file_toml = "config.toml"
    config = toml.load(file_toml)
    year = config["downloader"]["year"]
    taxi_type = config["downloader"]["taxi_type"]
    dir_dataset = config["downloader"]["dir_dataset"]

    config_downloader = DataDownloaderConfig(
        dir_dataset=os.path.join(dir_dataset, taxi_type, str(year)),
        year=year,
        taxi_type=taxi_type,
    )
    downloader(config_downloader)

    return


if __name__ == "__main__":
    main()
