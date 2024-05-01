import os
from typing import List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataDownloaderConfig:
    dir_dataset: str
    taxi_type: str = field(default="green")  # ["green", "yellow", "fhv", "fhvhv"]
    file_base_url: str = field(
        default="https://d37ci6vzurychx.cloudfront.net/trip-data"
    )
    year: int = field(default=2021)
