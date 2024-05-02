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


@dataclass(frozen=True)
class DataLoaderConfig:
    dir_dataset: str
    taxi_type: str = field(default="green")  # ["green", "yellow", "fhv", "fhvhv"]
    year: int = field(default=2021)
    month: int = field(default=0)


@dataclass(frozen=True)
class FeatureTargetConfig:
    categorical: List[str] = field(
        default_factory=[
            "PULocationID",
            "DOLocationID",
            "VendorID",
            "payment_type",
            "trip_type",
        ]
    )
    numerical: List[str] = field(default_factory=["trip_distance"])
    target: str = field(default="duration")


@dataclass(frozen=True)
class PipelineConfig:
    regressor_type: str = field(
        default="linear"
    )  # ["linear", "ridge", "ransac", "light_gbm", "random_forest"]
