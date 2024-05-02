import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

from ny_taxi.config.config import PipelineConfig
from ny_taxi.modeling.model import get_regressor


def get_pipeline(config_pipeline: PipelineConfig) -> Tuple[Pipeline, dict]:
    dict_vectorizer = DictVectorizer()
    regressor, pipeline_params = get_regressor(config_pipeline.regressor_type)
    pipeline_reg = Pipeline(
        [("dict_vectorizer", dict_vectorizer), ("regressor", regressor)]
    )
    return pipeline_reg, pipeline_params
