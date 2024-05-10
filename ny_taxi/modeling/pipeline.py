import os
import numpy as np
import pandas as pd


from typing import Union, Tuple
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge

from ny_taxi.config.config import PipelineConfig


@task
def get_regressor(
    regressor_type: str,
) -> Tuple[Union[LinearRegression, Ridge, XGBRegressor], dict]:
    logger = get_run_logger()
    regressor_params = {}
    if regressor_type == "linear":
        regressor = LinearRegression(n_jobs=-1)
        regressor_params = {"regressor__fit_intercept": [True, False]}
    elif regressor_type == "ridge":
        regressor = Ridge(random_state=41)
        regressor_params = {"regressor__alpha": np.array([0.05, 0.1, 0.5, 1, 3, 5])}
    elif regressor_type == "xgboost":
        regressor = XGBRegressor(random_state=41, n_jobs=-1)
        regressor_params = {
            "regressor__n_estimators": np.array([100]),
            "regressor__max_depth": np.array([20, 30, 40, 50]),
            "regressor__learning_rate": np.array([0.1, 0.5, 1]),
            "regressor__max_leaves": np.array([0, 32, 64, 128]),
            "regressor__reg_lambda": np.array([0.1, 0.5, 1]),
        }
    else:
        logger.info(f"unidentified regressor_type: {regressor_type}")
    return regressor, regressor_params


@flow(validate_parameters=False)
def get_pipeline(config_pipeline: PipelineConfig) -> Tuple[Pipeline, dict]:
    dict_vectorizer = DictVectorizer()
    regressor, pipeline_params = get_regressor(config_pipeline.regressor_type)
    pipeline_reg = Pipeline(
        [("dict_vectorizer", dict_vectorizer), ("regressor", regressor)]
    )
    return pipeline_reg, pipeline_params


@task()
def do_grid_search(pipeline: Pipeline, pipeline_params: dict, X_train_dicts: list, Y_train: np.ndarray) -> GridSearchCV:
    # setup grid search with k-fold cross validation
    k_fold_cv = KFold(n_splits=5, shuffle=True, random_state=7)
    grid_cv = GridSearchCV(
        pipeline,
        pipeline_params,
        scoring="neg_root_mean_squared_error",
        cv=k_fold_cv,
    )

    # train the grid search model
    grid_cv.fit(X_train_dicts, Y_train)
    return grid_cv
