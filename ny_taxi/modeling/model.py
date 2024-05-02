import os
import sys
import numpy as np

from typing import Union, Tuple
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge


def get_regressor(
    regressor_type: str,
) -> Tuple[Union[LinearRegression, Ridge, XGBRegressor], dict]:
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
            "regressor__n_estimators": np.array([100, 200]),
            "regressor__max_depth": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
            "regressor__learning_rate": np.array([0.1, 0.5, 1, 2]),
            "regressor__max_leaves": np.array([0, 32, 64, 128, 256, 512]),
            "regressor__reg_lambda": np.array([0.1, 0.5, 1, 2]),
        }
    else:
        logging.info(f"unidentified regressor_type: {regressor_type}")
    return regressor, regressor_params
