import os
import sys
import numpy as np

from typing import Union, Tuple
from sklearn.linear_model import LinearRegression, Ridge


def get_regressor(
    regressor_type: str,
) -> Tuple[Union[LinearRegression, Ridge], dict]:
    regressor_params = {}
    if regressor_type == "linear":
        regressor = LinearRegression(n_jobs=-1)
        regressor_params = {"regressor__fit_intercept": [True, False]}
    elif regressor_type == "ridge":
        regressor = Ridge(random_state=41)
        regressor_params = {"regressor__alpha": np.array([0.05, 0.1, 0.5, 1, 3, 5])}
    else:
        logging.info(f"unidentified regressor_type: {regressor_type}")
    return regressor, regressor_params
