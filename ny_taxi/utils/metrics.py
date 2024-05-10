import os
import numpy as np

from typing import Tuple
from prefect import task
from sklearn.metrics import r2_score, mean_squared_error


@task()
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2
