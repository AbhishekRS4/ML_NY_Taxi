import os
import numpy as np

from typing import Tuple
from sklearn.metrics import r2_score, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2
