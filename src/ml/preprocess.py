# src/ml/preprocess.py
from typing import List, Tuple
import numpy as np
import pandas as pd

def load_returns(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def make_multivariate_windows(returns: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build X, y and target dates for multivariate dataset used by per-asset univariate models.
    X: (T-window, window, N) - past windows
    y: (T-window, N) - next-day returns (targets)
    dates: index aligned with each y (i.e. date of the target return)
    """
    rets = returns.values
    T, N = rets.shape
    X_list, y_list, dates = [], [], []
    for t in range(window, T):
        X_list.append(rets[t-window:t])   # window x N
        y_list.append(rets[t])            # N (target for day t)
        dates.append(returns.index[t])
    if len(X_list) == 0:
        return np.empty((0, window, N)), np.empty((0, N)), pd.DatetimeIndex([])
    X = np.stack(X_list)   # (T-window, window, N)
    y = np.stack(y_list)   # (T-window, N)
    return X, y, pd.DatetimeIndex(dates)

class SimpleScaler:
    """Per-asset z-score scaler using numpy arrays (fit on training data)."""
    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, arr: np.ndarray):
        # arr shape: (n_samples, window) or (n_values,)
        self.mean = arr.mean()
        self.std = arr.std() + self.eps

    def transform(self, arr: np.ndarray):
        return (arr - self.mean) / self.std

    def inverse(self, val: float):
        return val * self.std + self.mean
