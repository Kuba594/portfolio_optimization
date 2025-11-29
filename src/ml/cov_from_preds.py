# src/ml/cov_from_preds.py
import numpy as np
import pandas as pd
from pathlib import Path

def rolling_cov_from_preds(preds: np.ndarray, dates: pd.DatetimeIndex, tickers: list[str], window: int = 60):
    """
    preds: (T, N) predicted returns aligned with `dates` (dates[t] is the target day for preds[t])
    returns:
      covs: list of covariance matrices (M x N x N) for times t = window..T-1
      cov_dates: corresponding dates (pd.DatetimeIndex) of length M (these are dates[t] for which cov is computed)
    """
    T, N = preds.shape
    cov_list = []
    cov_dates = []
    for t in range(window, T):
        seg = preds[t-window:t, :]    # window x N
        if np.isnan(seg).any():
            # skip windows with NaN predictions (may happen near the start)
            continue
        cov = np.cov(seg.T)          # N x N
        cov_list.append(cov)
        cov_dates.append(dates[t])
    if len(cov_list) == 0:
        return np.empty((0, N, N)), pd.DatetimeIndex([])

    return np.stack(cov_list), pd.DatetimeIndex(cov_dates)

def save_cov_long(covs: np.ndarray, cov_dates: pd.DatetimeIndex, tickers: list[str], out_path: str):
    """
    Convert covs (M x N x N) into long lower-tri DataFrame and save as parquet.
    Columns: date, row, col, value (lower-tri including diag)
    """
    rows = []
    M, N, _ = covs.shape
    for t in range(M):
        d = cov_dates[t]
        S = covs[t]
        for i in range(N):
            for j in range(i+1):
                rows.append((d, tickers[i], tickers[j], float(S[i, j])))
    df = pd.DataFrame(rows, columns=["date", "row", "col", "value"])
    df.sort_values(["date", "row", "col"], inplace=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df
