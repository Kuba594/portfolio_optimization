# src/utils/cov_io.py
import numpy as np
import pandas as pd

def load_cov_long(path: str) -> pd.DataFrame:
    """Load covariance series in long lower-triangular format."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "row", "col"])

def reconstruct_cov_for_date(df: pd.DataFrame, date, tickers: list[str]) -> np.ndarray:
    """Reconstruct full symmetric covariance matrix Σ for a given date."""
    sub = df[df["date"] == pd.Timestamp(date)]
    n = len(tickers)
    S = np.zeros((n, n), dtype=float)

    # Build a lookup dict for speed
    lut = {(r, c): v for r, c, v in zip(sub["row"], sub["col"], sub["value"])}

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            # lower-tri was stored as (max_index, min_index)
            key = (ti, tj) if (ti, tj) in lut else (tj, ti)
            v = lut.get(key, np.nan)
            S[i, j] = v

    # small ridge for numerical stability
    S = S + 1e-8 * np.eye(n)
    return S

def get_cov_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(df["date"].unique()).sort_values()