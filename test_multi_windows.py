import pandas as pd
import numpy as np
from src.ml.preprocess import load_returns, make_multivariate_windows, SimpleScaler

# Fake returns data for 3 assets
idx = pd.date_range("2020-01-01", periods=10)
df = pd.DataFrame({
    "A": np.random.randn(10),
    "B": np.random.randn(10),
    "C": np.random.randn(10),
}, index=idx)

window = 3

X, y, dates = make_multivariate_windows(df, window)

print("X shape:", X.shape)   # expected (10-3, 3, 3) = (7, 3, 3)
print("y shape:", y.shape)   # expected (7, 3)
print("dates:", dates[:5])