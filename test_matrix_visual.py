import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.cov_io import load_cov_long, reconstruct_cov_for_date, get_cov_dates

# 1. Load DCC covariance
cov_df = load_cov_long("data/processed/dcc_cov_lowertri.parquet")

print("Loaded cov dataframe:")
print(cov_df.head())

# 2. Extract available dates
dates = get_cov_dates(cov_df)
print(f"\nAvailable covariance dates: {len(dates)}")
print("First:", dates[0])
print("Last: ", dates[-1])

# 3. Pick a sample date (middle of the dataset)
date = dates[len(dates) // 2]
print(f"\nUsing date: {date}")

# 4. Get list of tickers from rows/cols
tickers = sorted(cov_df["row"].unique().tolist())
print(f"Tickers: {tickers}")

# 5. Reconstruct covariance matrix
Sigma = reconstruct_cov_for_date(cov_df, date, tickers)

print("\nCovariance matrix shape:", Sigma.shape)

# 6. Symmetry check
print("Symmetric:", np.allclose(Sigma, Sigma.T, atol=1e-10))

# 7. Positive-definite check
eigvals = np.linalg.eigvalsh(Sigma)
print("Min eigenvalue:", eigvals.min())
print("Max eigenvalue:", eigvals.max())
print("Positive definite:", np.all(eigvals > 0))

# 8. Heatmap visualization
plt.figure(figsize=(7, 6))
sns.heatmap(Sigma, annot=False, cmap="viridis",
            xticklabels=tickers, yticklabels=tickers)
plt.title(f"DCC-GARCH Covariance Matrix\n{date.date()}")
plt.tight_layout()
plt.show()