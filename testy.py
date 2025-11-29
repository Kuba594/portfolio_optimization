import numpy as np
import pandas as pd

# Load saved DCC covariances
df = pd.read_parquet("data/processed/dcc_cov_lowertri.parquet")

tickers = df["row"].unique().tolist()
n = len(tickers)
dates = sorted(df["date"].unique())

def reconstruct_cov(df, date, tickers):
    """Reconstruct full symmetric covariance matrix for a given date."""
    sub = df[df["date"] == date]
    S = np.zeros((n, n))
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            v = sub.loc[(sub["row"] == ti) & (sub["col"] == tj), "value"]
            if not v.empty:
                S[i, j] = v.values[0]
                S[j, i] = v.values[0]
    return S

# --- Loop over all dates and check ---
results = []
for d in dates:
    S = reconstruct_cov(df, d, tickers)
    sym_ok = np.allclose(S, S.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(S)
    pd_ok = np.all(eigvals > 0)
    results.append((d, sym_ok, pd_ok, eigvals.min(), eigvals.max()))

check_df = pd.DataFrame(results, columns=["date", "symmetric", "pos_def", "min_eig", "max_eig"])

# --- Summary ---
print(check_df.head())
print("\nSymmetric OK:", check_df["symmetric"].all())
print("Positive-definite OK:", check_df["pos_def"].all())
print("\nFraction of valid matrices:",
      f"{check_df['pos_def'].mean():.2%} positive-definite out of {len(check_df)}")

# --- Optional: flag problematic dates ---
bad = check_df.loc[~check_df["pos_def"] | ~check_df["symmetric"]]
if not bad.empty:
    print("\n⚠️  Problematic dates:")
    print(bad.head())
else:
    print("\n✅ All covariance matrices symmetric and positive-definite.")