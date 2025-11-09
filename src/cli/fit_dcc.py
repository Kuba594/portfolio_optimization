import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.econometric.dcc_garch_r import dcc_roll_cov_series

def save_cov_series(dates, covs, columns, out_path):
    rows = []
    for t, dt in enumerate(dates):
        S = covs[t]
        n = len(columns)
        for i in range(n):
            for j in range(i + 1):
                rows.append((dt, columns[i], columns[j], float(S[i, j])))
    df = pd.DataFrame(rows, columns=["date", "row", "col", "value"])
    df.sort_values(["date", "row", "col"], inplace=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(df.head())
    df.to_parquet(out_path)
    print(f"Saved {len(dates)} forecasts → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="data/processed/daily_returns.parquet")
    ap.add_argument("--out", default="data/processed/dcc_cov_lowertri.parquet")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--refit_every", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    rets = pd.read_parquet(args.returns).dropna()
    print(f"Data shape after dropna: {rets.shape}")
    print(f"Running rolling DCC with window={args.window}, "
          f"refit_every={args.refit_every}, n_jobs={args.n_jobs}")

    dates, covs = dcc_roll_cov_series(
        rets,
        window=args.window,
        refit_every=args.refit_every,
        n_jobs=args.n_jobs
    )

    if len(dates) == 0 or np.all(np.isnan(covs)):
        print("⚠️  DCC roll produced no valid forecasts (check data or parameters).")
    else:
        save_cov_series(dates, covs, rets.columns.tolist(), args.out)

if __name__ == "__main__":
    main()