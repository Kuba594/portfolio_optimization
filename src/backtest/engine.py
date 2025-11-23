# src/backtest/engine.py
import numpy as np
import pandas as pd
from src.utils.cov_io import reconstruct_cov_for_date, get_cov_dates
from src.optimize.meanvar import solve_min_var, solve_mean_var

def backtest_portfolio(returns_path: str,
                       cov_path: str,
                       mode: str = "min_var",
                       risk_aversion: float = 10.0,
                       rebalance_freq: str = "W-FRI",
                       long_only: bool = True) -> pd.DataFrame:
    """
    Generic backtest:
    - mode = 'min_var' or 'mean_var'
    - cov_path: long lower-tri covariances (DCC or ML)
    """
    rets = pd.read_parquet(returns_path).dropna()
    rets.index = pd.to_datetime(rets.index).tz_localize(None)

    cov_df = pd.read_parquet(cov_path)
    cov_df["date"] = pd.to_datetime(cov_df["date"]).dt.tz_localize(None)

    tickers = rets.columns.tolist()
    cov_dates = get_cov_dates(cov_df)

    # Align cov dates to return index
    cov_dates = cov_dates.intersection(rets.index)

    # Choose rebalance dates
    if rebalance_freq is not None:
        # resample cov dates to requested frequency on returns index
        sched = (
            cov_dates.to_series()
            .resample(rebalance_freq)
            .last()
            .dropna()
            .index
        )
        rebalance_dates = sched.intersection(cov_dates)
    else:
        rebalance_dates = cov_dates

    # Simple constant μ: in-sample average (can be replaced by rolling/ML later)
    mu_const = rets.mean().values  # daily
    # You can annualize later when interpreting risk/return.

    rows = []
    w = np.ones(len(tickers)) / len(tickers)  # start equal-weighted

    for d in rebalance_dates:
        # 1) Build Σ̂_t from cov_df
        Sigma = reconstruct_cov_for_date(cov_df, d, tickers)

        # 2) Choose weights based on selected mode
        if mode == "min_var":
            w = solve_min_var(Sigma, long_only=long_only)
        elif mode == "mean_var":
            w = solve_mean_var(Sigma, mu_const, risk_aversion=risk_aversion, long_only=long_only)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 3) Apply weights to next-day returns
        try:
            idx = rets.index.get_loc(d)
        except KeyError:
            # if date not exactly in index, skip
            continue
        if idx + 1 >= len(rets.index):
            break

        next_date = rets.index[idx + 1]
        r_next = rets.iloc[idx + 1].values
        port_ret = float(np.nansum(w * r_next))

        rows.append((next_date, port_ret, *w))

    cols = ["date", "port_ret"] + [f"w_{t}" for t in tickers]
    out = pd.DataFrame(rows, columns=cols).set_index("date")
    return out