# src/cli/backtest.py
import argparse
from pathlib import Path
import pandas as pd
from src.backtest.engine import backtest_portfolio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="../../data/processed/daily_returns.parquet")
    ap.add_argument("--cov", default="../../data/processed/dcc_cov_lowertri.parquet")
    ap.add_argument("--out", default="../../data/processed/backtest_dcc_minvar.parquet")
    ap.add_argument("--mode", choices=["min_var", "mean_var"], default="min_var")
    ap.add_argument("--risk_aversion", type=float, default=10.0)
    ap.add_argument("--rebalance", default="W-FRI")
    args = ap.parse_args()

    bt = backtest_portfolio(
        returns_path=args.returns,
        cov_path=args.cov,
        mode=args.mode,
        risk_aversion=args.risk_aversion,
        rebalance_freq=args.rebalance,
        long_only=True,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(args.out)
    print(f"Backtest rows: {len(bt)} → {args.out}")

    # quick stats
    r = bt["port_ret"]
    ann = 252
    mu_ann = r.mean() * ann
    sd_ann = r.std() * (ann**0.5)
    sharpe = mu_ann / sd_ann if sd_ann > 0 else float("nan")
    wealth = (1 + r).cumprod()
    max_dd = ((wealth / wealth.cummax()) - 1).min()

    print(f"Ann. return: {mu_ann:.2%}")
    print(f"Ann. vol:    {sd_ann:.2%}")
    print(f"Sharpe:      {sharpe:.2f}")
    print(f"Max drawdown:{max_dd:.2%}")

if __name__ == "__main__":
    main()