# src/cli/ml_run.py
import argparse
import pandas as pd
from src.ml.preprocess import load_returns, make_multivariate_windows
from src.ml.train_predict_tf import expanding_window_predict
from src.ml.cov_from_preds import rolling_cov_from_preds, save_cov_long

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="../../data/processed/daily_returns.parquet")
    ap.add_argument("--tickers", nargs="+", required=False)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--refit_every", type=int, default=50)
    ap.add_argument("--cov_out", default="../../data/processed/ml_cov_lowertri.parquet")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    returns = load_returns(args.returns)
    if args.tickers:
        returns = returns[args.tickers]
    tickers = returns.columns.tolist()

    X, y, dates = make_multivariate_windows(returns, window=args.window)
    print("Built dataset:", X.shape, y.shape, "dates:", len(dates))

    preds, pred_dates = expanding_window_predict(
        X, y, dates,
        model_type=args.model,
        hidden=args.hidden,
        lr=args.lr,
        epochs=args.epochs,
        refit_every=args.refit_every,
        batch_size=args.batch_size,
        verbose=1,
        seed=91
    )

    covs, cov_dates = rolling_cov_from_preds(preds, pred_dates, tickers, window=args.window)
    print("Built ML covariances:", covs.shape, "dates:", len(cov_dates))

    df = save_cov_long(covs, cov_dates, tickers, args.cov_out)
    print("Saved ML covariances to:", args.cov_out)
    print(df.head())

if __name__ == "__main__":
    main()
