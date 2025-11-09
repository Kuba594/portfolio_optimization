import argparse, yaml
from pathlib import Path
import pandas as pd, numpy as np

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_prices(tickers, start, end, raw_dir="data/raw"):
    frames = []
    for t in tickers:
        csv_path = Path(raw_dir) / f"{t}.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["Date"])
        else:
            raise FileNotFoundError(f"No data file found for {t} in {raw_dir}")

        df = df.rename(columns={c.lower(): c for c in df.columns})
        if "Adj Close" in df.columns:
            df = df[["Date", "Adj Close"]]
        elif "Adj close" in df.columns:
            df = df[["Date", "Adj close"]].rename(columns={"Adj close": "Adj Close"})
        else:
            raise KeyError(f"{t}: expected column 'Adj Close' in file")

        df = df.rename(columns={"Adj Close": t})
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.set_index("Date").sort_index()
        df.index = df.index.tz_convert(None)

        df = df.loc[start:end]
        df.rename(columns={"Adj Close": t}, inplace=True)
        frames.append(df)

    prices = pd.concat(frames, axis=1).dropna(how="all")
    return prices

def compute_log_returns(prices):
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    tickers = cfg["universe"]
    start, end = cfg["start"], cfg["end"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    prices = load_prices(tickers, start, end)
    prices.index = prices.index.date

    returns = compute_log_returns(prices)
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cfg["price_path"])
    returns.to_parquet(cfg["returns_path"])

    #log massage
    print(f"Saved {len(prices.columns)} tickers from {prices.index[0]} "
          f"to {prices.index[-1]}")
    print(f"Prices → {cfg['price_path']}")
    print(f"Returns → {cfg['returns_path']}")

if __name__ == "__main__":
    main()