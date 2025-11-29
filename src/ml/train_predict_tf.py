# src/ml/train_predict_tf.py
from typing import Literal
import numpy as np
import pandas as pd
from .preprocess import SimpleScaler
from .models_tf import build_univariate_lstm, build_univariate_gru
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

def expanding_window_predict(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    model_type: Literal["lstm", "gru"] = "lstm",
    hidden: int = 32,
    lr: float = 1e-3,
    epochs: int = 5,
    refit_every: int = 1,
    batch_size: int = 8,
    verbose: int = 0,
    seed: int | None = None
):
    """
    FIXED VERSION:
    - train only when needed: at t=1 and every refit_every
    - no useless extra training at t==1 + t==refit simultaneously
    - no training inside t=0
    - added verbose progress
    """

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    T, window, N = X.shape
    preds = np.full((T, N), np.nan, dtype=float)

    for asset in range(N):
        if verbose:
            print(f"\n[asset {asset+1}/{N}]")

        model = None
        scaler = SimpleScaler()

        # start from t = 1 because t = 0 has no past samples
        for t in range(1, T):

            # ---------------------------------------
            # Decide when to train / retrain
            # ---------------------------------------
            need_retrain = (model is None) or (refit_every > 0 and (t % refit_every == 0))

            if need_retrain:
                # training data is everything before t
                X_train = X[:t, :, asset]      # shape (t, window)
                y_train = y[:t, asset]         # shape (t,)

                if verbose:
                    print(f"  t={t}/{T} → training on {len(X_train)} samples")

                # fit scaler on window values (flattened)
                scaler.fit(X_train.reshape(-1))

                # choose model
                if model_type == "lstm":
                    model = build_univariate_lstm(window=window, hidden=hidden, lr=lr)
                else:
                    model = build_univariate_gru(window=window, hidden=hidden, lr=lr)

                # scale training data
                X_scaled = (X_train - scaler.mean) / scaler.std
                X_scaled = X_scaled.reshape((-1, window, 1)).astype(np.float32)

                y_scaled = ((y_train - scaler.mean) / scaler.std).astype(np.float32).reshape(-1, 1)

                # train model
                model.fit(
                    X_scaled, y_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1 if verbose else 0
                )

            # ---------------------------------------
            # Predict for time t
            # ---------------------------------------
            X_last = X[t, :, asset].reshape(1, window, 1).astype(np.float32)
            X_last_scaled = (X_last - scaler.mean) / scaler.std

            pred_scaled = float(model.predict(X_last_scaled, verbose=0)[0, 0])
            preds[t, asset] = scaler.inverse(pred_scaled)

    return preds, dates