#!/usr/bin/env python3
import argparse
import numpy as np
import minnt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Parse arguments
#GRU: 64, 128, 2, 0.3, 50, 0.0005, 0.001
#LSTM 64, 128, 2, 0.3, 50,  0.001, 0.005 x2

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--learning_rate_decay", default="cosine", choices=["cosine", "linear", "none"])
parser.add_argument("--seq_len", default=22, type=int)
parser.add_argument("--model_type", default="gru", choices=["lstm", "gru"])
parser.add_argument("--weight_decay", default=1e-3, type=float)
parser.add_argument("--port_weight", default=0, type=float,
                    help="Weight of the portfolio-variance term in combined_loss.")
parser.add_argument("--ridge", default=5e-4, type=float,
                    help="Relative ridge added to Sigma before solving for min-var weights.")


parser.add_argument("--cv_splits", default=5, type=int)
parser.add_argument("--cv_val_size", default=100, type=int)
parser.add_argument("--val_method", default="dev", choices=["cv", "dev"])
parser.add_argument("--save_predictions", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=1, type=int)


class Model(minnt.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        input_size = 36
        self.n = 8

        if args.model_type == "lstm":
            self.rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                batch_first=True,
                dropout=args.dropout if args.num_layers > 1 else 0.0,
            )
        else:
            self.rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                batch_first=True,
                dropout=args.dropout if args.num_layers > 1 else 0.0,
            )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_size, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.head(out)


class TimeSeriesDataset(Dataset):
    def __init__(self, X_data, Sigma_data, seq_len):
        self.X = torch.as_tensor(np.asarray(X_data), dtype=torch.float32)
        self.Sigma = torch.as_tensor(np.asarray(Sigma_data), dtype=torch.float32)
        self.seq_len = seq_len
        assert len(self.X) == len(self.Sigma), "equal length."

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        if idx + self.seq_len >= len(self.Sigma):
            raise IndexError
        return self.X[idx:idx + self.seq_len], self.Sigma[idx + self.seq_len]

_TRIL_IDX_CACHE = {}

def _tri_embed(n, device, dtype):
    key = (n, str(device), str(dtype))
    if key not in _TRIL_IDX_CACHE:
        m = n * (n + 1) // 2
        M = torch.zeros(n, n, m)
        k = 0
        for i in range(n):
            for j in range(i + 1):
                M[i, j, k] = 1.0
                k += 1
        _TRIL_IDX_CACHE[key] = M.to(device=device, dtype=dtype)
    return _TRIL_IDX_CACHE[key]


def _vec_to_lower_tri(vec, n):
    M = _tri_embed(n, vec.device, vec.dtype)
    return torch.einsum("ijk,bk->bij", M, vec)


def _lower_tri_to_vec(L, n):
    M = _tri_embed(n, L.device, L.dtype)
    return torch.einsum("ijk,bij->bk", M, L)


def vec_to_cov(vec, n=8, eig_floor=0.1):
    L = _vec_to_lower_tri(vec, n)
    eye = torch.eye(n, device=L.device, dtype=L.dtype)
    Sigma = L @ L.transpose(-1, -2)
    return Sigma + eig_floor * eye


def vec_to_sym(vec, n=8):

    L = _vec_to_lower_tri(vec, n)
    eye = torch.eye(n, device=L.device, dtype=L.dtype)
    return L + L.transpose(-1, -2) - (L * eye)

def min_var_weights(Sigma, ridge=1e-6):
    B, n, _ = Sigma.shape
    eye = torch.eye(n, device=Sigma.device, dtype=Sigma.dtype)
    diag_mean = (Sigma * eye).sum(dim=(1, 2)) / n
    diag_mean = diag_mean.view(B, 1, 1)

    Sigma = Sigma + ridge * diag_mean * eye
    ones = torch.ones(B, n, 1, device=Sigma.device, dtype=Sigma.dtype)

    x = torch.inverse(Sigma) @ ones
    w = x / x.sum(dim=1, keepdim=True)
    return w.squeeze(-1)

def make_combined_loss(port_weight=0.1, ridge=1e-4, n=8,
                       ):
    def combined_loss(pred_L, target_cov_vec):
        Sigma_pred = vec_to_cov(pred_L, n)
        Sigma_true = vec_to_sym(target_cov_vec, n)
        cov_loss = F.mse_loss(Sigma_pred, Sigma_true)

        if port_weight != 0:
            w_pred = min_var_weights(Sigma_pred, ridge=ridge).unsqueeze(1)
            port_var = torch.bmm(torch.bmm(w_pred, Sigma_true), w_pred.transpose(1, 2))
            port_loss = port_var.reshape(-1).mean()
        else:
            port_loss = 0.0
        return cov_loss  + port_weight * port_loss

    return combined_loss



def walk_forward_cv(X_raw, Sigma_raw, n_splits=5, val_size=200, seq_len=22, train_end=2792):
    fold_size = (train_end - val_size) // n_splits

    folds = []
    for i in range(n_splits):
        train_idx_end = fold_size * (i + 1)
        val_idx_start = train_idx_end - seq_len
        val_idx_end = train_idx_end + val_size

        fold_mean = X_raw[:train_idx_end].mean(axis=0)
        fold_std = X_raw[:train_idx_end].std(axis=0) + 1e-8

        fold_train_X = (X_raw[:train_idx_end] - fold_mean) / fold_std
        fold_val_X = (X_raw[val_idx_start:val_idx_end] - fold_mean) / fold_std

        fold_train_S = Sigma_raw[:train_idx_end]
        fold_val_S = Sigma_raw[val_idx_start:val_idx_end]

        folds.append((fold_train_X, fold_train_S, fold_val_X, fold_val_S))

    return folds

def make_optimizer(model, args):
    dictionary_bias   = {"params": [], "weight_decay": 0.0}
    dictionary_weight = {"params": [], "weight_decay": args.weight_decay}
    for name, param in model.named_parameters():
        if "bias" in name or param.ndim == 1:
            dictionary_bias["params"].append(param)
        else:
            dictionary_weight["params"].append(param)
    optimizer = torch.optim.AdamW(
        [dictionary_bias, dictionary_weight], lr=args.learning_rate
    )

    # clip gradients to max-norm 1.0 right before each optimizer step
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    def _clip_hook(opt, args_, kwargs_):
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

    optimizer.register_step_pre_hook(_clip_hook)

    return optimizer

class CovMSE(minnt.metrics.MeanSquaredError):
    def __init__(self, n=8):
        super().__init__()
        self.n = n

    def update(self, y_pred, y_true):
        Sigma_pred = vec_to_cov(y_pred, self.n)
        Sigma_true = vec_to_sym(y_true, self.n)
        super().update(Sigma_pred, Sigma_true)


def _scale_sanity_check(model, loader, label="train"):
    """Print Σ-diagonal scale of predictions vs targets on one batch.

    Run *before* training to verify that the diagonal-bias init lands
    Σ_pred in the same order of magnitude as Σ_true.  If these differ
    by more than ~5x you should re-tune `--diag_bias_init`.
    """
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(loader))
        device = next(model.parameters()).device
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        Sigma_pred = vec_to_cov(pred)
        Sigma_true = vec_to_sym(yb)
        pred_diag = Sigma_pred.diagonal(dim1=1, dim2=2).mean().item()
        true_diag = Sigma_true.diagonal(dim1=1, dim2=2).mean().item()
        print(f"[scale-check/{label}] Σ_pred diag mean = {pred_diag:.6e}   "
              f"Σ_true diag mean = {true_diag:.6e}   "
              f"ratio = {pred_diag / max(true_diag, 1e-12):.3f}")
    model.train()

def _loss_scale_check(model, loader, args, label="train"):
    """Print the natural scales of cov_loss and port_loss on one batch."""
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(loader))
        device = next(model.parameters()).device
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        Sigma_pred = vec_to_cov(pred)
        Sigma_true = vec_to_sym(yb)
        cov_loss = F.mse_loss(Sigma_pred, Sigma_true).item()
        w_pred = min_var_weights(Sigma_pred, ridge=args.ridge).unsqueeze(1)
        port_var = torch.bmm(torch.bmm(w_pred, Sigma_true),
                             w_pred.transpose(1, 2))
        port_loss = port_var.reshape(-1).mean().item()
    print(f"[loss-scale/{label}] cov_loss = {cov_loss:.4e}   "
          f"port_loss = {port_loss:.4e}   "
          f"equal-influence port_weight = {cov_loss / max(port_loss, 1e-12):.4e}")
    model.train()

def main(args: argparse.Namespace) -> None:
    minnt.startup(args.seed, args.threads)
    minnt.global_keras_initializers()

    X_raw = np.loadtxt("data/cholesky_kernel.txt", dtype=np.float32)
    TARGET_SCALE = 1
    Sigma_raw = np.loadtxt("data/rc_kernel.txt", dtype=np.float32) * TARGET_SCALE

    loss_fn = make_combined_loss(args.port_weight, args.ridge)

    if args.val_method == "cv":
        folds = walk_forward_cv(X_raw, Sigma_raw,
                                n_splits=5, val_size=200, seq_len=args.seq_len)
        dev_losses = []
        for fold_idx, (fold_train_X, fold_train_S, fold_val_X, fold_val_S) in enumerate(folds):
            fold_train_loader = DataLoader(
                TimeSeriesDataset(fold_train_X, fold_train_S, args.seq_len),
                batch_size=args.batch_size, shuffle=False
            )
            fold_val_loader = DataLoader(
                TimeSeriesDataset(fold_val_X, fold_val_S, args.seq_len),
                batch_size=args.batch_size
            )
            fold_model = Model(args)
            fold_optimizer = make_optimizer(fold_model, args)
            fold_model.configure(
                optimizer=fold_optimizer,
                scheduler=minnt.schedulers.GenericDecay(
                    fold_optimizer,
                    args.epochs * len(fold_train_loader),
                    args.learning_rate_decay
                ),
                loss=loss_fn,
                metrics={"cov_mse": CovMSE(n=8)},
                logdir=minnt.format_logdir(
                    f"logs/{{file-}}{{timestamp}}{{-config}}_fold{fold_idx}",
                    **vars(args)
                ),
            )
            fold_model.fit(fold_train_loader, dev=fold_val_loader,
                           epochs=args.epochs, log_config=vars(args))
            result = fold_model.evaluate(fold_val_loader)
            dev_losses.append(result)
            print(f"Fold {fold_idx + 1} dev loss: {result}")
        return

    train_mean = X_raw[:2792].mean(axis=0)
    train_std = X_raw[:2792].std(axis=0) + 1e-8
    X = (X_raw - train_mean) / train_std

    train_X = X[:2792]
    dev_X = X[2792 - args.seq_len:3392]
    test_X = X[3392 - args.seq_len:]

    train_S = Sigma_raw[:2792]
    dev_S = Sigma_raw[2792 - args.seq_len: 3392]
    test_S = Sigma_raw[3392 - args.seq_len:]

    train = DataLoader(TimeSeriesDataset(train_X, train_S, args.seq_len), batch_size=args.batch_size, shuffle=True)

    dev = DataLoader(TimeSeriesDataset(dev_X, dev_S, args.seq_len), batch_size=args.batch_size)

    test = DataLoader(TimeSeriesDataset(test_X, test_S, args.seq_len), batch_size=args.batch_size)

    model = Model(args)
    _scale_sanity_check(model, train, label="train")
    optimizer = make_optimizer(model, args)

    model.configure(
        optimizer=optimizer,
        scheduler=minnt.schedulers.GenericDecay(
            optimizer,
            args.epochs * len(train),
            args.learning_rate_decay
        ),
        loss=loss_fn,
        metrics={"cov_mse": CovMSE(n=8)},
        logdir=minnt.format_logdir("logs/{file-}{timestamp}{-config}_final",
                                   **vars(args)),
    )
    _loss_scale_check(model, train, args, "train")
    model.fit(train, dev=dev, epochs=args.epochs, log_config=vars(args), log_graph=True)
    model.evaluate(test)
    # Per-sample diagnostics on test
    # --- per-sample diagnostics on test ---
    device = next(model.parameters()).device
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test:
            x = x.to(device);
            y = y.to(device)
            all_pred.append(vec_to_cov(model(x)).cpu())
            all_true.append(vec_to_sym(y).cpu())
    Sp = torch.cat(all_pred)
    St = torch.cat(all_true)

    eigs = torch.linalg.eigvalsh(Sp)  # (N, 8)
    print("Σ_pred eigenvalues — min:", eigs.min().item(),
          " 1%:", eigs.flatten().quantile(0.01).item(),
          " median:", eigs.median().item(),
          " max:", eigs.max().item())

    inv = torch.linalg.solve(Sp, St)
    qlike = torch.diagonal(inv, dim1=1, dim2=2).sum(-1) - torch.logdet(inv) - 8
    print("QLIKE — mean:", qlike.mean().item(),
          " median:", qlike.median().item(),
          " 95%:", qlike.quantile(0.95).item(),
          " max:", qlike.max().item())
    if args.save_predictions:
        device = next(model.parameters()).device
        model.eval()
        preds = []
        with torch.no_grad():  # <-- add this
            for x, _ in test:
                x = x.to(device)
                pred = model(x)
                t_std = torch.tensor(train_std, dtype=pred.dtype, device=device)
                t_mean = torch.tensor(train_mean, dtype=pred.dtype, device=device)
                Sigma = vec_to_cov(pred)
                Sigma_vec = _lower_tri_to_vec(Sigma, n=8)
                preds.append(Sigma_vec.cpu().numpy())
        preds = np.concatenate(preds, axis=0) / TARGET_SCALE
        np.savetxt(f"data/pred_{args.model_type}_rc_kernel.txt", preds, delimiter='\t')
        print(f"Saved predictions to data/pred_{args.model_type}_rc_kernel.txt")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)