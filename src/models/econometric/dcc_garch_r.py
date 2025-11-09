import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

# --- R CODE BLOCK START ---
robjects.r("""
suppressMessages(library(rugarch))
suppressMessages(library(rmgarch))

dcc_roll_forecast <- function(ret_df,
                              spec='sGARCH',
                              window=500,
                              refit_every=1,
                              n_jobs=4) {

  # --- basic sanity checks ---
  print(paste("Input:", nrow(ret_df), "rows,", ncol(ret_df), "columns"))
  if (nrow(ret_df) <= window) {
    warning("Not enough data for rolling DCC estimation.")
    return(array(NA_real_, dim = c(ncol(ret_df), ncol(ret_df), 0)))
  }

  colnames(ret_df) <- make.names(colnames(ret_df))
  uspec <- ugarchspec(
    variance.model = list(model = spec, garchOrder = c(1,1)),
    mean.model     = list(armaOrder = c(0,0), include.mean = FALSE),
    distribution.model = 'norm'
  )
  mspec   <- multispec(replicate(ncol(ret_df), uspec))
  dccspec <- dccspec(uspec = mspec, dccOrder = c(1,1), distribution = 'mvnorm')

  # --- cluster type: FORK (Linux/Mac) vs PSOCK (Windows) ---
  cluster_type <- if (.Platform$OS.type == "windows") "PSOCK" else "FORK"
  
  # Drop columns with zero variance or all-NA
valid_cols <- apply(ret_df, 2, function(x) all(is.finite(x)) && var(x, na.rm = TRUE) > 0)
if (!all(valid_cols)) {
  warning(paste("Dropping", sum(!valid_cols), "columns with zero variance or invalid values"))
  ret_df <- ret_df[, valid_cols, drop = FALSE]
}

  # --- main rolling fit ---
  roll <- tryCatch(
  dccroll(
    dccspec,
    data = as.matrix(ret_df),
    n.ahead = 1,
    forecast.length = nrow(ret_df) - window,
    refit.every = refit_every,
    refit.window = "moving",
    solver = "nlminb",
    fit.control = list( scale = TRUE),
    solver.control = list(trace = 0)
  ),
  error = function(e) {
    print(paste("Error in dccroll:", e$message))
    NULL
  }
)
warns <- warnings()
if (length(warns) > 0) {
  cat("Collected", length(warns), "warnings from dccroll()\\n")
  print(warns)
}
  # --- handle failed run ---
  if (is.null(roll)) {
    n <- ncol(ret_df)
    warning("DCC roll returned NULL → returning empty array")
    return(array(NA_real_, dim = c(n, n, 0)))
  }

  rc <- rcov(roll)

  # --- ensure rcov always returns a 3D array [N, N, T] ---
  if (is.list(rc)) {
    rc <- simplify2array(rc)
  }

  rc
}
""")

_r_dccroll = robjects.globalenv["dcc_roll_forecast"]

def dcc_roll_cov_series(returns_df, window=500, refit_every=1, n_jobs=4):
    """Run rolling DCC-GARCH forecast via R and return aligned covariance series."""
    X = returns_df.dropna()
    with localconverter(default_converter + pandas2ri.converter):
        rcov_arr = np.array(
            _r_dccroll(X,
                       window=window,
                       refit_every=refit_every,
                       n_jobs=n_jobs)
        )

    # rcov_arr shape: [N, N, T]
    if rcov_arr.ndim == 2:
        rcov_arr = rcov_arr[:, :, np.newaxis]

    covs = np.transpose(rcov_arr, (2, 0, 1))

    # --- correct date alignment: use last T dates ---
    dates = X.index[-covs.shape[0]:]

    return dates, covs