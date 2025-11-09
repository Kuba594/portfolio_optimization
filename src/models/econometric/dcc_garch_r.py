#src/models/econometric/dcc_garch_r.py
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

_R = robjects.r
_R("""
suppressMessages(library(rugarch))
suppressMessages(library(rmgarch))

dcc_fit_forecast <- function(ret_df, spec='sGARCH', dist='norm', horizon=1) {
  colnames(ret_df) <- make.names(colnames(ret_df))
  uspec <- ugarchspec(
    variance.model=list(model=spec, garchOrder=c(1,1)),
    mean.model=list(armaOrder=c(0,0), include.mean=FALSE),
    distribution.model=dist
  )
  mspec <- multispec(replicate(ncol(ret_df), uspec))
  dccspec <- dccspec(uspec=mspec, dccOrder=c(1,1), distribution=dist)
  fit <- dccfit(dccspec, data=as.matrix(ret_df))
  fcast <- dccforecast(fit, n.ahead=horizon)
  cov1 <- rcov(fcast)[[1]][,,1]
  colnames(cov1) <- colnames(ret_df)
  rownames(cov1) <- colnames(ret_df)
  cov1
}
""")
_r_dcc = robjects.globalenv["dcc_fit_forecast"]

def dcc_one_step_cov(returns_df: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)
    X = returns_df.dropna()
    if X.shape[0] < 50:
        raise ValueError("Need at least 50 observations for DCC.")
    cov = _r_dcc(X)
    cov = np.array(cov).reshape((X.shape[1], X.shape[1]))
    return cov