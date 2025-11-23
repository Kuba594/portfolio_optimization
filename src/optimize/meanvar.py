# src/optimize/meanvar.py
import numpy as np
import cvxpy as cp

def solve_min_var(Sigma: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Min-variance portfolio: min w'Σw s.t. sum w = 1, w>=0 (if long_only)."""
    n = Sigma.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    return np.array(w.value).ravel()

def solve_mean_var(Sigma: np.ndarray,
                   mu: np.ndarray,
                   risk_aversion: float = 10.0,
                   long_only: bool = True) -> np.ndarray:
    """
    Mean-variance: min 0.5*w'Σw - (1/λ)*μ'w
    Larger λ → more risk-averse.
    """
    n = Sigma.shape[0]
    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w
    objective = cp.Minimize(0.5 * risk - (1.0 / risk_aversion) * ret)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    return np.array(w.value).ravel()