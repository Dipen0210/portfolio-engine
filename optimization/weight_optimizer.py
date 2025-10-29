# optimization/weight_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .constraints import weight_sum_constraint, non_negative_constraint


def mean_variance_optimize(mu, cov_matrix, risk_level="medium"):
    """
    Mean-Variance Optimization (Markowitz)
    Maximizes Sharpe Ratio for given risk preference.
    """

    tickers = mu.index
    n = len(tickers)

    # Risk tolerance based on user level
    risk_map = {"low": 2.0, "medium": 4.0, "high": 8.0}
    risk_aversion = risk_map.get(risk_level.lower(), 4.0)

    mu_vec = mu.values
    cov = cov_matrix.values

    def portfolio_return(w):
        return np.dot(w, mu_vec)

    def portfolio_volatility(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    def objective(w):
        # Minimize negative utility: -return + λ * risk
        return -portfolio_return(w) + risk_aversion * portfolio_volatility(w)

    # Initial guess and constraints
    x0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]  # no shorting
    constraints = {'type': 'eq', 'fun': weight_sum_constraint}

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    weights = pd.Series(result.x, index=tickers)
    return weights / weights.sum()  # normalized
