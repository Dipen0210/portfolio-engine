# risk_management/var_cvar.py
import numpy as np
import pandas as pd
from scipy.stats import norm


def parametric_var_cvar(portfolio_returns: pd.Series, alpha: float = 0.95):
    """
    Parametric (Gaussian) one-period VaR & CVaR on daily returns.
    Returns VaR (<0) and CVaR (<0).
    """
    r = portfolio_returns.dropna()
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan, np.nan

    z = norm.ppf(1 - alpha)  # negative number for alpha>0.5
    var = mu + z * sigma
    # CVaR under normal: mu - sigma * (pdf(z)/(1-alpha))
    cvar = mu - sigma * (norm.pdf(z) / (1 - alpha))
    return float(var), float(cvar)


def historical_var_cvar(portfolio_returns: pd.Series, alpha: float = 0.95):
    """
    Historical one-period VaR & CVaR.
    """
    r = portfolio_returns.dropna().sort_values()
    if len(r) == 0:
        return np.nan, np.nan

    idx = int((1 - alpha) * len(r))
    var = r.iloc[idx]  # quantile on left tail
    cvar = r.iloc[: idx + 1].mean() if idx >= 0 else r.min()
    return float(var), float(cvar)


def portfolio_returns_from_weights(returns_df: pd.DataFrame, weights: pd.Series | np.ndarray) -> pd.Series:
    """
    Daily portfolio returns from individual asset returns and weights.
    """
    w = np.asarray(weights).ravel()
    aligned = returns_df.dropna(how="all").fillna(0.0)
    return aligned @ w
