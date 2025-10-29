# risk_management/risk_pipeline.py
import numpy as np
import pandas as pd

from .risk_models import compute_log_returns, compute_covariance_matrix, portfolio_vol, portfolio_mean, sharpe_ratio
from .var_cvar import portfolio_returns_from_weights, parametric_var_cvar, historical_var_cvar
from .stress_test import historical_scenario, apply_parametric_shock
from .exposures import basic_sanity_checks


def build_risk_report(
    price_df: pd.DataFrame,      # Close prices (index=dates, columns=tickers)
    weights: pd.Series,          # index=tickers, sum=1
    # daily expected returns (optional for Sharpe/mean)
    mu: pd.Series | None = None,
    cov_method: str = "ledoit_wolf",
    alpha: float = 0.95,
    rf_annual: float = 0.0
) -> dict:
    """
    Returns a dict with key risk metrics and small tables you can show on dashboard.
    """
    # Align
    price_df = price_df.dropna(how="all").dropna(axis=1, how="all")
    weights = weights.reindex(price_df.columns).fillna(0.0)
    basic_sanity_checks(weights)

    # Returns & cov
    r = compute_log_returns(price_df)
    cov = compute_covariance_matrix(r, method=cov_method)

    # Portfolio daily returns
    pr = portfolio_returns_from_weights(r, weights)

    # Core risk numbers (annualized vol, mean, Sharpe)
    port_vol = portfolio_vol(weights, cov)
    port_mean = portfolio_mean(weights, mu) if mu is not None else np.nan
    port_sharpe = sharpe_ratio(
        weights, mu, cov, rf_annual=rf_annual) if mu is not None else np.nan

    # VaR / CVaR (parametric & historical, 1-day)
    p_var, p_cvar = parametric_var_cvar(pr, alpha=alpha)
    h_var, h_cvar = historical_var_cvar(pr, alpha=alpha)

    # Simple stress: uniform -5% shock
    shock_vec = apply_parametric_shock(r, shock=-0.05)
    shock_pnl = float(
        (weights.values * shock_vec.reindex(weights.index).fillna(0.0).values).sum())

    report = {
        "weights": weights,
        "cov": cov,
        "daily_returns": r,
        "portfolio_daily_returns": pr,
        "annualized_vol": port_vol,
        "annualized_mean": port_mean,
        "sharpe": port_sharpe,
        "parametric_VaR": p_var,
        "parametric_CVaR": p_cvar,
        "historical_VaR": h_var,
        "historical_CVaR": h_cvar,
        "stress_uniform_minus5_pct_portfolio_return": shock_pnl,
    }
    return report
