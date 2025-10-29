# portfolio/portfolio_builder.py

import pandas as pd
import numpy as np
from .portmetrics import compute_portfolio_metrics


def build_portfolio(final_weights_df, mu, cov_matrix, sector_map=None, risk_free_rate=0.04):
    """
    Build final portfolio and compute its expected metrics.

    Parameters
    ----------
    final_weights_df : pd.DataFrame
        Columns: ['Ticker', 'Final_Weight']
    mu : pd.Series
        Expected returns indexed by Ticker
    cov_matrix : pd.DataFrame
        Covariance matrix of returns
    sector_map : dict or pd.Series, optional
        Mapping of ticker -> sector
    risk_free_rate : float
        Annualized risk-free rate (default 4%)

    Returns
    -------
    portfolio : dict
        Full portfolio object with weights, metrics, sector exposure
    """

    # Align μ and Σ with portfolio tickers
    tickers = final_weights_df['Ticker'].values
    weights = final_weights_df['Final_Weight'].values

    mu_vec = mu.reindex(tickers).fillna(0).values
    cov = cov_matrix.reindex(index=tickers, columns=tickers).fillna(0).values

    # Portfolio stats
    portfolio_return, portfolio_vol, sharpe = compute_portfolio_metrics(
        mu_vec, cov, weights, risk_free_rate)

    # Sector allocation
    sector_alloc = None
    if sector_map is not None:
        sectors = [sector_map.get(t, "Unknown") for t in tickers]
        df = pd.DataFrame(
            {"Ticker": tickers, "Sector": sectors, "Weight": weights})
        sector_alloc = df.groupby(
            "Sector")["Weight"].sum().sort_values(ascending=False)

    portfolio = {
        "Tickers": tickers,
        "Weights": weights,
        "Expected_Return": portfolio_return,
        "Volatility": portfolio_vol,
        "Sharpe": sharpe,
        "Sector_Allocation": sector_alloc
    }

    return portfolio
