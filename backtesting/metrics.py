# backtesting/metrics.py
import numpy as np


def compute_performance_metrics(returns, risk_free_rate=0.04):
    """
    Compute key performance metrics from return series.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily or periodic returns
    risk_free_rate : float
        Annual risk-free rate (default 4%)

    Returns
    -------
    dict
        Contains Sharpe, Sortino, Max Drawdown, CAGR, Volatility
    """

    mean_return = returns.mean()
    vol = returns.std()
    downside_vol = returns[returns < 0].std()
    annual_factor = 252  # trading days

    sharpe = ((mean_return * annual_factor) - risk_free_rate) / \
        (vol * np.sqrt(annual_factor)) if vol > 0 else 0
    sortino = ((mean_return * annual_factor) - risk_free_rate) / \
        (downside_vol * np.sqrt(annual_factor)) if downside_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    max_drawdown = (1 - cumulative / cumulative.cummax()).max()

    n_years = len(returns) / annual_factor
    cagr = (cumulative.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        "Annualized Return": mean_return * annual_factor,
        "Annualized Volatility": vol * np.sqrt(annual_factor),
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "CAGR": cagr
    }
