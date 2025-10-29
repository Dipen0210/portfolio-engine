# performance/performance_metrics.py
import numpy as np
import pandas as pd


def compute_daily_returns(value_history: pd.DataFrame) -> pd.Series:
    """
    Compute daily returns from portfolio value history.
    """
    if len(value_history) < 2:
        return pd.Series(dtype=float)

    value_history = value_history.sort_values("Date").reset_index(drop=True)
    returns = value_history["Value"].pct_change().dropna()
    returns.index = value_history["Date"].iloc[1:]
    return returns


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.04, periods_per_year=252):
    """
    Annualized Sharpe Ratio.
    """
    if len(returns) == 0:
        return np.nan
    excess = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1)


def sortino_ratio(returns: pd.Series, risk_free_rate=0.04, periods_per_year=252):
    """
    Annualized Sortino Ratio (downside deviation only).
    """
    if len(returns) == 0:
        return np.nan
    downside = returns[returns < 0]
    if downside.std(ddof=1) == 0:
        return np.nan
    excess = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1)


def max_drawdown(value_history: pd.DataFrame):
    """
    Maximum drawdown from peak to trough.
    """
    if "Value" not in value_history.columns:
        return np.nan

    cum_max = value_history["Value"].cummax()
    drawdowns = (value_history["Value"] - cum_max) / cum_max
    return drawdowns.min()


def cumulative_return(value_history: pd.DataFrame):
    """
    Total cumulative portfolio return.
    """
    if len(value_history) < 2:
        return 0
    start_val = value_history["Value"].iloc[0]
    end_val = value_history["Value"].iloc[-1]
    return (end_val - start_val) / start_val
