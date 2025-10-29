# performance/analytics_engine.py
import pandas as pd
from .performance_metrics import (
    compute_daily_returns,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cumulative_return
)


def summarize_performance(portfolio_state, benchmark_returns=None):
    """
    Compute full performance summary for a given PortfolioState.

    Parameters
    ----------
    portfolio_state : PortfolioState
        Contains portfolio_value_history
    benchmark_returns : pd.Series or None
        Optional benchmark daily returns (e.g. S&P 500)

    Returns
    -------
    dict : performance metrics summary
    """
    value_df = portfolio_state.portfolio_value_history.copy()
    returns = compute_daily_returns(value_df)

    metrics = {
        "Cumulative Return": cumulative_return(value_df),
        "Annualized Sharpe": sharpe_ratio(returns),
        "Annualized Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(value_df),
        "Volatility (Ann.)": returns.std(ddof=1) * (252 ** 0.5),
    }

    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        # Align indexes
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["Portfolio", "Benchmark"]
        cov = aligned.cov().iloc[0, 1]
        beta = cov / aligned["Benchmark"].var(ddof=1)
        alpha = aligned["Portfolio"].mean() - beta * \
            aligned["Benchmark"].mean()
        metrics["Alpha"] = alpha * 252
        metrics["Beta"] = beta

    return metrics
