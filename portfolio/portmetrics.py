# portfolio/portfolio_metrics.py

import numpy as np


def compute_portfolio_metrics(mu_vec, cov_matrix, weights, risk_free_rate=0.04):
    """
    Compute expected portfolio return, volatility, and Sharpe ratio.
    """

    portfolio_return = np.dot(weights, mu_vec)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    if portfolio_vol == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (portfolio_return -
                        risk_free_rate / 252) / portfolio_vol

    return portfolio_return, portfolio_vol, sharpe_ratio
