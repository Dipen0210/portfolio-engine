# risk_management/risk_models.py
import numpy as np
import pandas as pd

try:
    # Optional: nicer sample->shrinkage covariance (if installed)
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: columns = tickers, index = DatetimeIndex, values = Close prices
    returns: daily log returns aligned across tickers
    """
    r = np.log(price_df / price_df.shift(1))
    return r.dropna(how="all").dropna(axis=1, how="all")


def compute_covariance_matrix(
    returns_df: pd.DataFrame,
    method: str = "sample",
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    method: 'sample' | 'ewma' | 'ledoit_wolf'
    - sample: plain sample covariance
    - ewma: exponentially weighted covariance (RiskMetrics)
    - ledoit_wolf: shrinkage to identity (requires scikit-learn)
    """
    if method == "sample":
        cov = returns_df.cov(min_periods=2)

    elif method == "ewma":
        # RiskMetrics style: S_t = λ S_{t-1} + (1-λ) r_t r_t'
        X = returns_df.dropna().values
        if X.shape[0] < 2:
            return pd.DataFrame(np.nan, index=returns_df.columns, columns=returns_df.columns)
        lam = ewma_lambda
        S = np.cov(X[:2].T)  # init with first 2 obs
        for t in range(2, X.shape[0]):
            rt = X[t, :][:, None]
            S = lam * S + (1 - lam) * (rt @ rt.T)
        cov = pd.DataFrame(S, index=returns_df.columns, columns=returns_df.columns)

    elif method == "ledoit_wolf":
        if not _HAS_SKLEARN:
            # fallback to sample if sklearn not present
            cov = returns_df.cov(min_periods=2)
        else:
            lw = LedoitWolf().fit(returns_df.dropna().values)
            cov = pd.DataFrame(lw.covariance_, index=returns_df.columns, columns=returns_df.columns)
    else:
        raise ValueError("Unknown method for covariance matrix.")

    return cov


def annualize_vol(series_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility from daily returns series."""
    return np.sqrt(periods_per_year) * series_returns.std(ddof=1)


def portfolio_vol(weights: pd.Series | np.ndarray, cov: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Annualized portfolio volatility."""
    w = np.asarray(weights).reshape(-1, 1)
    sigma2 = float(w.T @ cov.values @ w)
    return np.sqrt(sigma2 * periods_per_year)


def portfolio_mean(weights: pd.Series | np.ndarray, mu: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized portfolio expected return (assuming μ are daily expected returns)."""
    w = np.asarray(weights).ravel()
    return float(w @ (mu.values * periods_per_year))


def sharpe_ratio(weights, mu, cov, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe: (E[R]-rf)/σ."""
    er = portfolio_mean(weights, mu, periods_per_year)
    vol = portfolio_vol(weights, cov, periods_per_year)
    return (er - rf_annual) / (vol + 1e-12)
