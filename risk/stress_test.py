# risk_management/stress_tests.py
import numpy as np
import pandas as pd


def apply_parametric_shock(returns_df: pd.DataFrame, shock: float = -0.05) -> pd.Series:
    """
    Apply uniform one-day shock to all assets (e.g., -5%).
    Returns a single-day shocked portfolio return vector (per-asset).
    """
    return pd.Series(shock, index=returns_df.columns)


def sector_shock(returns_df: pd.DataFrame, sector_map: pd.Series, shock_dict: dict) -> pd.Series:
    """
    Apply sector-specific shocks, e.g., {'Technology': -0.08, 'Energy': -0.02}
    sector_map: index=tickers, value=sector
    """
    shocked = pd.Series(0.0, index=returns_df.columns)
    for ticker in shocked.index:
        sec = sector_map.get(ticker, None)
        shocked[ticker] = shock_dict.get(sec, 0.0)
    return shocked


def historical_scenario(returns_df: pd.DataFrame, scenario_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """
    Pull realized returns from specific dates (e.g., crash days) as a scenario matrix.
    Returns a DataFrame [scenario_date x tickers].
    """
    r = returns_df.copy()
    scenarios = []
    idx = []
    for d in scenario_dates:
        if d in r.index:
            scenarios.append(r.loc[d])
            idx.append(d)
    if not scenarios:
        return pd.DataFrame(columns=returns_df.columns)
    return pd.DataFrame(scenarios, index=idx)


def monte_carlo_paths(mu: pd.Series, cov: pd.DataFrame, horizon_days=5, n_sims=5000, seed: int | None = 42):
    """
    Simulate multi-asset returns over horizon using Gaussian (can be extended).
    Returns array shape (n_sims, horizon_days, n_assets).
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov.values + 1e-12 * np.eye(cov.shape[0]))
    n = cov.shape[0]
    sims = rng.standard_normal(size=(n_sims, horizon_days, n))
    # correlate
    sims = sims @ L.T
    # add drift mu (daily)
    drift = mu.values.reshape(1, 1, n)
    return sims + drift
