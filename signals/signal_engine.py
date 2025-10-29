# signals_generation/portfolio_signal_engine.py
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .signal_logger import log_signals


def _normalize_date(as_of_date) -> str:
    if as_of_date is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(as_of_date, datetime):
        return as_of_date.strftime("%Y-%m-%d")
    return str(as_of_date)


def generate_portfolio_signals(
    old_portfolio_df: pd.DataFrame | None,
    new_portfolio_df: pd.DataFrame,
    drift_threshold: float = 0.03,
    as_of_date=None,
):
    """
    Compare old vs new optimized portfolios and generate precise trade instructions.

    Parameters
    ----------
    old_portfolio_df : pd.DataFrame
        Previous portfolio (Ticker, Weight)
    new_portfolio_df : pd.DataFrame
        New optimized portfolio (Ticker, Weight)
    drift_threshold : float
        % weight deviation allowed before triggering rebalance (default 3%)

    Returns
    -------
    signals_df : pd.DataFrame
        Columns: ['Ticker', 'Signal', 'Old_Weight', 'New_Weight', 'Reason']
    """

    results = []
    date_str = _normalize_date(as_of_date)

    old_tickers = set(old_portfolio_df['Ticker']
                      ) if old_portfolio_df is not None else set()
    new_tickers = set(new_portfolio_df['Ticker'])

    # 1️⃣ Initial entry (first portfolio → BUY all)
    if not old_tickers:
        for _, row in new_portfolio_df.iterrows():
            results.append({
                "Ticker": row['Ticker'],
                "Signal": "BUY",
                "Old_Weight": 0,
                "New_Weight": row['Weight'],
                "Reason": "Initial portfolio allocation",
                "Date": date_str,
            })
        signals_df = pd.DataFrame(results)
        log_signals(signals_df)
        return signals_df

    # Convert to dicts for quick lookup
    old_weights = dict(
        zip(old_portfolio_df['Ticker'], old_portfolio_df['Weight']))
    new_weights = dict(
        zip(new_portfolio_df['Ticker'], new_portfolio_df['Weight']))

    # 2️⃣ SELL – stocks removed from portfolio
    for ticker in old_tickers - new_tickers:
        results.append({
            "Ticker": ticker,
            "Signal": "SELL",
            "Old_Weight": old_weights.get(ticker, 0),
            "New_Weight": 0,
            "Reason": "Removed from new optimized portfolio",
            "Date": date_str,
        })

    # 3️⃣ BUY – new stocks added to portfolio
    for ticker in new_tickers - old_tickers:
        results.append({
            "Ticker": ticker,
            "Signal": "BUY",
            "Old_Weight": 0,
            "New_Weight": new_weights[ticker],
            "Reason": "Newly added to optimized portfolio",
            "Date": date_str,
        })

    # 4️⃣ REBALANCE / HOLD – existing stocks (check weight drift)
    for ticker in old_tickers & new_tickers:
        old_w = old_weights[ticker]
        new_w = new_weights[ticker]
        drift = abs(new_w - old_w) / old_w if old_w > 0 else 1.0

        if drift > drift_threshold:
            results.append({
                "Ticker": ticker,
                "Signal": "REBALANCE",
                "Old_Weight": old_w,
                "New_Weight": new_w,
                "Reason": f"Weight drifted by {drift:.2%}",
                "Date": date_str,
            })
        else:
            results.append({
                "Ticker": ticker,
                "Signal": "HOLD",
                "Old_Weight": old_w,
                "New_Weight": new_w,
                "Reason": "No significant change",
                "Date": date_str,
            })

    signals_df = pd.DataFrame(results)
    signals_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_signals(signals_df)
    return signals_df
