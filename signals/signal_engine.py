# signals_generation/portfolio_signal_engine.py
from __future__ import annotations

from datetime import datetime

import pandas as pd

from utils.formatting import normalize_and_truncate_weights, truncate_series

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
    Returns
    -------
    signals_df : pd.DataFrame
        Columns: ['Ticker', 'Signal', 'Old_Weight', 'New_Weight', 'Reason']
    """

    def _prepare_portfolio(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Ticker", "Weight"])
        working = df.copy()
        if "Ticker" not in working.columns or "Weight" not in working.columns:
            return pd.DataFrame(columns=["Ticker", "Weight"])
        working["Ticker"] = working["Ticker"].astype(str).str.strip()
        working = working[working["Ticker"] != ""]
        working["Weight"] = pd.to_numeric(working["Weight"], errors="coerce")
        working = working.dropna(subset=["Weight"])
        if "CarryForward" in working.columns:
            working = working.drop(columns=["CarryForward"])
        working = (
            working.sort_values(["Ticker"])
            .groupby("Ticker", as_index=False, sort=False)
            .last()
        )
        working = normalize_and_truncate_weights(working, decimals=4)
        working["Weight"] = truncate_series(working["Weight"], decimals=4)
        return working[["Ticker", "Weight"]]

    results = []
    date_str = _normalize_date(as_of_date)

    prepared_old = _prepare_portfolio(old_portfolio_df)
    prepared_new = _prepare_portfolio(new_portfolio_df)

    old_tickers = set(prepared_old["Ticker"])
    new_tickers = set(prepared_new["Ticker"])

    # 1️⃣ Initial entry (first portfolio → BUY all)
    if not old_tickers:
        for _, row in prepared_new.iterrows():
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
    old_weights = dict(zip(prepared_old['Ticker'], prepared_old['Weight']))
    new_weights = dict(zip(prepared_new['Ticker'], prepared_new['Weight']))

    tolerance = 1e-6

    if not new_tickers and old_tickers:
        for ticker in sorted(old_tickers):
            old_w = old_weights.get(ticker, 0.0)
            results.append({
                "Ticker": ticker,
                "Signal": "HOLD",
                "Old_Weight": old_w,
                "New_Weight": old_w,
                "Reason": "Allocation carried forward (no updated weights available)",
                "Date": date_str,
            })
        signals_df = pd.DataFrame(results)
        signals_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_signals(signals_df)
        return signals_df

    # 2️⃣ SELL – stocks removed from portfolio
    for ticker in old_tickers - new_tickers:
        old_w = old_weights.get(ticker, 0.0)
        if abs(old_w) <= tolerance:
            results.append({
                "Ticker": ticker,
                "Signal": "HOLD",
                "Old_Weight": old_w,
                "New_Weight": old_w,
                "Reason": "Position weight unchanged within tolerance",
                "Date": date_str,
            })
            continue
        results.append({
            "Ticker": ticker,
            "Signal": "SELL",
            "Old_Weight": old_w,
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
        delta = abs(new_w - old_w)
        if delta <= tolerance:
            results.append({
                "Ticker": ticker,
                "Signal": "HOLD",
                "Old_Weight": old_w,
                "New_Weight": new_w,
                "Reason": "Allocation unchanged",
                "Date": date_str,
            })
            continue

        results.append({
            "Ticker": ticker,
            "Signal": "REBALANCE",
            "Old_Weight": old_w,
            "New_Weight": new_w,
            "Reason": "Weight updated in latest optimization",
            "Date": date_str,
        })

    signals_df = pd.DataFrame(results)
    signals_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_signals(signals_df)
    return signals_df
