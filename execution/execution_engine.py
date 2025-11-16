# execution/execution_engine.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .portfolio_state import PortfolioState
from utils.formatting import truncate_value


def snapshot_prices(price_data_dict: dict, trading_date) -> dict[str, float]:
    """
    Build an execution snapshot of opening prices for the supplied trading date.
    """
    ts = pd.Timestamp(trading_date).normalize()
    snapshot: dict[str, float] = {}
    for ticker, df in price_data_dict.items():
        if df is None or df.empty:
            continue
        working = df.copy()
        if "Date" in working.columns:
            working["Date"] = pd.to_datetime(working["Date"], errors="coerce").dt.normalize()
            matches = working[working["Date"] == ts]
        else:
            working.index = pd.to_datetime(working.index)
            matches = working.loc[working.index.normalize() == ts]
        if matches.empty or "Open" not in matches.columns:
            continue
        snapshot[ticker] = float(matches["Open"].iloc[-1])
    if not snapshot:
        raise ValueError(f"No open prices available on {ts.date()} for the provided tickers.")
    return snapshot


def target_shares_from_weights(
    target_weights: pd.DataFrame,
    prices: dict[str, float],
    equity: float,
) -> dict[str, float]:
    """
    Convert target weights to fractional share targets using execution-day open prices.
    """
    targets: dict[str, float] = {}
    for _, row in target_weights.iterrows():
        ticker = row["Ticker"]
        weight = float(row["Weight"])
        price = prices.get(ticker)
        if price is None or np.isnan(price) or price <= 0:
            raise ValueError(f"Missing open price for {ticker}; unable to size target shares.")
        notional = equity * weight
        qty = truncate_value(notional / price, decimals=4)
        targets[ticker] = max(qty, 0.0)
    return targets


def reconcile_orders(state: PortfolioState, target_shares: dict[str, float]) -> pd.DataFrame:
    """
    Create order instructions by comparing existing holdings vs desired share counts.
    """
    orders: list[dict] = []
    current = state.positions.copy()
    tickers = set(current.keys()) | set(target_shares.keys())
    for ticker in tickers:
        current_qty = float(current.get(ticker, 0.0))
        target_qty = float(target_shares.get(ticker, 0.0))
        delta = target_qty - current_qty
        if delta > 1e-8:
            orders.append({"Ticker": ticker, "Side": "BUY", "Qty": delta})
        elif delta < -1e-8:
            orders.append({"Ticker": ticker, "Side": "SELL", "Qty": abs(delta)})

    if not orders:
        return pd.DataFrame()
    return pd.DataFrame(orders).sort_values(
        by="Side", key=lambda col: col != "SELL"
    ).reset_index(drop=True)


def execute_orders(
    state: PortfolioState,
    orders_df: pd.DataFrame,
    prices: dict[str, float],
    date=None,
    commission_per_trade: float = 1.0,
) -> None:
    """
    Fill all orders at the execution-day open price and update portfolio accounting.
    """
    if orders_df is None or orders_df.empty:
        return

    exec_date = date or datetime.now().strftime("%Y-%m-%d")

    for _, order in orders_df.iterrows():
        ticker = order["Ticker"]
        side = str(order["Side"]).upper()
        qty = float(order["Qty"])
        price = prices.get(ticker)
        if price is None or np.isnan(price) or price <= 0 or qty <= 0:
            continue

        fee = commission_per_trade if commission_per_trade > 0 else 0.0

        if side == "BUY":
            available_cash = state.cash - fee
            if available_cash <= 0:
                continue
            max_affordable = available_cash / price
            max_affordable = truncate_value(max_affordable, decimals=4)
            if max_affordable <= 0:
                continue
            qty = min(qty, max_affordable)
            if qty <= 0:
                continue

        qty_delta = qty if side == "BUY" else -qty
        realized = state.update_position(ticker, qty_delta, price=price, side=side)
        if fee > 0:
            state.deduct_fee(fee)

        state.append_trade(
            {
                "Date": exec_date,
                "Ticker": ticker,
                "Side": side,
                "Qty": truncate_value(qty, decimals=4),
                "Price": truncate_value(price, decimals=4),
                "Notional": truncate_value(price * qty, decimals=4),
                "RealizedPnL": truncate_value(realized, decimals=4),
                "Fee": truncate_value(fee, decimals=4),
                "CashAfter": truncate_value(state.cash, decimals=4),
            }
        )


def run_rebalance_cycle(
    state: PortfolioState,
    price_data_dict: dict,
    new_portfolio_weights: pd.DataFrame,
    date=None,
    commission_per_trade: float = 1.0,
) -> pd.DataFrame:
    """
    Execute a rebalance using target weights and execution-day open prices.
    """
    date = date or (list(price_data_dict.values())[0].index[-1] if price_data_dict else None)
    if date is None:
        raise ValueError("Execution date is required to perform a rebalance.")

    open_prices = snapshot_prices(price_data_dict, date)

    equity = state.current_equity(open_prices)

    weights = new_portfolio_weights.copy()
    weights["Weight"] = pd.to_numeric(weights["Weight"], errors="coerce")
    weights = weights.dropna(subset=["Weight"])
    if weights.empty:
        return pd.DataFrame(columns=["Ticker", "Side", "Qty"])
    weights["Weight"] = weights["Weight"] / weights["Weight"].sum()
    targets = target_shares_from_weights(weights, open_prices, equity)

    orders_df = reconcile_orders(state, targets)
    execute_orders(
        state=state,
        orders_df=orders_df,
        prices=open_prices,
        date=date,
        commission_per_trade=commission_per_trade,
    )

    state.mark_to_market(date, open_prices)
    try:
        state.last_price_snapshot = dict(open_prices)
    except Exception:
        state.last_price_snapshot = open_prices
    return orders_df
