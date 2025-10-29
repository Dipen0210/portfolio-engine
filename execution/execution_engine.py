# execution/execution_engine.py
import pandas as pd
import numpy as np
from datetime import datetime
from .portfolio_state import PortfolioState
from .transaction_cost import calc_commission, apply_slippage


def snapshot_prices(price_data_dict: dict) -> dict:
    """
    Get latest close/last for each ticker from your OHLCV dict.
    price_data_dict: {ticker: df with 'Close'}
    returns: {ticker: last_price}
    """
    snap = {}
    for t, df in price_data_dict.items():
        if "Close" in df.columns and len(df) > 0:
            snap[t] = float(df["Close"].iloc[-1])
    return snap


def target_shares_from_weights(target_weights: pd.DataFrame, prices: dict, equity: float) -> dict:
    """
    Convert target weights to integer share targets.
    target_weights: DataFrame with ['Ticker','Weight'] summing to 1
    prices: {ticker: price}
    equity: total portfolio equity (cash + positions marked)
    """
    targets = {}
    for _, row in target_weights.iterrows():
        t, w = row["Ticker"], float(row["Weight"])
        p = prices.get(t, np.nan)
        if np.isnan(p) or p <= 0:
            continue
        notional = equity * w
        qty = int(notional // p)  # floor to whole shares
        if qty > 0:
            targets[t] = qty
        else:
            targets[t] = 0
    return targets


def reconcile_orders(state: PortfolioState, target_shares: dict) -> pd.DataFrame:
    """
    Create order list by comparing current positions vs target shares.
    Returns DataFrame: ['Ticker','Side','Qty']
    """
    orders = []
    current = state.positions.copy()

    tickers = set(current.keys()) | set(target_shares.keys())
    for t in tickers:
        cur = int(current.get(t, 0))
        tgt = int(target_shares.get(t, 0))
        delta = tgt - cur
        if delta > 0:
            orders.append({"Ticker": t, "Side": "BUY", "Qty": delta})
        elif delta < 0:
            orders.append({"Ticker": t, "Side": "SELL", "Qty": abs(delta)})
    return pd.DataFrame(orders)


def execute_orders(state: PortfolioState,
                   orders_df: pd.DataFrame,
                   prices: dict,
                   date=None,
                   commission_per_trade: float = 1.0,
                   commission_bps: float = 0.0,
                   slippage_bps: float = 5.0):
    """
    Fills orders against current prices with slippage & commission, updates state.
    """
    if orders_df is None or len(orders_df) == 0:
        return

    date = date or datetime.now().strftime("%Y-%m-%d")

    for _, od in orders_df.iterrows():
        t = od["Ticker"]
        side = od["Side"].upper()
        qty = int(od["Qty"])
        mid = float(prices.get(t, np.nan))
        if np.isnan(mid) or qty <= 0:
            continue

        # Slippage-adjusted fill
        fill_price = apply_slippage(mid, side, slippage_bps=slippage_bps)
        gross = fill_price * qty * (1 if side == "SELL" else -1)

        # Commission on notional (absolute)
        commission = calc_commission(order_value=fill_price * qty,
                                     per_trade=commission_per_trade,
                                     bps=commission_bps)

        # Net cash impact
        # SELL: cash in (gross positive) - commission; BUY: negative - commission
        net = gross - commission

        # Update state
        state.cash += net
        state.update_position(t, qty if side == "BUY" else -qty)

        # Log
        state.append_trade({
            "Date": date,
            "Ticker": t,
            "Side": side,
            "Qty": qty,
            "Price": round(fill_price, 6),
            "Gross": round(gross, 2),
            "Commission": round(commission, 2),
            "Slippage": round((fill_price - mid) * (1 if side == "BUY" else -1) * qty, 2),
            "Net": round(net, 2),
        })


def run_rebalance_cycle(
    state: PortfolioState,
    price_data_dict: dict,
    # ['Ticker','Weight'] (weights sum ≈ 1)
    new_portfolio_weights: pd.DataFrame,
    date=None,
    commission_per_trade: float = 1.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 5.0,
):
    """
    Master function for a rebalance:
      1) Mark-to-market to compute equity
      2) Convert target weights -> target shares
      3) Create orders (delta shares)
      4) Execute orders -> update state
      5) Mark-to-market post-trade (optional)
    Returns orders_df for inspection.
    """
    date = date or (list(price_data_dict.values())[
                    0].index[-1] if price_data_dict else None)

    # 1) Mark current equity
    prices = snapshot_prices(price_data_dict)
    equity = state.mark_to_market(date, prices)

    # 2) Target shares
    tw = new_portfolio_weights.copy()
    tw["Weight"] = tw["Weight"] / tw["Weight"].sum()  # ensure normalized
    targets = target_shares_from_weights(tw, prices, equity)

    # 3) Orders
    orders_df = reconcile_orders(state, targets)

    # 4) Execute orders
    execute_orders(
        state=state,
        orders_df=orders_df,
        prices=prices,
        date=date,
        commission_per_trade=commission_per_trade,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )

    # 5) Post-trade mark
    state.mark_to_market(date, prices)

    return orders_df
