# execution/portfolio_state.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from utils.formatting import truncate_value


def _normalize_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.to_pydatetime()
    if value is None:
        return datetime.now()
    return pd.to_datetime(value).to_pydatetime()


@dataclass
class PortfolioState:
    cash: float = 100_000.0
    positions: Dict[str, float] = field(default_factory=dict)
    position_cost_basis: Dict[str, float] = field(default_factory=dict)
    initial_capital: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    portfolio_value_history: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["Date", "Value", "HoldingsValue", "Cash", "RealizedPnL", "UnrealizedPnL", "FeesPaid"]
        )
    )
    trade_log: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Side",
                "Qty",
                "Price",
                "Notional",
                "RealizedPnL",
                "Fee",
                "CashAfter",
            ]
        )
    )
    last_allocation: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["Ticker", "Weight"])
    )
    signal_log: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "Date",
                "Signal_Date",
                "Execution_Date",
                "Data_Through",
                "Ticker",
                "Signal",
                "Old_Weight",
                "New_Weight",
                "Reason",
                "Timestamp",
            ]
        )
    )
    cycle_log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.initial_capital is None:
            self.initial_capital = float(self.cash)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def position_qty(self, ticker: str) -> float:
        return float(self.positions.get(ticker, 0.0))

    def update_position(self, ticker: str, qty_delta: float, price: float, side: str) -> float:
        """
        Apply a trade to holdings and update cash/cost basis.

        Returns the realized P&L contribution from the trade (sells only).
        """
        if qty_delta == 0:
            return 0.0

        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side '{side}' for {ticker}.")

        realized_delta = 0.0
        if side == "BUY":
            cost = price * qty_delta
            if cost - 1e-9 > self.cash:
                raise ValueError("Insufficient cash for purchase.")
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0.0) + qty_delta
            self.position_cost_basis[ticker] = self.position_cost_basis.get(ticker, 0.0) + cost
        else:
            current_qty = self.positions.get(ticker, 0.0)
            shares_to_sell = abs(qty_delta)
            if shares_to_sell - current_qty > 1e-9:
                raise ValueError(f"Attempting to sell {shares_to_sell} shares of {ticker} but only {current_qty} held.")
            total_cost = self.position_cost_basis.get(ticker, 0.0)
            cost_per_share = total_cost / current_qty if current_qty else 0.0
            cost_removed = cost_per_share * shares_to_sell
            proceeds = shares_to_sell * price
            realized_delta = proceeds - cost_removed
            self.realized_pnl += realized_delta
            self.cash += proceeds
            remaining_cost = total_cost - cost_removed
            remaining_qty = current_qty - shares_to_sell
            if remaining_qty <= 1e-8:
                self.positions.pop(ticker, None)
                self.position_cost_basis.pop(ticker, None)
            else:
                self.positions[ticker] = remaining_qty
                self.position_cost_basis[ticker] = max(remaining_cost, 0.0)

        return realized_delta

    # ------------------------------------------------------------------
    # Fees & logging
    # ------------------------------------------------------------------
    def deduct_fee(self, amount: float):
        if amount <= 0:
            return
        if amount - 1e-9 > self.cash:
            raise ValueError("Insufficient cash to pay trading fee.")
        self.cash -= amount
        self.fees_paid += amount

    def append_trade(self, row: dict):
        entry = {}
        for col in self.trade_log.columns:
            entry[col] = row.get(col)
        self.trade_log.loc[len(self.trade_log)] = entry

    # ------------------------------------------------------------------
    # Valuation utilities
    # ------------------------------------------------------------------
    def _require_price_snapshot(self, price_snapshot: dict[str, float]):
        missing = [ticker for ticker in self.positions if ticker not in price_snapshot]
        if missing:
            raise ValueError(f"Missing prices for holdings: {', '.join(sorted(missing))}")

    def current_equity(self, price_snapshot: dict[str, float]) -> float:
        self._require_price_snapshot(price_snapshot)
        holdings_value = sum(price_snapshot[t] * qty for t, qty in self.positions.items())
        return holdings_value + self.cash

    def mark_to_market(self, date, price_snapshot: dict[str, float]) -> float:
        """
        Compute portfolio MV using provided open-price snapshot,
        update unrealized P&L, and verify reconciliation identity.
        """
        self._require_price_snapshot(price_snapshot)
        holdings_value = sum(price_snapshot[t] * qty for t, qty in self.positions.items())
        unrealized_components = []
        for ticker, qty in self.positions.items():
            basis = self.position_cost_basis.get(ticker, 0.0)
            unrealized_components.append(price_snapshot[ticker] * qty - basis)
        self.unrealized_pnl = float(sum(unrealized_components))
        total_value = holdings_value + self.cash

        lhs = holdings_value + self.cash - float(self.initial_capital)
        rhs = self.realized_pnl + self.unrealized_pnl - self.fees_paid
        if abs(lhs - rhs) > 1e-4:
            raise ValueError(
                f"Reconciliation failed on {date}: holdings+cash-initial ({lhs:.4f}) "
                f"!= realized+unrealized-fees ({rhs:.4f})"
            )

        normalized_date = _normalize_date(date)
        self.portfolio_value_history.loc[len(self.portfolio_value_history)] = [
            normalized_date,
            truncate_value(total_value, decimals=4),
            truncate_value(holdings_value, decimals=4),
            truncate_value(self.cash, decimals=4),
            truncate_value(self.realized_pnl, decimals=4),
            truncate_value(self.unrealized_pnl, decimals=4),
            truncate_value(self.fees_paid, decimals=4),
        ]
        self.cycle_log.append(
            {
                "Date": normalized_date,
                "HoldingsValue": truncate_value(holdings_value, decimals=4),
                "Cash": truncate_value(self.cash, decimals=4),
                "RealizedPnL": truncate_value(self.realized_pnl, decimals=4),
                "UnrealizedPnL": truncate_value(self.unrealized_pnl, decimals=4),
                "FeesPaid": truncate_value(self.fees_paid, decimals=4),
            }
        )
        return total_value
