# execution/portfolio_state.py
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PortfolioState:
    cash: float = 100_000.0
    positions: dict = field(default_factory=dict)  # {ticker: shares}
    # optional book-keeping
    portfolio_value_history: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["Date", "Value"]))
    trade_log: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["Date", "Ticker", "Side", "Qty", "Price",
                 "Gross", "Commission", "Slippage", "Net"]
    ))
    last_allocation: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["Ticker", "Weight"])
    )
    signal_log: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["Date", "Ticker", "Signal", "Old_Weight", "New_Weight", "Reason", "Timestamp"]
        )
    )

    def position_qty(self, ticker: str) -> int:
        return int(self.positions.get(ticker, 0))

    def update_position(self, ticker: str, qty_delta: int):
        new_qty = self.position_qty(ticker) + qty_delta
        if new_qty == 0:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = new_qty

    def mark_to_market(self, date, price_snapshot: dict):
        """
        price_snapshot: {ticker: last_price}
        """
        pos_val = sum(price_snapshot.get(t, 0.0) *
                      q for t, q in self.positions.items())
        total = self.cash + pos_val
        self.portfolio_value_history.loc[len(self.portfolio_value_history)] = [
            date, total]
        return total

    def append_trade(self, row: dict):
        self.trade_log.loc[len(self.trade_log)] = row
