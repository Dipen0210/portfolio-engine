import pandas as pd
from datetime import datetime


class PortfolioState:
    """
    Tracks holdings, cash, and portfolio performance across rebalancing cycles.
    Acts as the persistent state object for the trading engine.
    """

    def __init__(self, cash: float = 100_000.0):
        # Initial state
        self.cash = cash
        self.holdings = {}  # {ticker: {'shares': float, 'price': float}}
        self.trade_log = []  # list of executed trades
        self.portfolio_history = []  # list of snapshots over time

    # --------------------------------------------------------
    # ✅ Record keeping
    # --------------------------------------------------------
    def log_trade(self, ticker, side, shares, price):
        trade = {
            "timestamp": datetime.now(),
            "ticker": ticker,
            "side": side,  # 'BUY' or 'SELL'
            "shares": shares,
            "price": price,
            "value": shares * price,
        }
        self.trade_log.append(trade)

    def snapshot(self, market_prices: dict):
        """
        Record a snapshot of the portfolio value.
        """
        total_value = self.cash
        for ticker, pos in self.holdings.items():
            if ticker in market_prices:
                total_value += pos["shares"] * market_prices[ticker]
        self.portfolio_history.append(
            {
                "timestamp": datetime.now(),
                "total_value": total_value,
                "cash": self.cash,
                "holdings_value": total_value - self.cash,
            }
        )

    # --------------------------------------------------------
    # ✅ Portfolio management
    # --------------------------------------------------------
    def update_position(self, ticker, side, shares, price):
        """
        Update holdings after a trade is executed.
        """
        if side == "BUY":
            cost = shares * price
            if self.cash < cost:
                raise ValueError("Not enough cash to complete purchase.")
            self.cash -= cost
            if ticker not in self.holdings:
                self.holdings[ticker] = {"shares": 0.0, "price": 0.0}
            prev_shares = self.holdings[ticker]["shares"]
            self.holdings[ticker]["shares"] = prev_shares + shares
            self.holdings[ticker]["price"] = price

        elif side == "SELL":
            if ticker not in self.holdings or self.holdings[ticker]["shares"] < shares:
                raise ValueError("Not enough shares to sell.")
            revenue = shares * price
            self.cash += revenue
            self.holdings[ticker]["shares"] -= shares
            if self.holdings[ticker]["shares"] <= 0:
                del self.holdings[ticker]

        # log every trade
        self.log_trade(ticker, side, shares, price)

    # --------------------------------------------------------
    # ✅ Portfolio valuation
    # --------------------------------------------------------
    def total_value(self, market_prices: dict) -> float:
        total = self.cash
        for ticker, pos in self.holdings.items():
            if ticker in market_prices:
                total += pos["shares"] * market_prices[ticker]
        return total

    def current_weights(self, market_prices: dict) -> pd.DataFrame:
        """
        Compute portfolio weights based on current market prices.
        """
        total_val = self.total_value(market_prices)
        weights = []
        for ticker, pos in self.holdings.items():
            weight = (pos["shares"] * market_prices.get(ticker,
                      pos["price"])) / total_val
            weights.append({"Ticker": ticker, "Weight": weight})
        return pd.DataFrame(weights)
