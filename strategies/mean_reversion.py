# strategies/mean_reversion.py

from __future__ import annotations

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.indicators import RSI


class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="Mean Reversion")

    def generate_signals(self, df):
        df = RSI(df)
        df['Signal'] = 0
        df.loc[df['RSI_14'] < 30, 'Signal'] = 1   # buy
        df.loc[df['RSI_14'] > 70, 'Signal'] = -1  # sell
        return df

    def score_stock(self, df):
        return 100 - df['RSI_14'].iloc[-1]  # lower RSI = higher score


def run_mean_reversion_strategy(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Execute the mean reversion strategy across a dictionary of OHLCV DataFrames.
    Returns a DataFrame with Strategy_Score and Rank per ticker.
    """
    strategy = MeanReversionStrategy()
    results: list[dict[str, float | str]] = []

    for ticker, df in stock_data_dict.items():
        if df.empty:
            continue
        try:
            enriched = strategy.generate_signals(df.copy())
            score = float(strategy.score_stock(enriched))
        except Exception:
            score = 0.0
        results.append({"Ticker": ticker, "Strategy_Score": score})

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df["Strategy_Score"] = result_df["Strategy_Score"].fillna(0.0)
    result_df = result_df.sort_values("Strategy_Score", ascending=False).reset_index(drop=True)
    result_df["Rank"] = result_df.index + 1
    return result_df
