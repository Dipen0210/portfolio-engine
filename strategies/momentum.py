# strategies/momentum.py

from __future__ import annotations

import math

import pandas as pd

from .indicators import EMA, MACD, RSI, SMA


def _clip(value: float, limit: float = 1.0) -> float:
    """Clamp value to [-limit, limit]."""
    if math.isnan(value):
        return 0.0
    return max(-limit, min(limit, value))


def calculate_momentum_score(df: pd.DataFrame) -> float:
    """
    Compute a momentum score combining several smoothed signals.

    The score ranges roughly from -1.0 (very bearish) to +1.0 (very bullish)
    by averaging four normalized sub-scores:
        - Medium-term price change (20-trading-day return)
        - Price distance to SMA-20
        - RSI deviation from neutral (50)
        - MACD histogram strength
    """

    if df is None or df.empty:
        return 0.0

    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < 30:
        # Not enough data for the indicators to stabilize; neutral score.
        return 0.0

    df = SMA(df.copy(), period=20)
    df = EMA(df, period=20)
    df = RSI(df, period=14)
    df = MACD(df)

    close = df["Close"].iloc[-1]
    sma20 = df["SMA_20"].iloc[-1]
    ema20 = df["EMA_20"].iloc[-1]
    rsi14 = df["RSI_14"].iloc[-1]
    macd_line = df["MACD"].iloc[-1]
    macd_signal = df["MACD_signal"].iloc[-1]

    # 1) Medium-term price momentum (20-day percent change, clipped at ±20%)
    if len(df) >= 21 and df["Close"].iloc[-21] > 0:
        ret_20 = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1.0
    else:
        ret_20 = 0.0
    score_return = _clip(ret_20 / 0.20)

    # 2) Distance of price to SMA/EMA trend band (favoring price above both)
    trend_numer = (close - sma20) + (close - ema20)
    denom = max(abs(sma20), abs(ema20), 1e-9)
    score_trend = _clip((trend_numer / denom) / 0.10)

    # 3) RSI deviation from neutral 50 (scaled so RSI 70 => +0.8, RSI 30 => -0.8)
    score_rsi = _clip((rsi14 - 50.0) / 25.0)

    # 4) MACD histogram strength relative to price (scaled so 1% of price => score 1)
    macd_hist = macd_line - macd_signal
    macd_denom = abs(close) * 0.01 if close else 1.0
    score_macd = _clip(macd_hist / macd_denom)

    aggregate = (score_return + score_trend + score_rsi + score_macd) / 4.0
    return float(round(aggregate, 4))


def run_momentum_strategy(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run the momentum strategy for each ticker and return a scored ranking.
    """
    results: list[dict[str, float | str]] = []
    for ticker, df in stock_data_dict.items():
        try:
            score = calculate_momentum_score(df)
        except Exception:
            score = 0.0
        results.append({"Ticker": ticker, "Strategy_Score": score})

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df["Strategy_Score"] = pd.to_numeric(
        result_df["Strategy_Score"], errors="coerce"
    ).fillna(0.0)
    result_df = result_df.sort_values("Strategy_Score", ascending=False).reset_index(
        drop=True
    )
    result_df["Rank"] = result_df.index + 1
    return result_df
