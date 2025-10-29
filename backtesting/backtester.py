# backtesting/backtester.py
import pandas as pd
import numpy as np
from .metrics import compute_performance_metrics


def _prepare_close_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Ensure the OHLCV DataFrame is indexed by datetime and return the Close series.
    """
    if "Close" not in df.columns:
        return None

    working = df.copy()
    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"])
        working = working.set_index("Date")
    if not isinstance(working.index, pd.DatetimeIndex):
        try:
            working.index = pd.to_datetime(working.index)
        except Exception:
            return None

    working = working.sort_index()
    return working["Close"].rename(ticker)


def backtest_portfolio(
    price_data_dict,
    weights_df: pd.DataFrame,
    rebalance_freq: str = "M",
    initial_capital: float = 100000,
    benchmark_df: pd.DataFrame | None = None,
    start_date=None,
    end_date=None,
):
    """
    Simulate a rebalancing backtest with explicit transaction logging.

    Parameters
    ----------
    price_data_dict : dict[str, pd.DataFrame]
        Mapping of ticker to OHLCV DataFrame (must include 'Close').
    weights_df : pd.DataFrame
        Optimised weights with columns ['Ticker', 'Weight'].
    rebalance_freq : str, optional
        Pandas offset alias marking rebalance cadence (default 'M').
    initial_capital : float, optional
        Starting portfolio value.
    benchmark_df : pd.DataFrame, optional
        OHLCV DataFrame for benchmark asset.
    start_date : datetime/date/str, optional
        Inclusive backtest start.
    end_date : datetime/date/str, optional
        Inclusive backtest end.

    Returns
    -------
    dict
        Contains portfolio curve, metrics, benchmark comparison, transaction
        history, and rebalance diagnostics.
    """

    if weights_df.empty:
        raise ValueError("Backtest failed: no weights provided for simulation.")

    weights = (
        weights_df.set_index("Ticker")["Weight"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    weights = weights[weights > 0]
    if weights.empty:
        raise ValueError("Backtest failed: weights must contain positive allocations.")

    close_series = []
    for ticker in weights.index:
        df = price_data_dict.get(ticker)
        if df is None or df.empty:
            continue
        series = _prepare_close_series(df, ticker)
        if series is not None and not series.empty:
            close_series.append(series)

    if not close_series:
        raise ValueError("Backtest failed: no valid price history for the selected tickers.")

    close_prices = pd.concat(close_series, axis=1)
    close_prices = close_prices.sort_index()
    close_prices = close_prices.loc[~close_prices.index.duplicated()]
    close_prices = close_prices.dropna(axis=1, how="all")

    if close_prices.empty:
        raise ValueError("Backtest failed: price history is empty after alignment.")

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
    else:
        start_ts = close_prices.index[0]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
    else:
        end_ts = close_prices.index[-1]

    if start_ts > end_ts:
        raise ValueError("Backtest failed: start date occurs after end date.")

    window_mask = (close_prices.index >= start_ts) & (close_prices.index <= end_ts)
    close_prices = close_prices.loc[window_mask]
    close_prices = close_prices.ffill().dropna()

    available_tickers = [t for t in weights.index if t in close_prices.columns]
    if not available_tickers:
        raise ValueError("Backtest failed: no overlapping price history within the selected window.")

    close_prices = close_prices[available_tickers]
    weights = weights.reindex(available_tickers)
    weights = weights / weights.sum()

    trading_days = close_prices.index
    if len(trading_days) == 0:
        raise ValueError("Backtest failed: no trading days available in the window.")

    def _align_to_trading_day(candidate: pd.Timestamp) -> pd.Timestamp | None:
        positions = trading_days.get_indexer([candidate], method="backfill")
        if positions.size == 0:
            return None
        loc = positions[0]
        if loc == -1:
            return None
        return trading_days[loc]

    schedule = [trading_days[0]]
    rebalance_rule = (rebalance_freq or "").strip().upper()
    if rebalance_rule:
        raw_schedule = pd.date_range(
            start=trading_days[0],
            end=trading_days[-1],
            freq=rebalance_rule,
        )
        for candidate in raw_schedule[1:]:
            aligned = _align_to_trading_day(candidate)
            if aligned is not None and aligned != schedule[-1]:
                schedule.append(aligned)
    rebalance_candidates = schedule[1:]
    rebalance_candidate_set = set(rebalance_candidates)

    holdings = pd.Series(0.0, index=weights.index)
    transactions: list[dict] = []
    portfolio_records: list[dict] = []
    executed_rebalance_dates: list[pd.Timestamp] = []

    for idx, day in enumerate(trading_days):
        prices = close_prices.loc[day]

        if idx == 0:
            for ticker, weight in weights.items():
                price = prices[ticker]
                if pd.isna(price) or price <= 0:
                    continue
                allocation_value = initial_capital * weight
                shares = allocation_value / price if price else 0.0
                holdings.at[ticker] = shares
                trade_value = shares * price
                transactions.append(
                    {
                        "Date": day,
                        "Event": "Initial Allocation",
                        "Ticker": ticker,
                        "Action": "BUY",
                        "Shares": shares,
                        "Price": price,
                        "TradeValue": trade_value,
                        "CashFlow": -trade_value,
                    }
                )
        elif day in rebalance_candidate_set:
            portfolio_value = float((holdings * prices).sum())
            trades_executed = False
            for ticker, weight in weights.items():
                price = prices[ticker]
                if pd.isna(price) or price <= 0:
                    continue
                target_value = portfolio_value * weight
                target_shares = target_value / price if price else 0.0
                delta_shares = target_shares - holdings.at[ticker]
                if abs(delta_shares) > 1e-9:
                    trades_executed = True
                    action = "BUY" if delta_shares > 0 else "SELL"
                    trade_value = abs(delta_shares) * price
                    cash_flow = -trade_value if delta_shares > 0 else trade_value
                    transactions.append(
                        {
                            "Date": day,
                            "Event": "Scheduled Rebalance",
                            "Ticker": ticker,
                            "Action": action,
                            "Shares": delta_shares,
                            "Price": price,
                            "TradeValue": trade_value,
                            "CashFlow": cash_flow,
                        }
                    )
                holdings.at[ticker] = target_shares
            if trades_executed:
                executed_rebalance_dates.append(day)

        portfolio_value = float((holdings * prices).sum())
        portfolio_records.append(
            {
                "Date": day,
                "Portfolio Value": portfolio_value,
            }
        )

    final_day = trading_days[-1]
    final_prices = close_prices.loc[final_day]
    for ticker, shares in holdings.items():
        if abs(shares) > 1e-9:
            price = final_prices[ticker]
            trade_value = abs(shares) * price
            transactions.append(
                {
                    "Date": final_day,
                    "Event": "Final Liquidation",
                    "Ticker": ticker,
                    "Action": "SELL",
                    "Shares": -shares,
                    "Price": price,
                    "TradeValue": trade_value,
                    "CashFlow": trade_value,
                }
            )
            holdings.at[ticker] = 0.0

    portfolio_df = pd.DataFrame(portfolio_records).set_index("Date")
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    portfolio_df["Portfolio Return"] = (
        portfolio_df["Portfolio Value"].pct_change().fillna(0.0)
    )

    portfolio_returns = portfolio_df["Portfolio Return"]
    metrics = compute_performance_metrics(portfolio_returns) if not portfolio_returns.empty else {}

    benchmark_curve = None
    benchmark_metrics = None
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_series = _prepare_close_series(benchmark_df, "Benchmark")
        if benchmark_series is not None and not benchmark_series.empty:
            benchmark_series = benchmark_series.reindex(portfolio_df.index).ffill().dropna()
            if not benchmark_series.empty:
                benchmark_returns = benchmark_series.pct_change().fillna(0.0)
                benchmark_value = (1 + benchmark_returns).cumprod() * initial_capital
                benchmark_curve = pd.DataFrame(
                    {
                        "Benchmark Value": benchmark_value,
                        "Benchmark Return": benchmark_returns,
                    }
                )
                benchmark_metrics = compute_performance_metrics(benchmark_returns)

    transactions_df = pd.DataFrame(transactions)
    if not transactions_df.empty:
        transactions_df["Date"] = pd.to_datetime(transactions_df["Date"])
        for col in ["Shares", "Price", "TradeValue", "CashFlow"]:
            transactions_df[col] = pd.to_numeric(transactions_df[col], errors="coerce")
        transactions_df["Transaction Cost"] = (
            transactions_df["TradeValue"].abs() - transactions_df["CashFlow"].abs()
        )
        transactions_df["Transaction Cost"] = transactions_df["Transaction Cost"].round(4)
        transactions_df = transactions_df.sort_values(["Date", "Event", "Ticker"]).reset_index(drop=True)

    summary = {
        "initial_capital": float(initial_capital),
        "final_value": float(portfolio_df["Portfolio Value"].iloc[-1]),
        "return_amount": float(portfolio_df["Portfolio Value"].iloc[-1] - initial_capital),
        "return_pct": float(portfolio_df["Portfolio Value"].iloc[-1] / initial_capital - 1),
        "rebalance_count": len(executed_rebalance_dates),
        "start_date": portfolio_df.index[0],
        "end_date": portfolio_df.index[-1],
    }

    return {
        "portfolio": portfolio_df,
        "metrics": metrics,
        "benchmark": benchmark_curve,
        "benchmark_metrics": benchmark_metrics,
        "transactions": transactions_df,
        "rebalance_dates": executed_rebalance_dates,
        "rebalance_count": len(executed_rebalance_dates),
        "summary": summary,
    }
