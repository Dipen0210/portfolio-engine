# rebalancing/rebalance_controller.py

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable

import pandas as pd
import yfinance as yf

from hardfilters.filters import filter_universe_by_risk_and_sector
from strategies.registry import resolve_strategy
from forecasting.forecast import compute_expected_returns
from risk.risk_models import compute_log_returns, compute_covariance_matrix
from risk.risk_pipeline import build_risk_report
from optimization.weight_optimizer import mean_variance_optimize
from optimization.hybrid_allocator import allocate_hybrid_weights
from backtesting.backtester import backtest_portfolio
from signals.signal_engine import generate_portfolio_signals

DEFAULT_TOP_N_PER_SECTOR = 10
FINAL_SELECTION_PER_SECTOR = 3
BENCHMARK_TICKER = "^GSPC"


def _ensure_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def fetch_stock_data(
    tickers: Iterable[str],
    start_date: datetime,
    end_date: datetime,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple tickers in batch."""
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
        )
        if df.empty:
            continue
        df = df.dropna().reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            flattened_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    primary = col[0] if col[0] else col[-1]
                    flattened_cols.append(primary)
                else:
                    flattened_cols.append(col)
            df.columns = flattened_cols
        data[ticker] = df
    return data


def _build_price_matrix(price_history: Dict[str, pd.DataFrame], tickers: Iterable[str]) -> pd.DataFrame:
    """Construct a price matrix (index=date, columns=tickers) of Close prices."""
    series_list = []
    for ticker in tickers:
        df = price_history.get(ticker)
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            ser = df.sort_values("Date").set_index("Date")["Close"]
        else:
            ser = df.sort_index()["Close"]

        if isinstance(ser, pd.DataFrame):
            # yfinance can occasionally return a single-column DataFrame; squeeze to Series
            ser = ser.squeeze("columns")
            if isinstance(ser, pd.DataFrame):
                ser = ser.iloc[:, 0]

        ser = ser.rename(ticker)
        series_list.append(ser)
    if not series_list:
        return pd.DataFrame()

    price_df = pd.concat(series_list, axis=1).dropna(how="all")
    price_df = price_df.loc[:, ~price_df.columns.duplicated()]
    return price_df


def run_rebalance_cycle(
    state,
    company_universe_df: pd.DataFrame,
    user_strategy: str = "Momentum",
    user_sectors: list[str] | None = None,
    risk_level: str = "Medium",
    capital: float = 100_000,
    as_of_date=None,
    execution_date=None,
    backtest_start_date=None,
    backtest_end_date=None,
    lookback_window: int | None = None,
    top_n_per_sector: int = DEFAULT_TOP_N_PER_SECTOR,
    top_k_final: int = FINAL_SELECTION_PER_SECTOR,
    backtest_rebalance_freq: str = "M",
    drift_threshold: float = 0.03,
):
    """
    Build the candidate portfolio by filtering the universe, running the
    selected strategy per sector, and returning the top-ranked picks as of a
    specific evaluation date.
    """

    try:
        strategy_spec, strategy_runner = resolve_strategy(user_strategy)
    except NotImplementedError as exc:
        raise ValueError(str(exc)) from exc

    strategy_lookback = max(
        2, lookback_window if lookback_window is not None else strategy_spec.lookback_window
    )

    analysis_dt = _ensure_datetime(as_of_date) or datetime.today()
    execution_dt = _ensure_datetime(execution_date) or analysis_dt
    requested_backtest_start = _ensure_datetime(backtest_start_date)

    if execution_dt < analysis_dt:
        raise ValueError("Execution date must be on or after the analysis date.")
    if requested_backtest_start and requested_backtest_start > analysis_dt:
        requested_backtest_start = analysis_dt

    lookback_buffer = max(strategy_lookback * 2, strategy_lookback + 5)
    strategy_start_dt = analysis_dt - timedelta(days=lookback_buffer)
    start_dt = strategy_start_dt
    if requested_backtest_start:
        start_dt = min(strategy_start_dt, requested_backtest_start)
    backtest_start_dt = requested_backtest_start or strategy_start_dt
    backtest_end_dt = execution_dt

    previous_allocation = getattr(
        state, "last_allocation", pd.DataFrame(columns=["Ticker", "Weight"])
    )
    if not hasattr(state, "last_allocation"):
        state.last_allocation = previous_allocation.copy()
    trade_signals: pd.DataFrame | None = None
    # 1) Filter universe by sector and risk preferences.
    sector_slices = filter_universe_by_risk_and_sector(
        company_df=company_universe_df,
        sectors=user_sectors,
        risk_level=risk_level,
        top_n_per_sector=top_n_per_sector,
    )
    if not sector_slices:
        raise ValueError("No stocks remain after applying filters.")

    candidate_frames = [
        df.assign(sector=sector) for sector, df in sector_slices.items()
    ]
    candidate_pool_df = (
        pd.concat(candidate_frames, ignore_index=True)
        .sort_values(by=["sector", "market cap"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # 2) Fetch OHLCV data for all unique tickers in the candidate pool.
    ordered_tickers = candidate_pool_df["ticker"].tolist()
    seen = set()
    unique_tickers = []
    for ticker in ordered_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)

    fetch_end_dt = max(analysis_dt, execution_dt) + timedelta(days=2)

    price_history = fetch_stock_data(
        unique_tickers,
        start_date=start_dt - timedelta(days=5),
        end_date=fetch_end_dt,
    )
    missing_tickers = [t for t in unique_tickers if t not in price_history]

    benchmark_history = fetch_stock_data(
        [BENCHMARK_TICKER],
        start_date=start_dt - timedelta(days=5),
        end_date=fetch_end_dt,
    )
    benchmark_df = benchmark_history.get(BENCHMARK_TICKER)

    end_ts = pd.Timestamp(analysis_dt)
    cutoff_date = end_ts.normalize()
    analysis_price_history: Dict[str, pd.DataFrame] = {}
    analysis_last_date: pd.Timestamp | None = None
    for ticker, df in price_history.items():
        if df is None or df.empty:
            continue
        temp = df.copy()
        if "Date" in temp.columns:
            temp["Date"] = pd.to_datetime(temp["Date"])
            truncated = temp[temp["Date"] <= end_ts]
            last_available = truncated["Date"].max() if not truncated.empty else None
        else:
            truncated = temp.copy()
            truncated.index = pd.to_datetime(truncated.index)
            truncated = truncated[truncated.index <= end_ts]
            last_available = truncated.index.max() if not truncated.empty else None
        if not truncated.empty:
            analysis_price_history[ticker] = truncated.copy()
            if last_available is not None:
                last_available = pd.Timestamp(last_available).normalize()
                if analysis_last_date is None or last_available > analysis_last_date:
                    analysis_last_date = last_available

    sector_price_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sector, sector_df in sector_slices.items():
        sector_tickers = [t for t in sector_df["ticker"] if t in analysis_price_history]
        if sector_tickers:
            sector_price_data[sector] = {t: analysis_price_history[t] for t in sector_tickers}

    # 3) Run strategy scoring per sector.
    sector_rankings: Dict[str, pd.DataFrame] = {}
    final_selection_frames = []

    for sector, sector_df in sector_slices.items():
        sector_data = sector_price_data.get(sector, {})
        if not sector_data:
            continue

        ranking = strategy_runner(sector_data)
        if ranking.empty:
            continue

        ranking["Sector"] = sector
        sector_rankings[sector] = ranking

        final_selection_frames.append(ranking.head(top_k_final))

    final_portfolio_df = (
        pd.concat(final_selection_frames, ignore_index=True)
        if final_selection_frames
        else pd.DataFrame(columns=["Ticker", "Strategy_Score", "Rank", "Sector"])
    )
    if not final_portfolio_df.empty and "Strategy_Score" in final_portfolio_df.columns:
        final_portfolio_df["Strategy_Score"] = (
            pd.to_numeric(final_portfolio_df["Strategy_Score"], errors="coerce")
            .fillna(0.0)
        )

    expected_returns_df: pd.DataFrame | None = None
    expected_returns_series: pd.Series | None = None
    forecast_exclusions: list[str] = []
    price_matrix = pd.DataFrame()
    returns_matrix = pd.DataFrame()
    cov_matrix = pd.DataFrame()
    optimized_weights: pd.Series | None = None
    hybrid_allocation: pd.DataFrame | None = None
    risk_report: dict | None = None
    backtest_results: dict | None = None
    backtest_error: str | None = None
    allocation_carry_forward = False
    new_allocation = pd.DataFrame(columns=["Ticker", "Weight"])

    if not final_portfolio_df.empty:
        selected_tickers = final_portfolio_df["Ticker"].unique().tolist()
        selected_price_history = {
            ticker: analysis_price_history[ticker]
            for ticker in selected_tickers
            if ticker in analysis_price_history
        }
        if selected_price_history:
            min_series_len = min(len(df) for df in selected_price_history.values())
            forecast_window = max(2, min(strategy_lookback, min_series_len))
            expected_returns_df, expected_returns_series = compute_expected_returns(
                selected_price_history, window=forecast_window
            )

    if expected_returns_series is not None and not expected_returns_series.empty:
        available_tickers = expected_returns_series.index.tolist()
        missing_for_forecast = sorted(set(final_portfolio_df["Ticker"]) - set(available_tickers))
        forecast_exclusions = missing_for_forecast
        final_portfolio_df = (
            final_portfolio_df[final_portfolio_df["Ticker"].isin(available_tickers)]
            .reset_index(drop=True)
        )

        price_matrix = _build_price_matrix(analysis_price_history, available_tickers)
        if not price_matrix.empty:
            returns_matrix = compute_log_returns(price_matrix)
            returns_matrix = returns_matrix.dropna(how="all").dropna(axis=1, how="all")
            if not returns_matrix.empty:
                cov_matrix = compute_covariance_matrix(returns_matrix, method="ledoit_wolf")
            else:
                cov_matrix = pd.DataFrame()

        if not cov_matrix.empty:
            try:
                common = expected_returns_series.index.intersection(cov_matrix.index)
                common = common.intersection(cov_matrix.columns)
                mu_aligned = expected_returns_series.reindex(common).dropna()
                cov_aligned = cov_matrix.reindex(index=common, columns=common)
                cov_aligned = cov_aligned.dropna(how="all", axis=0).dropna(how="all", axis=1)
                common = cov_aligned.index.intersection(cov_aligned.columns)
                mu_aligned = mu_aligned.reindex(common).dropna()
                cov_aligned = cov_aligned.reindex(index=common, columns=common)
                if not mu_aligned.empty:
                    if len(mu_aligned) == 1:
                        optimized_weights = pd.Series(
                            [1.0], index=mu_aligned.index, name="weight"
                        )
                    elif not cov_aligned.empty and cov_aligned.shape[0] >= 2:
                        optimized_weights = mean_variance_optimize(
                            mu_aligned,
                            cov_aligned,
                            risk_level=risk_level,
                        )
                    # keep only tickers used in optimization downstream
                    final_portfolio_df = final_portfolio_df[
                        final_portfolio_df["Ticker"].isin(mu_aligned.index)
                    ].reset_index(drop=True)
            except ValueError:
                optimized_weights = None

        if optimized_weights is not None and not final_portfolio_df.empty:
            try:
                hybrid_allocation = allocate_hybrid_weights(
                    final_portfolio_df, optimized_weights
                )
            except Exception:
                hybrid_allocation = None
        weights_for_risk = None
        if hybrid_allocation is not None and not hybrid_allocation.empty:
            weights_for_risk = (
                hybrid_allocation.set_index("Ticker")["Final_Weight"]
                .reindex(price_matrix.columns)
                .dropna()
            )
        elif optimized_weights is not None and not optimized_weights.empty:
            weights_for_risk = optimized_weights.reindex(price_matrix.columns).dropna()

        new_allocation = pd.DataFrame(columns=["Ticker", "Weight"])
        if weights_for_risk is not None and not weights_for_risk.empty:
            new_allocation = (
                weights_for_risk.rename("Weight")
                .reset_index()
                .rename(columns={"index": "Ticker"})
            )
        elif hybrid_allocation is not None and not hybrid_allocation.empty:
            new_allocation = (
                hybrid_allocation.loc[:, ["Ticker", "Final_Weight"]]
                .rename(columns={"Final_Weight": "Weight"})
            )
        elif optimized_weights is not None and not optimized_weights.empty:
            new_allocation = (
                optimized_weights.rename("Weight")
                .reset_index()
                .rename(columns={"index": "Ticker"})
            )

        if new_allocation.empty and previous_allocation is not None and not previous_allocation.empty:
            preserved = previous_allocation.copy()
            if "CarryForward" in preserved.columns:
                preserved = preserved.drop(columns=["CarryForward"])
            if {"Ticker", "Weight"}.issubset(preserved.columns):
                preserved["Ticker"] = preserved["Ticker"].astype(str).str.strip()
                preserved = preserved.dropna(subset=["Ticker", "Weight"])
                preserved = preserved[preserved["Ticker"] != ""]
                if not preserved.empty:
                    preserved = (
                        preserved.groupby("Ticker", as_index=False)["Weight"]
                        .last()
                    )
                    new_allocation = preserved.copy()
                    allocation_carry_forward = True

        if weights_for_risk is not None and not weights_for_risk.empty and not price_matrix.empty:
            try:
                risk_report = build_risk_report(
                    price_matrix[weights_for_risk.index],
                    weights_for_risk,
                    mu=expected_returns_series.reindex(weights_for_risk.index) if expected_returns_series is not None else None,
                )
            except Exception:
                risk_report = None

        if not new_allocation.empty:
            new_allocation = new_allocation.copy()
            new_allocation["Ticker"] = new_allocation["Ticker"].astype(str).str.strip()
            new_allocation = new_allocation[
                ~new_allocation["Ticker"].str.lower().isin({"", "nan", "none"})
            ]
            new_allocation = new_allocation.dropna(subset=["Ticker", "Weight"])
            new_allocation["Weight"] = pd.to_numeric(new_allocation["Weight"], errors="coerce")
            new_allocation = new_allocation.dropna(subset=["Weight"])

        if not new_allocation.empty:
            preserve_cols = ["Ticker", "Weight"] + [col for col in new_allocation.columns if col not in {"Ticker", "Weight"}]
            new_allocation = (
                new_allocation.loc[:, preserve_cols]
                .sort_values(["Ticker"] + ([ "Date"] if "Date" in new_allocation.columns else []))
                .groupby("Ticker", as_index=False, sort=False)
                .last()
            )
            weight_sum = new_allocation["Weight"].sum()
            if weight_sum > 0:
                new_allocation["Weight"] = new_allocation["Weight"] / weight_sum

        if allocation_carry_forward and new_allocation.empty:
            new_allocation = previous_allocation.copy()

        if "Date" in new_allocation.columns:
            new_allocation = new_allocation.copy()
            new_allocation["Date"] = pd.to_datetime(new_allocation["Date"], errors="coerce")
            new_allocation["Date"] = new_allocation["Date"].fillna(execution_dt).dt.strftime("%Y-%m-%d")
        else:
            new_allocation = new_allocation.copy()
            new_allocation["Date"] = execution_dt.strftime("%Y-%m-%d")

        if allocation_carry_forward:
            new_allocation["CarryForward"] = True

        try:
            trade_signals = generate_portfolio_signals(
                previous_allocation,
                new_allocation,
                drift_threshold=drift_threshold,
                as_of_date=analysis_dt,
            )
        except Exception:
            trade_signals = None
        else:
            if trade_signals is not None and not trade_signals.empty:
                trade_signals = trade_signals.copy()
                trade_signals["Execution_Date"] = execution_dt.strftime("%Y-%m-%d")
                trade_signals["Data_Through"] = analysis_dt.strftime("%Y-%m-%d")
                trade_signals["Date"] = trade_signals["Execution_Date"]

        state.last_allocation = new_allocation.copy()
        if trade_signals is not None and not trade_signals.empty:
            try:
                state.signal_log = pd.concat(
                    [state.signal_log, trade_signals], ignore_index=True
                )
            except Exception:
                pass
        trade_log_df = getattr(state, "signal_log", pd.DataFrame()).copy()
        if not trade_log_df.empty:
            trade_log_df = trade_log_df.copy()
            if "Ticker" in trade_log_df.columns:
                trade_log_df["Ticker"] = trade_log_df["Ticker"].astype(str).str.strip()
            if "Date" in trade_log_df.columns:
                trade_log_df["Date"] = pd.to_datetime(
                    trade_log_df["Date"], errors="coerce"
                )
            if "Timestamp" in trade_log_df.columns:
                trade_log_df["Timestamp"] = pd.to_datetime(
                    trade_log_df["Timestamp"], errors="coerce"
                )
            trade_log_df = trade_log_df.dropna(subset=["Date", "Ticker"])
            sort_cols = ["Date", "Ticker"]
            if "Timestamp" in trade_log_df.columns:
                sort_cols.append("Timestamp")
            trade_log_df = trade_log_df.sort_values(sort_cols, kind="mergesort")
            trade_log_df = (
                trade_log_df.groupby(["Date", "Ticker"], as_index=False)
                .last()
                .reset_index(drop=True)
            )

        if backtest_results is None and not trade_log_df.empty:
            if "Execution_Date" in trade_log_df.columns:
                weight_history = (
                    trade_log_df.loc[:, ["Execution_Date", "Ticker", "New_Weight"]]
                    .rename(columns={"Execution_Date": "Date", "New_Weight": "Weight"})
                )
            else:
                weight_history = (
                    trade_log_df.loc[:, ["Date", "Ticker", "New_Weight"]]
                    .rename(columns={"New_Weight": "Weight"})
                )
            weight_history["Date"] = pd.to_datetime(weight_history["Date"], errors="coerce")
            weight_history["Weight"] = pd.to_numeric(weight_history["Weight"], errors="coerce")
            weight_history = weight_history.dropna(subset=["Date", "Ticker", "Weight"])
            if not weight_history.empty:
                weight_history["OriginalDate"] = weight_history["Date"]
                weight_history["Date"] = weight_history["Date"].dt.normalize()
                weight_history = (
                    weight_history.sort_values(["Date", "Ticker", "OriginalDate"])
                    .groupby(["Date", "Ticker"], as_index=False)
                    .last()
                )
                weight_history["Ticker"] = weight_history["Ticker"].astype(str)
                if backtest_start_dt is not None:
                    weight_history = weight_history[
                        weight_history["Date"] >= pd.Timestamp(backtest_start_dt)
                    ]
                if execution_dt is not None:
                    weight_history = weight_history[
                        weight_history["Date"] <= pd.Timestamp(execution_dt)
                    ]
                ticker_universe = weight_history["Ticker"].unique().tolist()
                backtest_price_history = {
                    ticker: price_history[ticker]
                    for ticker in ticker_universe
                    if ticker in price_history
                }
                if backtest_price_history:
                    try:
                        backtest_results = backtest_portfolio(
                            backtest_price_history,
                            weight_history,
                            trade_log_df=trade_log_df,
                            rebalance_freq=backtest_rebalance_freq,
                            initial_capital=capital,
                            benchmark_df=benchmark_df,
                            start_date=backtest_start_dt,
                            end_date=backtest_end_dt,
                        )
                    except Exception as exc:
                        backtest_results = None
                        if backtest_error is None:
                            backtest_error = str(exc) or exc.__class__.__name__
        if (
            backtest_results is None
            and weights_for_risk is not None
            and not weights_for_risk.empty
        ):
            weights_df = (
                weights_for_risk.rename("Weight")
                .reset_index()
                .rename(columns={"index": "Ticker"})
            )
            backtest_price_history = {
                ticker: price_history[ticker]
                for ticker in weights_df["Ticker"]
                if ticker in price_history
            }
            if backtest_price_history:
                try:
                    backtest_results = backtest_portfolio(
                        backtest_price_history,
                        weights_df,
                        rebalance_freq=backtest_rebalance_freq,
                        initial_capital=capital,
                        benchmark_df=benchmark_df,
                        start_date=backtest_start_dt,
                        end_date=backtest_end_dt,
                    )
                except Exception as exc:
                    backtest_results = None
                    if backtest_error is None:
                        backtest_error = str(exc) or exc.__class__.__name__
        if backtest_results is None and new_allocation is not None and not new_allocation.empty:
            candidate_allocation = new_allocation.copy()
            if "Date" not in candidate_allocation.columns:
                candidate_allocation["Date"] = execution_dt.strftime("%Y-%m-%d")
            candidate_allocation["Date"] = pd.to_datetime(candidate_allocation["Date"], errors="coerce")
            candidate_allocation["Weight"] = pd.to_numeric(candidate_allocation["Weight"], errors="coerce")
            candidate_allocation = candidate_allocation.dropna(subset=["Date", "Ticker", "Weight"])
            if not candidate_allocation.empty:
                candidate_allocation = (
                    candidate_allocation.sort_values(["Date", "Ticker"])
                    .groupby(["Date", "Ticker"], as_index=False)
                    .last()
                )
                ticker_universe = candidate_allocation["Ticker"].unique().tolist()
                backtest_price_history = {
                    ticker: price_history[ticker]
                    for ticker in ticker_universe
                    if ticker in price_history
                }
                if backtest_price_history:
                    try:
                        backtest_results = backtest_portfolio(
                        backtest_price_history,
                        candidate_allocation,
                        trade_log_df=trade_log_df if not trade_log_df.empty else None,
                        rebalance_freq=backtest_rebalance_freq,
                        initial_capital=capital,
                        benchmark_df=benchmark_df,
                        start_date=backtest_start_dt,
                        end_date=backtest_end_dt,
                    )
                    except Exception as exc:
                        backtest_results = None
                        if backtest_error is None:
                            backtest_error = str(exc) or exc.__class__.__name__

    if "Date" not in new_allocation.columns:
        new_allocation = new_allocation.copy()
        new_allocation["Date"] = execution_dt.strftime("%Y-%m-%d")

    return {
        "candidate_pool": candidate_pool_df,
        "sector_rankings": sector_rankings,
        "final_portfolio": final_portfolio_df,
        "missing_tickers": missing_tickers,
        "start_date": start_dt,
        "strategy_start_date": strategy_start_dt,
        "backtest_start_date": backtest_start_dt,
        "analysis_date": analysis_dt,
        "execution_date": execution_dt,
        "end_date": execution_dt,
        "price_history": price_history,
        "sector_price_data": sector_price_data,
        "strategy_name": strategy_spec.name,
        "expected_returns": expected_returns_df,
        "expected_returns_series": expected_returns_series,
        "price_matrix": price_matrix,
        "returns_matrix": returns_matrix,
        "cov_matrix": cov_matrix,
        "optimized_weights": optimized_weights,
        "hybrid_allocation": hybrid_allocation,
        "forecast_exclusions": forecast_exclusions,
        "risk_report": risk_report,
        "backtest_results": backtest_results,
        "backtest_error": backtest_error,
        "trade_signals": trade_signals,
        "benchmark_history": benchmark_df,
        "current_allocation": new_allocation,
        "trade_log": trade_log_df.copy(),
        "as_of_date": execution_dt.strftime("%Y-%m-%d"),
        "backtest_rebalance_freq": backtest_rebalance_freq,
        "strategy_lookback_window": strategy_lookback,
        "weights_as_of": analysis_last_date or cutoff_date,
        "carried_forward_allocation": allocation_carry_forward,
    }
