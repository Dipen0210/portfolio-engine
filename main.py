# main.py

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from execution.portfolio_state import PortfolioState
from hardfilters.data_utils import load_company_universe
from rebalance.rebalance_controller import run_rebalance_cycle

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="smallQ Portfolio Manager", layout="wide")
st.title("📊 smallQ: Integrated Portfolio Manager")

# ---------------------------------------------------------
# Sidebar — User Inputs
# ---------------------------------------------------------
st.sidebar.header("User Input")


def load_company_data():
    return load_company_universe()


company_universe = load_company_data()

if company_universe.empty:
    st.error("Company universe is empty. Please verify the dataset.")

strategy = st.sidebar.selectbox(
    "Select Strategy", ["Momentum", "Mean Reversion", "Value", "Growth", "Quality"]
)

risk_level = st.sidebar.radio("Risk Level", ["Low", "Medium", "High"])
capital = st.sidebar.number_input(
    "Capital ($)", min_value=1000, step=1000, value=10000
)

raw_as_of = st.sidebar.date_input("As of Date", datetime.today())
if isinstance(raw_as_of, datetime):
    as_of_date = raw_as_of
else:
    as_of_date = datetime.combine(raw_as_of, datetime.min.time())
today = datetime.today()
if as_of_date.date() > today.date():
    st.sidebar.warning("Future dates are not supported. Using today's date instead.")
    as_of_date = datetime.combine(today.date(), datetime.min.time())
st.sidebar.caption("Lookback window is determined automatically by the chosen strategy.")

# Sector selection
available_sectors = sorted(
    company_universe["sector"].dropna().unique().tolist()
)

if available_sectors:
    sectors = st.sidebar.multiselect(
        "Select Sectors", available_sectors, default=[]
    )
else:
    st.sidebar.warning("No sectors available in the company universe.")
    sectors = []

rebalance_options = {"Monthly": "M", "Weekly": "W"}
rebalance_choice = st.sidebar.selectbox(
    "Backtest Rebalance Frequency", list(rebalance_options.keys()), index=0
)
rebalance_code = rebalance_options[rebalance_choice]

drift_threshold = st.sidebar.slider(
    "Trade Drift Threshold",
    min_value=0.0,
    max_value=0.10,
    value=0.03,
    step=0.005,
    format="%.3f",
)

def _compute_next_rebalance(base_dt: datetime, freq_code: str) -> datetime:
    base_ts = pd.Timestamp(base_dt)
    if freq_code == "M":
        next_ts = base_ts + pd.DateOffset(months=1)
    else:
        next_ts = base_ts + pd.DateOffset(weeks=1)
    return next_ts.to_pydatetime()


def _adjust_to_trading_day(target_dt: datetime) -> datetime:
    adjusted = target_dt
    while adjusted.weekday() >= 5:
        adjusted -= timedelta(days=1)
    return adjusted


sectors_token = "-".join(sorted(sectors)) if sectors else "all"
adjusted_start_date = _adjust_to_trading_day(as_of_date)
timeline_config = (
    f"{adjusted_start_date.strftime('%Y-%m-%d')}_{rebalance_code}_{strategy}_{risk_level}_{capital}_{drift_threshold:.4f}_{sectors_token}"
)
timeline_state = st.session_state.get("rebalance_timeline")
if timeline_state is None or timeline_state.get("config") != timeline_config:
    st.session_state["rebalance_timeline"] = {
        "config": timeline_config,
        "start_date": adjusted_start_date.strftime("%Y-%m-%d"),
        "last_rebalance": None,
    }
    st.session_state["portfolio_state"] = PortfolioState(cash=capital)
timeline_state = st.session_state["rebalance_timeline"]
start_date_preview = datetime.strptime(timeline_state["start_date"], "%Y-%m-%d")
last_rebalance_preview = timeline_state.get("last_rebalance")
if last_rebalance_preview:
    preview_base = datetime.strptime(last_rebalance_preview, "%Y-%m-%d")
    next_candidate = _compute_next_rebalance(preview_base, rebalance_code)
else:
    preview_base = None
    next_candidate = start_date_preview
next_preview = _adjust_to_trading_day(next_candidate)
if next_preview > datetime.today():
    next_label = next_preview.strftime("%Y-%m-%d") + " (awaiting data)"
else:
    next_label = next_preview.strftime("%Y-%m-%d")
last_label = last_rebalance_preview or "None yet"
st.sidebar.caption(
    f"Last rebalance: {last_label} · Next scheduled: {next_label}"
)

if "portfolio_state" not in st.session_state:
    st.session_state["portfolio_state"] = PortfolioState(cash=capital)

if st.sidebar.button("Reset Portfolio State"):
    st.session_state["portfolio_state"] = PortfolioState(cash=capital)
    st.experimental_rerun()

run_button = st.sidebar.button("🚀 Run Rebalance Cycle")

# ---------------------------------------------------------
# Execute Rebalance
# ---------------------------------------------------------
if run_button:
    st.subheader("Rebalance Results")

    # Initialize state (simulated portfolio memory)
    state = st.session_state["portfolio_state"]
    state.cash = capital

    start_date = datetime.strptime(timeline_state["start_date"], "%Y-%m-%d")
    last_rebalance_str = timeline_state.get("last_rebalance")
    last_rebalance = (
        datetime.strptime(last_rebalance_str, "%Y-%m-%d")
        if last_rebalance_str
        else None
    )
    results = None
    if last_rebalance is None:
        target_candidate = start_date
    else:
        target_candidate = _compute_next_rebalance(last_rebalance, rebalance_code)
    target_date = _adjust_to_trading_day(target_candidate)
    today = datetime.today()
    if target_date > today:
        st.warning(
            "Next rebalance date exceeds available market data. Try again later."
        )
    elif last_rebalance and target_date <= last_rebalance:
        st.warning("No new trading days to process for the selected frequency.")
    else:
        try:
            results = run_rebalance_cycle(
                state,
                company_universe_df=company_universe,
                user_strategy=strategy,
                user_sectors=sectors,
                risk_level=risk_level,
                capital=capital,
                as_of_date=target_date,
                backtest_start_date=start_date,
                backtest_end_date=target_date,
                backtest_rebalance_freq=rebalance_code,
                drift_threshold=drift_threshold,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            timeline_state["last_rebalance"] = target_date.strftime("%Y-%m-%d")
            st.session_state["rebalance_timeline"] = timeline_state
            st.success(
                f"Rebalance executed on {target_date.strftime('%Y-%m-%d')}."
            )

    if results is not None:
        final_df = results["final_portfolio"]
        candidate_pool = results["candidate_pool"]
        missing = results["missing_tickers"]
        expected_returns_df = results.get("expected_returns")
        forecast_exclusions = results.get("forecast_exclusions", [])
        optimized_weights = results.get("optimized_weights")
        hybrid_allocation = results.get("hybrid_allocation")
        risk_report = results.get("risk_report")
        backtest_results = results.get("backtest_results")
        trade_signals = results.get("trade_signals")
        current_allocation = results.get("current_allocation")
        trade_log = results.get("trade_log")
        result_as_of = results.get("as_of_date", target_date)
        strategy_start_label = "-"
        strategy_start_dt = results.get("strategy_start_date")
        if strategy_start_dt is not None:
            strategy_start_label = pd.to_datetime(strategy_start_dt).strftime(
                "%Y-%m-%d"
            )
        backtest_start_label = "-"
        backtest_start_dt = results.get("backtest_start_date")
        if backtest_start_dt is not None:
            backtest_start_label = pd.to_datetime(backtest_start_dt).strftime(
                "%Y-%m-%d"
            )
        backtest_end_label = "-"
        backtest_end_dt = results.get("end_date", target_date)
        if backtest_end_dt is not None:
            backtest_end_label = pd.to_datetime(backtest_end_dt).strftime("%Y-%m-%d")

        st.write("### 📋 Selection Summary")
        summary_cols = st.columns(7)
        summary_cols[0].metric("Strategy", results["strategy_name"])
        summary_cols[1].metric("Risk Level", risk_level)
        summary_cols[2].metric("Capital ($)", f"{capital:,.0f}")
        summary_cols[3].metric("Start", strategy_start_label)
        summary_cols[4].metric("As of", str(result_as_of))
        strategy_lookback = results.get("strategy_lookback_window")
        if strategy_lookback:
            summary_cols[5].metric("Strategy Lookback", f"{int(strategy_lookback)}d")
        else:
            summary_cols[5].metric("Strategy Lookback", "Auto")
        summary_cols[6].metric("Drift Threshold", f"{drift_threshold:.1%}")
        st.caption(
            "Rebalance cadence: "
            + rebalance_choice
            + " · Backtest window: "
            + backtest_start_label
            + " → "
            + backtest_end_label
            + " · Selected sectors: "
            + (", ".join(sectors) if sectors else "All available sectors")
        )

        st.divider()
        st.write("### 💼 Current Allocation Snapshot")
        if current_allocation is None or current_allocation.empty:
            st.info("No allocation available. Optimization stage did not produce weights.")
        else:
            alloc_display = current_allocation.copy()
            if "Weight" in alloc_display.columns:
                alloc_display["Weight"] = alloc_display["Weight"].round(4)
            desired_order = ["Date", "Ticker", "Weight"]
            alloc_display = alloc_display[[col for col in desired_order if col in alloc_display.columns] + [c for c in alloc_display.columns if c not in desired_order]]
            st.dataframe(alloc_display.reset_index(drop=True), hide_index=True)
            st.caption("Allocation as of the selected date. Weights sum to 100% across active holdings.")
            if {"Ticker", "Weight"}.issubset(alloc_display.columns):
                pie_source = (
                    alloc_display.loc[:, ["Ticker", "Weight"]]
                    .assign(Weight=lambda df_: pd.to_numeric(df_["Weight"], errors="coerce"))
                    .dropna(subset=["Weight"])
                )
                pie_source = pie_source[pie_source["Weight"] > 0]
                if not pie_source.empty:
                    fig = px.pie(
                        pie_source,
                        names="Ticker",
                        values="Weight",
                        title="Allocation Breakdown",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.write("### 🧾 Hard Filter Winners (Top 10 per Sector)")
        if candidate_pool.empty:
            st.warning("No stocks passed the hard filters. Consider broadening your criteria.")
        else:
            st.dataframe(candidate_pool.reset_index(drop=True), hide_index=True)
            top_n_display = int(candidate_pool.groupby("sector").size().max())
            st.caption(f"Displayed: Top {top_n_display} per sector based on risk-adjusted market cap filter.")

        st.divider()
        st.write("### 🏁 Strategy Winners (Top 3 per Sector)")
        if final_df.empty:
            st.info("Strategy did not produce any qualifying picks. Try adjusting the filters or selecting a different strategy.")
        else:
            display_cols = ["Ticker", "Sector", "Strategy_Score", "Rank"]
            winners_df = (
                final_df.loc[:, display_cols]
                .sort_values(["Sector", "Rank"])
                .reset_index(drop=True)
            )
            if "Strategy_Score" in winners_df.columns:
                winners_df["Strategy_Score"] = winners_df["Strategy_Score"].astype(float).round(4)
            st.dataframe(winners_df.reset_index(drop=True), hide_index=True)
            st.caption("These top-ranked tickers will flow into the forecasting layer for further analysis.")

        if forecast_exclusions:
            st.warning(
                "The following tickers were removed before forecasting due to insufficient price history: "
                + ", ".join(forecast_exclusions)
            )

        st.divider()
        st.write("### 📈 Forecasted Returns (μ)")
        if expected_returns_df is None or expected_returns_df.empty:
            st.info("Not enough price history to compute expected returns for the selected winners.")
        else:
            st.dataframe(
                expected_returns_df.sort_values("Expected_Return", ascending=False).reset_index(drop=True),
                hide_index=True,
            )
            if strategy_lookback:
                st.caption(
                    f"Expected returns rely on the strategy-defined lookback window (~{int(strategy_lookback)} days) when sufficient price history exists."
                )
            else:
                st.caption("Expected returns rely on the strategy-defined lookback window when sufficient price history exists.")

        st.divider()
        st.write("### ⚖️ Optimization & Allocation")
        if optimized_weights is None or optimized_weights.empty:
            st.info("Optimization was skipped (not enough data or optimization failed).")
        else:
            opt_df = optimized_weights.rename("Opt_Weight").reset_index().rename(columns={"index": "Ticker"})
            opt_df["Opt_Weight"] = opt_df["Opt_Weight"].round(4)
            st.markdown("**Mean-Variance Optimized Weights**")
            st.dataframe(opt_df.reset_index(drop=True), hide_index=True)

        if hybrid_allocation is not None and not hybrid_allocation.empty:
            hybrid_display = hybrid_allocation.copy()
            hybrid_display["Final_Weight"] = hybrid_display["Final_Weight"].round(4)
            hybrid_display["Rank_Weight"] = hybrid_display["Rank_Weight"].round(4)
            hybrid_display["Opt_Weight"] = hybrid_display["Opt_Weight"].round(4)
            st.markdown("**Blended Allocation (Rank vs. Optimizer)**")
            st.dataframe(hybrid_display.reset_index(drop=True), hide_index=True)
            st.caption("Final weights blend rank-based intuition with mean-variance optimization.")

        st.divider()
        st.write("### 🛡️ Risk Snapshot")
        if not risk_report:
            st.info("Risk metrics unavailable (missing weights or price history).")
        else:
            risk_cols = st.columns(3)
            if not pd.isna(risk_report.get("annualized_vol", float("nan"))):
                risk_cols[0].metric("Annualized Volatility", f"{risk_report['annualized_vol']:.2%}")
            else:
                risk_cols[0].metric("Annualized Volatility", "N/A")

            if not pd.isna(risk_report.get("annualized_mean", float("nan"))):
                risk_cols[1].metric("Annualized Return", f"{risk_report['annualized_mean']:.2%}")
            else:
                risk_cols[1].metric("Annualized Return", "N/A")

            if not pd.isna(risk_report.get("sharpe", float("nan"))):
                risk_cols[2].metric("Sharpe Ratio", f"{risk_report['sharpe']:.2f}")
            else:
                risk_cols[2].metric("Sharpe Ratio", "N/A")

            var_cols = st.columns(2)
            var_cols[0].metric("Parametric VaR (95%)", f"{risk_report['parametric_VaR']:.2%}")
            var_cols[1].metric("Historical VaR (95%)", f"{risk_report['historical_VaR']:.2%}")
            st.caption("Risk metrics are derived from the blended allocation weights. Tail metrics shown in daily terms.")

        if missing:
            st.caption(
                f"OHLCV data unavailable for {len(missing)} tickers: "
                + ", ".join(missing)
            )

        st.divider()
        backtest_curve = backtest_results.get("portfolio") if backtest_results else None
        backtest_metrics = backtest_results.get("metrics", {}) if backtest_results else {}
        benchmark_curve = backtest_results.get("benchmark") if backtest_results else None
        benchmark_metrics = backtest_results.get("benchmark_metrics", {}) if backtest_results else {}
        backtest_summary = backtest_results.get("summary", {}) if backtest_results else {}
        rebalance_count = backtest_results.get("rebalance_count", 0) if backtest_results else 0
        rebalance_dates = backtest_results.get("rebalance_dates", []) if backtest_results else []
        transactions_df = backtest_results.get("transactions") if backtest_results else pd.DataFrame()
        holdings_df = backtest_results.get("holdings") if backtest_results else pd.DataFrame()
        backtest_available = backtest_results is not None

        def _fmt_dollar(value):
            return "N/A" if value is None or pd.isna(value) else f"${value:,.2f}"

        def _fmt_pct(value):
            return "N/A" if value is None or pd.isna(value) else f"{value * 100:.2f}%"

        invested_amt = backtest_summary.get("initial_capital")
        final_amt = backtest_summary.get("final_value")
        ret_amt = backtest_summary.get("return_amount")
        ret_pct = backtest_summary.get("return_pct")

        summary_start = backtest_summary.get("start_date")
        summary_end = backtest_summary.get("end_date")
        summary_start_label = pd.to_datetime(summary_start).strftime("%Y-%m-%d") if summary_start is not None else "-"
        summary_end_label = pd.to_datetime(summary_end).strftime("%Y-%m-%d") if summary_end is not None else "-"

        st.write("### 📬 Trade Signals")
        if trade_signals is None or trade_signals.empty:
            st.info("No trade signals generated. Portfolio unchanged.")
        else:
            signals_display = trade_signals.copy()
            if "Old_Weight" in signals_display.columns:
                signals_display["Old_Weight"] = (
                    pd.to_numeric(signals_display["Old_Weight"], errors="coerce").round(4)
                )
            if "New_Weight" in signals_display.columns:
                signals_display["New_Weight"] = (
                    pd.to_numeric(signals_display["New_Weight"], errors="coerce").round(4)
                )
                st.dataframe(signals_display, hide_index=True)
            st.caption(
                f"{len(signals_display)} trade instructions generated using a {drift_threshold:.1%} drift threshold."
            )

        st.divider()
        st.write("### 🧾 Trade Log History")
        if trade_log is None or trade_log.empty:
            st.info("Trade log is empty. Run at least one rebalance cycle.")
        else:
            log_display = trade_log.copy()
            if "Old_Weight" in log_display.columns:
                log_display["Old_Weight"] = (
                    pd.to_numeric(log_display["Old_Weight"], errors="coerce").round(4)
                )
            if "New_Weight" in log_display.columns:
                log_display["New_Weight"] = (
                    pd.to_numeric(log_display["New_Weight"], errors="coerce").round(4)
                )
            desired_log_order = ["Date", "Ticker", "Signal", "Old_Weight", "New_Weight", "Reason", "Timestamp"]
            log_display = log_display[[col for col in desired_log_order if col in log_display.columns]]
            st.dataframe(log_display.tail(100).reset_index(drop=True), hide_index=True)
            st.caption("Most recent 100 trade entries from the session log.")

        st.divider()
        st.write("### 🔁 Backtest Transactions")
        if transactions_df is None or transactions_df.empty:
            st.info("No backtest transactions available.")
        else:
            display_tx = transactions_df.copy()
            numeric_cols = [
                "Net_Weight",
                "Shares",
                "Price",
                "TradeValue",
                "CashFlow",
                "Transaction Cost",
            ]
            for col in numeric_cols:
                if col in display_tx.columns:
                    display_tx[col] = (
                        pd.to_numeric(display_tx[col], errors="coerce").round(4)
                    )
            display_tx = display_tx.rename(columns={"Net_Weight": "Net weight"})
            display_tx["Date"] = pd.to_datetime(display_tx["Date"]).dt.strftime("%Y-%m-%d")
            column_order = [
                "Date",
                "Ticker",
                "Signal",
                "Action",
                "Net weight",
                "Reason",
                "Shares",
                "Price",
                "TradeValue",
                "Transaction Cost",
                "CashFlow",
            ]
            display_tx = display_tx[
                [c for c in column_order if c in display_tx.columns]
            ]
            st.dataframe(display_tx, hide_index=True)
            st.caption(
                f"{len(display_tx)} trades recorded across the backtest window. Positive CashFlow reflects capital returned to cash."
            )

        st.divider()
        st.write("### 📂 Open Backtest Holdings")
        holdings_columns = ["Ticker", "Shares", "Price", "MarketValue"]
        if holdings_df is None or holdings_df.empty:
            st.info("No open holdings at the end of the backtest window.")
        else:
            holdings_display = holdings_df.copy()
            for col in ["Shares", "Price", "MarketValue"]:
                if col in holdings_display.columns:
                    holdings_display[col] = (
                        pd.to_numeric(holdings_display[col], errors="coerce")
                    ).round(6 if col != "MarketValue" else 4)
            holdings_display = holdings_display.reindex(columns=holdings_columns)
            st.dataframe(holdings_display, hide_index=True)
            holdings_label = summary_end_label if summary_end_label != "-" else "final backtest date"
            st.caption(f"Mark-to-market holdings as of {holdings_label}.")

        st.divider()
        st.write(f"### 🔄 Backtest ({rebalance_choice} Rebalance)")
        if not backtest_available:
            st.info(
                "Backtest unavailable. Ensure sufficient price history and weights, then rerun."
            )
        else:
            backtest_summary_cols = st.columns(5)
            backtest_summary_cols[0].metric("Invested", _fmt_dollar(invested_amt))
            backtest_summary_cols[1].metric("Final Value", _fmt_dollar(final_amt))
            backtest_summary_cols[2].metric("P/L", _fmt_dollar(ret_amt))
            backtest_summary_cols[3].metric("Return %", _fmt_pct(ret_pct))
            backtest_summary_cols[4].metric("Rebalances", str(int(rebalance_count)))

            if rebalance_count == 0:
                st.caption(
                    "No scheduled rebalances executed; holdings carried throughout the window."
                )
            elif rebalance_dates:
                formatted_rebalance_dates = [
                    pd.to_datetime(dt).strftime("%Y-%m-%d") for dt in rebalance_dates
                ]
                st.caption(
                    "Rebalances executed on: " + ", ".join(formatted_rebalance_dates)
                )
            calendar_notes = [
                note for note in backtest_summary.get("calendar_notes", []) if note
            ]
            if calendar_notes:
                st.caption("Calendar adjustments: " + " | ".join(calendar_notes))

            if backtest_curve is not None and not backtest_curve.empty:
                chart_df = backtest_curve.copy()
                if not isinstance(chart_df.index, pd.DatetimeIndex):
                    chart_df = chart_df.copy()
                    chart_df.index = pd.to_datetime(chart_df.index)
                value_df = chart_df[["Portfolio Value"]]
                if benchmark_curve is not None and not benchmark_curve.empty:
                    bench_chart = benchmark_curve.copy()
                    if not isinstance(bench_chart.index, pd.DatetimeIndex):
                        bench_chart.index = pd.to_datetime(bench_chart.index)
                    value_df = value_df.join(
                        bench_chart["Benchmark Value"], how="inner"
                    )
                    value_df = value_df.rename(columns={"Benchmark Value": "S&P 500"})
                st.line_chart(value_df.rename(columns={"Portfolio Value": "Portfolio"}))
                st.caption(
                    f"Portfolio value trajectory ({rebalance_choice.lower()} rebalance) versus S&P 500 benchmark."
                )

            if backtest_metrics:
                rows = [
                    "Annualized Return",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Max Drawdown",
                    "CAGR",
                ]
                metrics_data = {
                    "Metric": rows,
                    "Portfolio": [
                        f"{backtest_metrics.get('Annualized Return', float('nan')):.2%}",
                        f"{backtest_metrics.get('Annualized Volatility', float('nan')):.2%}",
                        f"{backtest_metrics.get('Sharpe Ratio', float('nan')):.2f}",
                        f"{backtest_metrics.get('Sortino Ratio', float('nan')):.2f}",
                        f"{backtest_metrics.get('Max Drawdown', float('nan')):.2%}",
                        f"{backtest_metrics.get('CAGR', float('nan')):.2%}",
                    ],
                }
                if benchmark_metrics:
                    metrics_data["S&P 500"] = [
                        f"{benchmark_metrics.get('Annualized Return', float('nan')):.2%}",
                        f"{benchmark_metrics.get('Annualized Volatility', float('nan')):.2%}",
                        f"{benchmark_metrics.get('Sharpe Ratio', float('nan')):.2f}",
                        f"{benchmark_metrics.get('Sortino Ratio', float('nan')):.2f}",
                        f"{benchmark_metrics.get('Max Drawdown', float('nan')):.2%}",
                        f"{benchmark_metrics.get('CAGR', float('nan')):.2%}",
                    ]
                metrics_df = pd.DataFrame(metrics_data)
                st.table(metrics_df)

    else:
        st.info(
            "👈 Configure inputs in the sidebar and click **Run Rebalance Cycle** to start."
        )
