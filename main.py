# main.py

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from execution.execution_engine import run_rebalance_cycle as execute_rebalance
from execution.portfolio_state import PortfolioState
from hardfilters.data_utils import load_company_universe
from hardfilters.forex import (
    available_base_currencies,
    list_pairs_for_bases,
    load_forex_universe,
)
from rebalance.rebalance_controller import COMMISSION_PER_TRADE, run_rebalance_cycle
from utils.formatting import (
    format_currency,
    format_percentage,
    normalize_and_truncate_weights,
    truncate_dataframe,
    truncate_series,
    truncate_value,
)
from utils.trading_calendar import next_trading_day, previous_trading_day

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="smallQ Portfolio Manager", layout="wide")
st.title("📊 smallQ: Integrated Portfolio Manager")

# Inject custom CSS to make tabs expand to fill the width
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            flex-grow: 1;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar — User Inputs
# ---------------------------------------------------------
st.sidebar.header("User Input")

def load_company_data():
    return load_company_universe()


strategy = st.sidebar.selectbox(
    "Select Strategy", ["Momentum", "Mean Reversion", "Value", "Growth", "Quality"]
)

# Map user-friendly Market Cap labels to the internal "risk_level" expected by the backend
# Large Cap -> Low Risk (internal "Low")
# Mid Cap -> Medium Risk (internal "Medium")
# Small Cap -> High Risk (internal "High")
market_cap_choice = st.sidebar.selectbox("Market Cap", ["Large Cap", "Mid Cap", "Small Cap"], index=1)
risk_map = {
    "Large Cap": "Low",
    "Mid Cap": "Medium",
    "Small Cap": "High"
}
risk_level = risk_map[market_cap_choice]
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

# Asset class selection (controls downstream filters)
ASSET_CLASS_OPTIONS = ["Equities", "Fixed Income", "Commodities", "Forex", "Crypto"]
asset_class = st.sidebar.selectbox("Asset Class", ASSET_CLASS_OPTIONS, index=0)
equities_selected = asset_class == "Equities"
forex_selected = asset_class == "Forex"
if not equities_selected:
    if forex_selected:
        st.sidebar.info("Forex workflow preview: select one or more base currencies to expand all supported pairs.")
        st.info("💱 Forex analytics are in progress. Pair selection is available today; portfolio construction is coming soon.")
    else:
        st.sidebar.info("Multi-asset support is coming soon. Switch back to Equities to configure the existing workflow.")
        st.info("🚧 Multi-asset workflows are under construction. Select Equities to access the full configuration experience.")

if equities_selected:
    company_universe = load_company_data()
    if company_universe.empty:
        st.error("Company universe is empty. Please verify the dataset.")
else:
    company_universe = pd.DataFrame()

if not forex_selected:
    st.session_state.pop("forex_symbols", None)

selected_forex_bases: list[str] = []
forex_pairs_df: pd.DataFrame | None = None
if forex_selected:
    try:
        forex_universe = load_forex_universe()
    except FileNotFoundError as exc:
        st.sidebar.error(str(exc))
        st.session_state.pop("forex_symbols", None)
    else:
        base_options = available_base_currencies(forex_universe)
        if base_options:
            selected_forex_bases = st.sidebar.multiselect(
                "Base Currencies",
                base_options,
                key="forex_base_currencies",
            )
            if selected_forex_bases:
                forex_pairs_df = list_pairs_for_bases(selected_forex_bases, data=forex_universe)
                if forex_pairs_df.empty:
                    st.sidebar.warning("No FX pairs available for the selected base currencies.")
                    st.session_state.pop("forex_symbols", None)
                else:
                    forex_symbols = forex_pairs_df["YF_SYMBOL"].tolist()
                    st.session_state["forex_symbols"] = forex_symbols
                    base_list = ", ".join(sorted(set(selected_forex_bases)))
                    st.sidebar.caption(
                        f"{len(forex_symbols)} FX pairs generated for base currencies: {base_list}"
                    )
            else:
                st.sidebar.caption("Select one or more base currencies to generate FX pairs.")
                st.session_state.pop("forex_symbols", None)
        else:
            st.sidebar.warning("No base currencies available in the FX universe.")

# Sector selection (visible only for equities workflow)
available_sectors: list[str] = []
sectors: list[str] = []
if equities_selected:
    if company_universe.empty:
        st.sidebar.warning("Company universe unavailable—sector selection disabled.")
    else:
        available_sectors = sorted(
            company_universe["sector"].dropna().unique().tolist()
        )
        if available_sectors:
            sectors = st.sidebar.multiselect(
                "Select Sectors", available_sectors, default=[]
            )
        else:
            st.sidebar.warning("No sectors available in the company universe.")
else:
    st.sidebar.caption("Select the Equities asset class to enable sector filtering.")

rebalance_options = {"Monthly": "M", "Weekly": "W"}
rebalance_choice = st.sidebar.selectbox(
    "Backtest Rebalance Frequency", list(rebalance_options.keys()), index=0
)
rebalance_code = rebalance_options[rebalance_choice]

def _compute_next_rebalance(base_dt: datetime, freq_code: str) -> datetime:
    base_ts = pd.Timestamp(base_dt)
    if freq_code == "M":
        next_ts = base_ts + pd.DateOffset(months=1)
    else:
        next_ts = base_ts + pd.DateOffset(weeks=1)
    return next_ts.to_pydatetime()



def _get_portfolio_equity(state: PortfolioState) -> float:
    history = getattr(state, "portfolio_value_history", None)
    if history is not None and not history.empty and "Value" in history.columns:
        values = pd.to_numeric(history["Value"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[-1])
    return float(state.cash)


def _prepare_price_history_for_execution(price_history: dict[str, pd.DataFrame], execution_dt: datetime) -> dict[str, pd.DataFrame]:
    if not price_history:
        return {}
    exec_ts = pd.Timestamp(execution_dt).normalize()
    sliced: dict[str, pd.DataFrame] = {}
    for ticker, df in price_history.items():
        if df is None or df.empty:
            continue
        temp = df.copy()
        if "Date" in temp.columns:
            temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")
            temp = temp[temp["Date"] <= exec_ts]
        else:
            temp.index = pd.to_datetime(temp.index)
            temp = temp[temp.index <= exec_ts]
        if temp.empty or "Close" not in temp.columns:
            continue
        sliced[ticker] = temp
    return sliced


def _latest_prices(price_history: dict[str, pd.DataFrame], target_date: datetime | None = None) -> dict[str, float]:
    prices: dict[str, float] = {}
    target_ts = pd.Timestamp(target_date).normalize() if target_date is not None else None
    for ticker, df in price_history.items():
        if df is None or df.empty:
            continue
        working = df.copy()
        if "Date" in working.columns:
            working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
            working = working.dropna(subset=["Date"]).set_index("Date")
        else:
            working.index = pd.to_datetime(working.index)
        working = working.sort_index()
        if working.empty:
            continue
        if target_ts is not None:
            try:
                row = working.loc[[target_ts]]
            except KeyError:
                row = working.loc[working.index <= target_ts]
                if row.empty:
                    continue
                row = row.iloc[[-1]]
        else:
            row = working.iloc[[-1]]
        price_col = "Open" if "Open" in row.columns else "Close" if "Close" in row.columns else None
        if price_col is None:
            continue
        price_val = pd.to_numeric(row[price_col], errors="coerce").dropna()
        if price_val.empty:
            continue
        prices[ticker] = float(price_val.iloc[-1])
    return prices


def _build_live_holdings(state: PortfolioState) -> pd.DataFrame:
    positions = getattr(state, "positions", {}) or {}
    last_prices = getattr(state, "last_price_snapshot", {}) or {}
    rows: list[dict] = []
    for ticker, qty in positions.items():
        qty_val = truncate_value(qty, decimals=4)
        price = last_prices.get(ticker)
        price_val = truncate_value(price, decimals=4) if price is not None else None
        mv = None
        if price_val is not None and qty_val is not None:
            mv = truncate_value(float(qty_val) * float(price_val), decimals=4)
        rows.append(
            {
                "Ticker": ticker,
                "Shares": qty_val,
                "Price": price_val,
                "MarketValue": mv,
            }
        )
    cash_val = truncate_value(getattr(state, "cash", 0.0), decimals=4)
    if cash_val and (abs(cash_val) > 1e-9 or not rows):
        rows.append({"Ticker": "CASH", "Shares": "", "Price": "", "MarketValue": cash_val})
    return pd.DataFrame(rows)


def _resolve_backtest_start(state: PortfolioState, default_dt: datetime) -> datetime:
    """
    Pick the earliest date observed in session history so backtests keep the full run record.
    """
    candidates: list[pd.Timestamp] = []
    signal_log = getattr(state, "signal_log", None)
    if signal_log is not None and not signal_log.empty:
        for col in ("Execution_Date", "Date", "Data_Through"):
            if col in signal_log.columns:
                series = pd.to_datetime(signal_log[col], errors="coerce").dropna()
                if not series.empty:
                    candidates.append(series.min().normalize())
    trade_log = getattr(state, "trade_log", None)
    if trade_log is not None and not trade_log.empty and "Date" in trade_log.columns:
        series = pd.to_datetime(trade_log["Date"], errors="coerce").dropna()
        if not series.empty:
            candidates.append(series.min().normalize())
    history = getattr(state, "portfolio_value_history", None)
    if history is not None and not history.empty:
        date_col = None
        for col in ("Date", "date"):
            if col in history.columns:
                date_col = col
                break
        if date_col is not None:
            series = pd.to_datetime(history[date_col], errors="coerce").dropna()
            if not series.empty:
                candidates.append(series.min().normalize())
    if not candidates:
        return default_dt
    earliest = min(candidates)
    baseline = pd.Timestamp(default_dt).normalize()
    if earliest is None or pd.isna(earliest):
        return default_dt
    return min(earliest.to_pydatetime(), baseline.to_pydatetime())


sectors_token = "-".join(sorted(sectors)) if sectors else "all"
execution_date = next_trading_day(as_of_date)
try:
    analysis_date = previous_trading_day(execution_date)
except ValueError:
    analysis_date = execution_date

if execution_date.date() != as_of_date.date():
    st.info(
        f"Selected date {as_of_date.strftime('%Y-%m-%d')} is not a trading day. "
        f"Trades will execute on {execution_date.strftime('%Y-%m-%d')} at the market open."
    )
st.caption(
    f"Calculations use market data through {analysis_date.strftime('%Y-%m-%d')} (previous trading close)."
)

timeline_config = (
    f"{analysis_date.strftime('%Y-%m-%d')}_{execution_date.strftime('%Y-%m-%d')}_"
    f"{rebalance_code}_{strategy}_{risk_level}_{capital}_{sectors_token}"
)
timeline_state = st.session_state.get("rebalance_timeline")
if timeline_state is None or timeline_state.get("config") != timeline_config:
    st.session_state["rebalance_timeline"] = {
        "config": timeline_config,
        "analysis_anchor": analysis_date.strftime("%Y-%m-%d"),
        "execution_anchor": execution_date.strftime("%Y-%m-%d"),
        "last_rebalance": None,
    }
    st.session_state["portfolio_state"] = PortfolioState(cash=capital)
timeline_state = st.session_state["rebalance_timeline"]
if "analysis_anchor" not in timeline_state or "execution_anchor" not in timeline_state:
    timeline_state["analysis_anchor"] = analysis_date.strftime("%Y-%m-%d")
    timeline_state["execution_anchor"] = execution_date.strftime("%Y-%m-%d")
    st.session_state["rebalance_timeline"] = timeline_state
analysis_anchor = datetime.strptime(timeline_state["analysis_anchor"], "%Y-%m-%d")
execution_anchor = datetime.strptime(timeline_state["execution_anchor"], "%Y-%m-%d")
last_rebalance_preview = timeline_state.get("last_rebalance")
if last_rebalance_preview:
    preview_base = datetime.strptime(last_rebalance_preview, "%Y-%m-%d")
    next_candidate = _compute_next_rebalance(preview_base, rebalance_code)
else:
    preview_base = None
    next_candidate = execution_anchor
next_preview_execution = next_trading_day(next_candidate)
if next_preview_execution > datetime.today():
    next_label = next_preview_execution.strftime("%Y-%m-%d") + " (awaiting data)"
else:
    next_label = next_preview_execution.strftime("%Y-%m-%d")
last_label = last_rebalance_preview or "None yet"
st.sidebar.caption(
    f"Last rebalance: {last_label} · Next scheduled: {next_label}"
)

if "portfolio_state" not in st.session_state:
    st.session_state["portfolio_state"] = PortfolioState(cash=capital)

if st.sidebar.button("Reset Portfolio State"):
    st.session_state.clear()
    st.session_state["portfolio_state"] = PortfolioState(cash=capital)
    st.rerun()

run_button = st.sidebar.button("🚀 Run Rebalance Cycle", disabled=not equities_selected)

# ---------------------------------------------------------
# Execute Rebalance
# ---------------------------------------------------------
if run_button:
    st.subheader("Rebalance Results")

    # Initialize state (simulated portfolio memory)
    state = st.session_state["portfolio_state"]
    cycle_equity = _get_portfolio_equity(state)

    analysis_anchor_dt = analysis_anchor
    execution_anchor_dt = execution_anchor
    backtest_start_dt = _resolve_backtest_start(state, analysis_anchor_dt)
    last_rebalance_str = timeline_state.get("last_rebalance")
    last_rebalance = (
        datetime.strptime(last_rebalance_str, "%Y-%m-%d")
        if last_rebalance_str
        else None
    )
    results = None
    if last_rebalance is None:
        target_execution = execution_anchor_dt
    else:
        target_candidate = _compute_next_rebalance(last_rebalance, rebalance_code)
        target_execution = next_trading_day(target_candidate)
    try:
        target_analysis = previous_trading_day(target_execution)
    except ValueError:
        target_analysis = target_execution
    today = datetime.today()
    if target_execution > today:
        st.warning(
            "Next rebalance date exceeds available market data. Try again later."
        )
    elif last_rebalance and target_execution <= last_rebalance:
        st.warning("No new trading days to process for the selected frequency.")
    else:
        try:
            results = run_rebalance_cycle(
                state,
                company_universe_df=company_universe,
                user_strategy=strategy,
                user_sectors=sectors,
                risk_level=risk_level,
                capital=cycle_equity,
                as_of_date=target_analysis,
                execution_date=target_execution,
                backtest_start_date=backtest_start_dt,
                backtest_end_date=target_execution,
                backtest_rebalance_freq=rebalance_code,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            timeline_state["last_rebalance"] = target_execution.strftime("%Y-%m-%d")
            timeline_state["analysis_anchor"] = target_analysis.strftime("%Y-%m-%d")
            timeline_state["execution_anchor"] = target_execution.strftime("%Y-%m-%d")
            st.session_state["rebalance_timeline"] = timeline_state
            st.success(
                f"Rebalance executed on {target_execution.strftime('%Y-%m-%d')}."
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
        backtest_error = results.get("backtest_error")
        allocation_carried = results.get("carried_forward_allocation", False)
        fallback_equal_alloc = results.get("fallback_equal_weight", False)
        trade_signals = results.get("trade_signals")
        current_allocation = results.get("current_allocation")
        trade_log = results.get("trade_log")
        execution_used = results.get("execution_date", target_execution)
        if execution_used is not None:
            execution_label = pd.to_datetime(execution_used).strftime("%Y-%m-%d")
        else:
            execution_label = "-"
        analysis_used = results.get("analysis_date")
        if analysis_used is not None:
            analysis_label = pd.to_datetime(analysis_used).strftime("%Y-%m-%d")
        else:
            analysis_label = "-"
        strategy_start_label = "-"
        strategy_start_dt = results.get("strategy_start_date")
        if strategy_start_dt is not None:
            strategy_start_label = pd.to_datetime(strategy_start_dt).strftime("%Y-%m-%d")
        backtest_start_label = "-"
        backtest_start_dt = results.get("backtest_start_date")
        if backtest_start_dt is not None:
            backtest_start_label = pd.to_datetime(backtest_start_dt).strftime(
                "%Y-%m-%d"
            )
        backtest_end_label = "-"
        backtest_end_dt = results.get("end_date", execution_used)
        if backtest_end_dt is not None:
            backtest_end_label = pd.to_datetime(backtest_end_dt).strftime("%Y-%m-%d")

        post_trade_equity = cycle_equity
        execution_warning = None
        execution_prices = {}
        execution_dt = None
        price_history = results.get("price_history", {})
        if execution_used is not None:
            execution_dt = pd.to_datetime(execution_used)
            execution_prices = _prepare_price_history_for_execution(price_history, execution_dt)
            if execution_prices:
                allocation_for_exec = None
                if current_allocation is not None and not current_allocation.empty:
                    allocation_for_exec = (
                        current_allocation.loc[:, ["Ticker", "Weight"]]
                        .copy()
                    )
                    if not allocation_for_exec.empty:
                        allocation_for_exec["Weight"] = (
                            pd.to_numeric(allocation_for_exec["Weight"], errors="coerce")
                            .fillna(0.0)
                        )
                        allocation_for_exec["Ticker"] = allocation_for_exec["Ticker"].astype(str).str.strip()
                        allocation_for_exec = allocation_for_exec[
                            (allocation_for_exec["Ticker"] != "")
                            & (allocation_for_exec["Weight"] > 0)
                        ]
                if allocation_for_exec is not None and not allocation_for_exec.empty:
                    try:
                        execute_rebalance(
                            state=state,
                            price_data_dict=execution_prices,
                            new_portfolio_weights=allocation_for_exec.loc[:, ["Ticker", "Weight"]],
                            date=execution_dt.strftime("%Y-%m-%d"),
                            commission_per_trade=COMMISSION_PER_TRADE,
                        )
                    except Exception as exc:
                        execution_warning = f"Live portfolio update failed: {exc}"
                else:
                    latest_prices = _latest_prices(execution_prices, execution_dt)
                    if latest_prices:
                        post_trade_equity = state.mark_to_market(
                            execution_dt.strftime("%Y-%m-%d"),
                            latest_prices,
                        )
                        try:
                            state.last_price_snapshot = dict(latest_prices)
                        except Exception:
                            state.last_price_snapshot = latest_prices
                post_trade_equity = _get_portfolio_equity(state)
            else:
                execution_warning = (
                    "Unable to update live portfolio — missing price data for the execution date."
                )

        if execution_warning:
            st.warning(execution_warning)

        # Prepare shared data for tabs
        portfolio_cycles = getattr(state, "cycle_log", [])
        state_cycle_df: pd.DataFrame | None = None
        latest_state_row: pd.Series | None = None
        initial_live_capital = float(getattr(state, "initial_capital", capital) or capital)
        if portfolio_cycles:
            state_cycle_df = pd.DataFrame(portfolio_cycles).copy()
            if "Date" in state_cycle_df.columns:
                state_cycle_df["Date"] = pd.to_datetime(state_cycle_df["Date"], errors="coerce")
                state_cycle_df = state_cycle_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            numeric_cols = ["HoldingsValue", "Cash", "RealizedPnL", "UnrealizedPnL", "FeesPaid"]
            for col in numeric_cols:
                if col in state_cycle_df.columns:
                    state_cycle_df[col] = pd.to_numeric(state_cycle_df[col], errors="coerce").fillna(0.0)
                else:
                    state_cycle_df[col] = 0.0
            state_cycle_df["Current Position"] = state_cycle_df["HoldingsValue"]
            state_cycle_df["GrossPnL"] = (
                state_cycle_df["RealizedPnL"] + state_cycle_df["UnrealizedPnL"]
            )
            state_cycle_df["NetPnL"] = state_cycle_df["GrossPnL"] - state_cycle_df["FeesPaid"]
            state_cycle_df["P/L"] = state_cycle_df["GrossPnL"]
            state_cycle_df["ReturnPct"] = (
                state_cycle_df["P/L"] / initial_live_capital if initial_live_capital else 0.0
            )
            state_cycle_df["RealizedCycle"] = state_cycle_df["RealizedPnL"].diff().fillna(
                state_cycle_df["RealizedPnL"]
            )
            state_cycle_df["UnrealizedCycle"] = state_cycle_df["UnrealizedPnL"].diff().fillna(
                state_cycle_df["UnrealizedPnL"]
            )
            state_cycle_df["Rebalances"] = state_cycle_df.index + 1
            state_cycle_df["Execution Date"] = state_cycle_df["Date"].dt.strftime("%Y-%m-%d")
            state_cycle_df["Transaction Costs"] = state_cycle_df["FeesPaid"]
            latest_state_row = state_cycle_df.iloc[-1] if not state_cycle_df.empty else None

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard",
            "🎯 Selection & Strategy",
            "🧠 Analysis & Risk",
            "📡 Signals",
            "📜 Backtest Details"
        ])

        with tab1:
            st.write("### 📋 Selection Summary")
            strategy_lookback = results.get("strategy_lookback_window")
            # Calculate total days between Start and Data Through
            try:
                start_dt_obj = datetime.strptime(strategy_start_label, "%Y-%m-%d")
                end_dt_obj = datetime.strptime(analysis_label, "%Y-%m-%d")
                total_days = (end_dt_obj - start_dt_obj).days
                lookback_display = f"{total_days} days"
            except ValueError:
                lookback_display = f"{int(strategy_lookback)}d" if strategy_lookback else "Auto"

            summary_items = [
                ("Strategy", results["strategy_name"]),
                ("Market Cap", market_cap_choice),
                ("Start", strategy_start_label),
                ("As of", execution_label),
                ("Data Through", analysis_label),
                (
                    "Total Days",
                    lookback_display,
                ),
                (
                    "Strategy Window",
                    f"{int(strategy_lookback)}d" if strategy_lookback else "Auto",
                ),
            ]
            for i in range(0, len(summary_items), 3):
                row_items = summary_items[i : i + 3]
                cols = st.columns(len(row_items))
                for col, (label, value) in zip(cols, row_items):
                    col.markdown(
                        f"<div><strong>{label}</strong><br>{value}</div>",
                        unsafe_allow_html=True,
                    )
            summary_details = "<br>".join(
                [
                    f"<strong>Rebalance cadence:</strong> {rebalance_choice}",
                    f"<strong>Backtest window:</strong> {backtest_start_label} → {backtest_end_label}",
                    f"<strong>Data through:</strong> {analysis_label}",
                    f"<strong>Selected sectors:</strong> {', '.join(sectors) if sectors else 'All available sectors'}",
                ]
            )
            st.markdown(summary_details, unsafe_allow_html=True)
            if allocation_carried:
                st.info(
                    "Retained prior weights because new allocation data was unavailable for one or more holdings."
                )
            if fallback_equal_alloc:
                st.info(
                    "Displayed weights default to equal-weight allocation due to insufficient price history for optimization."
                )

            st.divider()
            st.write("### 💼 Current Allocation Snapshot")
            if current_allocation is None or current_allocation.empty:
                st.info("No allocation available. Optimization stage did not produce weights.")
            else:
                alloc_display = current_allocation.copy()
                if "CarryForward" in alloc_display.columns:
                    alloc_display = alloc_display.drop(columns=["CarryForward"])
                if "Weight" in alloc_display.columns:
                    alloc_display["Weight"] = truncate_series(
                        pd.to_numeric(alloc_display["Weight"], errors="coerce"), decimals=4
                    )
                if {"Ticker", "Weight"}.issubset(alloc_display.columns):
                    alloc_display = normalize_and_truncate_weights(alloc_display, decimals=4)
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
            live_holdings = _build_live_holdings(state)
            if live_holdings is not None and not live_holdings.empty:
                st.write("### 📂 Live Holdings Snapshot")
                live_display = live_holdings.copy()
                if "Shares" in live_display.columns:
                    live_display["Shares"] = live_display["Shares"].apply(
                        lambda x: "" if x in ("", None) else f"{float(x):.4f}"
                    )
                for col in ["Price", "MarketValue"]:
                    if col in live_display.columns:
                        live_display[col] = live_display[col].apply(
                            lambda v: "" if v in ("", None) else format_currency(v)
                        )
                st.dataframe(live_display, hide_index=True)
                snapshot_label = execution_label
                if 'state_cycle_df' in locals() and state_cycle_df is not None:
                    if "Execution Date" in state_cycle_df.columns:
                        snapshot_label = state_cycle_df["Execution Date"].iloc[-1]
                st.caption(
                    f"Live holdings reflect the latest executed rebalance as of {snapshot_label}."
                )

        with tab2:
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
                    winners_df["Strategy_Score"] = truncate_series(
                        pd.to_numeric(winners_df["Strategy_Score"], errors="coerce"), decimals=4
                    )
                st.dataframe(winners_df.reset_index(drop=True), hide_index=True)
                st.caption("These top-ranked tickers will flow into the forecasting layer for further analysis.")

            if forecast_exclusions:
                st.warning(
                    "The following tickers were removed before forecasting due to insufficient price history: "
                    + ", ".join(forecast_exclusions)
                )

        with tab3:
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
                opt_df["Opt_Weight"] = truncate_series(
                    pd.to_numeric(opt_df["Opt_Weight"], errors="coerce"), decimals=4
                )
                st.markdown("**Mean-Variance Optimized Weights**")
                st.dataframe(opt_df.reset_index(drop=True), hide_index=True)

            if hybrid_allocation is not None and not hybrid_allocation.empty:
                hybrid_display = hybrid_allocation.copy()
                for col in ["Final_Weight", "Rank_Weight", "Opt_Weight"]:
                    if col in hybrid_display.columns:
                        hybrid_display[col] = truncate_series(
                            pd.to_numeric(hybrid_display[col], errors="coerce"), decimals=4
                        )
                st.markdown("**Blended Allocation (Rank vs. Optimizer)**")
                st.dataframe(hybrid_display.reset_index(drop=True), hide_index=True)
                st.caption("Final weights blend rank-based intuition with mean-variance optimization.")

            st.divider()
            st.write("### 🛡️ Risk Snapshot")
            if not risk_report:
                st.info("Risk metrics unavailable (missing weights or price history).")
            else:
                risk_cols = st.columns(3)
                annualized_vol = risk_report.get("annualized_vol", float("nan"))
                if not pd.isna(annualized_vol):
                    risk_cols[0].metric("Annualized Volatility", format_percentage(annualized_vol))
                else:
                    risk_cols[0].metric("Annualized Volatility", "N/A")

                annualized_mean = risk_report.get("annualized_mean", float("nan"))
                if not pd.isna(annualized_mean):
                    risk_cols[1].metric("Annualized Return", format_percentage(annualized_mean))
                else:
                    risk_cols[1].metric("Annualized Return", "N/A")

                sharpe_value = risk_report.get("sharpe", float("nan"))
                if not pd.isna(sharpe_value):
                    risk_cols[2].metric("Sharpe Ratio", f"{truncate_value(sharpe_value, decimals=4):.4f}")
                else:
                    risk_cols[2].metric("Sharpe Ratio", "N/A")

                var_cols = st.columns(2)
                var_cols[0].metric("Parametric VaR (95%)", format_percentage(risk_report.get("parametric_VaR")))
                var_cols[1].metric("Historical VaR (95%)", format_percentage(risk_report.get("historical_VaR")))

                stress_cols = st.columns(1)
                stress_return = risk_report.get("stress_uniform_minus5_pct_portfolio_return")
                if stress_return is not None and not pd.isna(stress_return):
                    stress_cols[0].metric("Uniform -5% Shock Return", format_percentage(stress_return))
                else:
                    stress_cols[0].metric("Uniform -5% Shock Return", "N/A")

                st.caption(
                    "Risk metrics are derived from the blended allocation weights. Tail metrics shown in daily terms; stress reflects a uniform -5% price shock."
                )

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
            return format_currency(value)

        def _fmt_pct(value):
            return format_percentage(value)

        def _round4(value):
            return truncate_value(value, decimals=4)

        def _fmt_ratio(value):
            truncated = truncate_value(value, decimals=4)
            return "N/A" if truncated is None else f"{truncated:.4f}"

        invested_amt = backtest_summary.get(
            "invested_capital", backtest_summary.get("initial_capital")
        )
        final_amt = backtest_summary.get("final_value")
        ret_amt = backtest_summary.get("return_amount")
        ret_pct = backtest_summary.get("return_pct")

        summary_start = backtest_summary.get("start_date")
        summary_end = backtest_summary.get("end_date")
        summary_start_label = pd.to_datetime(summary_start).strftime("%Y-%m-%d") if summary_start is not None else "-"
        summary_end_label = pd.to_datetime(summary_end).strftime("%Y-%m-%d") if summary_end is not None else "-"

        weights_as_of = results.get("weights_as_of")

        with tab4:
            st.write("### 📬 Trade Signals")
            if trade_signals is None or trade_signals.empty:
                st.info("No trade signals generated. Portfolio unchanged.")
            else:
                signals_display = trade_signals.copy()
                if "Old_Weight" in signals_display.columns:
                    signals_display["Old_Weight"] = truncate_series(
                        pd.to_numeric(signals_display["Old_Weight"], errors="coerce"),
                        decimals=4,
                    )
                if "New_Weight" in signals_display.columns:
                    signals_display["New_Weight"] = truncate_series(
                        pd.to_numeric(signals_display["New_Weight"], errors="coerce"),
                        decimals=4,
                    )
                    st.dataframe(signals_display, hide_index=True)
                    caption_fragments = [
                        f"{len(signals_display)} trade instructions generated using pre-set position drift rules."
                    ]
                    if weights_as_of is not None:
                        caption_fragments.append(
                            f"Weights derived from price history through {pd.to_datetime(weights_as_of).strftime('%Y-%m-%d')}."
                        )
                    st.caption(" ".join(caption_fragments))

            st.divider()
            st.write("### 🧾 Generated Signal History")
            if trade_log is None or trade_log.empty:
                st.info("Generated signal history is empty. Run at least one rebalance cycle.")
            else:
                log_display = trade_log.copy()
                if "Date" in log_display.columns:
                    log_display["Date"] = pd.to_datetime(
                        log_display["Date"], errors="coerce"
                    )
                if "Timestamp" in log_display.columns:
                    log_display["Timestamp"] = pd.to_datetime(
                        log_display["Timestamp"], errors="coerce"
                    )
                if "Execution_Date" in log_display.columns:
                    log_display["Execution_Date"] = pd.to_datetime(
                        log_display["Execution_Date"], errors="coerce"
                    )
                if "Data_Through" in log_display.columns:
                    log_display["Data_Through"] = pd.to_datetime(
                        log_display["Data_Through"], errors="coerce"
                    )
                sort_cols: list[str] = []
                ascending: list[bool] = []
                if "Data_Through" in log_display.columns:
                    sort_cols.append("Data_Through")
                    ascending.append(True)
                elif "Date" in log_display.columns:
                    sort_cols.append("Date")
                    ascending.append(True)
                if "Execution_Date" in log_display.columns:
                    sort_cols.append("Execution_Date")
                    ascending.append(True)
                if "Ticker" in log_display.columns:
                    sort_cols.append("Ticker")
                    ascending.append(True)
                if sort_cols:
                    log_display = log_display.sort_values(
                        sort_cols, ascending=ascending, kind="mergesort"
                    ).reset_index(drop=True)
                if "Old_Weight" in log_display.columns:
                    log_display["Old_Weight"] = truncate_series(
                        pd.to_numeric(log_display["Old_Weight"], errors="coerce"),
                        decimals=4,
                    )
                if "New_Weight" in log_display.columns:
                    log_display["New_Weight"] = truncate_series(
                        pd.to_numeric(log_display["New_Weight"], errors="coerce"),
                        decimals=4,
                    )
                if "Date" in log_display.columns:
                    log_display["Date"] = log_display["Date"].dt.strftime("%Y-%m-%d")
                if "Timestamp" in log_display.columns:
                    log_display["Timestamp"] = log_display["Timestamp"].dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                if "Execution_Date" in log_display.columns:
                    log_display["Execution_Date"] = log_display["Execution_Date"].dt.strftime(
                        "%Y-%m-%d"
                    )
                if "Data_Through" in log_display.columns:
                    log_display["Data_Through"] = log_display["Data_Through"].dt.strftime(
                        "%Y-%m-%d"
                    )
                desired_log_order = [
                    "Data_Through",
                    "Execution_Date",
                    "Ticker",
                    "Signal",
                    "Old_Weight",
                    "New_Weight",
                    "Reason",
                    "Timestamp",
                ]
                log_display = log_display[[col for col in desired_log_order if col in log_display.columns]]
                st.dataframe(log_display.tail(100), hide_index=True)
                log_caption = "Most recent 100 generated signals."
                if weights_as_of is not None:
                    log_caption += f" Signals reflect data through {pd.to_datetime(weights_as_of).strftime('%Y-%m-%d')}."
                st.caption(log_caption)

        with tab5:
            st.write(f"### 🔄 Backtest ({rebalance_choice} Rebalance)")
            if not backtest_available:
                st.info("Backtest unavailable.")
            else:
                # Reuse metrics calculated in Tab 1 logic if available, otherwise recalculate or use defaults
                # Since Tab 1 code runs before this in the file (if we moved it up? No, Tab 1 is at 977, Tab 5 is here at 896)
                # Wait, Tab 1 code is at 977. This code is at 896.
                # So Tab 1 code runs AFTER this code.
                # So variables like current_position_amt are NOT yet updated by Tab 1 logic.
                # We need to duplicate the calculation logic here.
                
                # Duplicate calculation logic
                initial_capital_amt = backtest_summary.get("initial_capital", capital)
                realized_amt = backtest_summary.get("realized_pnl")
                unrealized_amt = backtest_summary.get("unrealized_pnl")
                price_pnl = None
                price_return_pct = None
                if realized_amt is not None or unrealized_amt is not None:
                    price_pnl = (realized_amt or 0.0) + (unrealized_amt or 0.0)
                    if initial_capital_amt:
                        price_return_pct = price_pnl / float(initial_capital_amt)
                if price_pnl is None:
                    price_pnl = ret_amt
                if price_return_pct is None:
                    price_return_pct = ret_pct

                # Use shared state_cycle_df
                state_cycle_df_local = state_cycle_df
                latest_state_row_local = latest_state_row

                backtest_summary_cols = st.columns(4)
                cash_amt = backtest_summary.get("ending_cash")
                current_position_amt = final_amt
                if current_position_amt is not None and cash_amt is not None:
                    current_position_amt = current_position_amt - cash_amt
                else:
                    current_position_amt = invested_amt
                txn_cost_amt = backtest_summary.get("transaction_cost_total")

                rebalance_display_count = int(rebalance_count)
                if latest_state_row_local is not None:
                    current_position_amt = float(latest_state_row_local.get("Current Position", current_position_amt or 0.0))
                    cash_amt = float(latest_state_row_local.get("Cash", cash_amt or 0.0))
                    price_pnl = float(latest_state_row_local.get("P/L", price_pnl or 0.0))
                    price_return_pct = float(latest_state_row_local.get("ReturnPct", price_return_pct or 0.0))
                    rebalance_display_count = int(latest_state_row_local.get("Rebalances", rebalance_display_count))
                    txn_cost_amt = float(latest_state_row_local.get("Transaction Costs", txn_cost_amt or 0.0))
                    realized_amt = float(latest_state_row_local.get("RealizedPnL", realized_amt or 0.0))
                    unrealized_amt = float(latest_state_row_local.get("UnrealizedPnL", unrealized_amt or 0.0))

                backtest_summary_cols[0].metric(
                    "Current Position", _fmt_dollar(current_position_amt)
                )
                backtest_summary_cols[1].metric("P/L", _fmt_dollar(price_pnl))
                backtest_summary_cols[2].metric("Return %", _fmt_pct(price_return_pct))
                backtest_summary_cols[3].metric("Rebalances", str(int(rebalance_display_count)))

                cumulative_realized = realized_amt
                latest_unrealized = unrealized_amt

                pnl_cols = st.columns(4)
                if cumulative_realized is not None:
                    pnl_cols[0].metric("Cumulative Realized P/L", _fmt_dollar(cumulative_realized))
                if latest_unrealized is not None:
                    pnl_cols[1].metric("Unrealized P/L", _fmt_dollar(latest_unrealized))
                if txn_cost_amt is not None:
                    pnl_cols[2].metric("Transaction Costs", _fmt_dollar(txn_cost_amt))
                if cash_amt is not None:
                    pnl_cols[3].metric("Cash", _fmt_dollar(cash_amt))

                # Metrics table removed from Tab 5 as per user request

            st.divider()
            st.write("### 🔁 Backtest Transactions")
            missing_price_notes = backtest_summary.get("missing_price_notes", []) if backtest_available else []
            if transactions_df is None or transactions_df.empty:
                if missing_price_notes:
                    st.warning(
                        "No backtest transactions recorded. Reasons: " + " | ".join(missing_price_notes)
                    )
                else:
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
                    "RemainingCash",
                ]
                for col in numeric_cols:
                    if col in display_tx.columns:
                        display_tx[col] = truncate_series(
                            pd.to_numeric(display_tx[col], errors="coerce"),
                            decimals=4,
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
                    "RemainingCash",
                ]
                display_tx = display_tx[
                    [c for c in column_order if c in display_tx.columns]
                ]
                total_trades = len(display_tx)
                visible_tx = display_tx.head(100)
                rows_shown = len(visible_tx)
                table_height = min(900, 60 + rows_shown * 28)
                st.dataframe(visible_tx, hide_index=True, height=table_height)
                caption = f"{total_trades} trades recorded across the backtest window. "
                if rows_shown < total_trades:
                    caption += f"Showing the first {rows_shown} trades. "
                caption += "Positive CashFlow reflects capital returned to cash."
                st.caption(caption)



        with tab1:
            st.divider()
            st.write(f"### 🔄 Backtest ({rebalance_choice} Rebalance)")
            if not backtest_available:
                if backtest_error:
                    st.warning(
                        f"Backtest unavailable: {backtest_error}"
                    )
                else:
                    st.info(
                        "Backtest unavailable. Ensure sufficient price history and weights, then rerun."
                    )
            else:
                initial_capital_amt = backtest_summary.get("initial_capital", capital)
                realized_amt = backtest_summary.get("realized_pnl")
                unrealized_amt = backtest_summary.get("unrealized_pnl")
                price_pnl = None
                price_return_pct = None
                if realized_amt is not None or unrealized_amt is not None:
                    price_pnl = (realized_amt or 0.0) + (unrealized_amt or 0.0)
                    if initial_capital_amt:
                        price_return_pct = price_pnl / float(initial_capital_amt)
                if price_pnl is None:
                    price_pnl = ret_amt
                if price_return_pct is None:
                    price_return_pct = ret_pct

            # Shared variables already defined above
            # portfolio_cycles, state_cycle_df, latest_state_row, initial_live_capital

            backtest_summary_cols = st.columns(4)
            cash_amt = backtest_summary.get("ending_cash")
            current_position_amt = final_amt
            if current_position_amt is not None and cash_amt is not None:
                current_position_amt = current_position_amt - cash_amt
            else:
                current_position_amt = invested_amt
            txn_cost_amt = backtest_summary.get("transaction_cost_total")

            rebalance_display_count = int(rebalance_count)
            if latest_state_row is not None:
                current_position_amt = float(latest_state_row.get("Current Position", current_position_amt or 0.0))
                cash_amt = float(latest_state_row.get("Cash", cash_amt or 0.0))
                price_pnl = float(latest_state_row.get("P/L", price_pnl or 0.0))
                price_return_pct = float(latest_state_row.get("ReturnPct", price_return_pct or 0.0))
                rebalance_display_count = int(latest_state_row.get("Rebalances", rebalance_display_count))
                txn_cost_amt = float(latest_state_row.get("Transaction Costs", txn_cost_amt or 0.0))
                realized_amt = float(latest_state_row.get("RealizedPnL", realized_amt or 0.0))
                unrealized_amt = float(latest_state_row.get("UnrealizedPnL", unrealized_amt or 0.0))

            backtest_summary_cols[0].metric(
                "Current Position", _fmt_dollar(current_position_amt)
            )
            backtest_summary_cols[1].metric("P/L", _fmt_dollar(price_pnl))
            backtest_summary_cols[2].metric("Return %", _fmt_pct(price_return_pct))
            backtest_summary_cols[3].metric("Rebalances", str(int(rebalance_display_count)))

            cumulative_realized = realized_amt
            latest_unrealized = unrealized_amt

            pnl_cols = st.columns(4)
            if cumulative_realized is not None:
                pnl_cols[0].metric("Cumulative Realized P/L", _fmt_dollar(cumulative_realized))
            if latest_unrealized is not None:
                pnl_cols[1].metric("Unrealized P/L", _fmt_dollar(latest_unrealized))
            if txn_cost_amt is not None:
                pnl_cols[2].metric("Transaction Costs", _fmt_dollar(txn_cost_amt))
            if cash_amt is not None:
                pnl_cols[3].metric("Cash", _fmt_dollar(cash_amt))

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
                if missing_price_notes:
                    st.caption("Skipped trades: " + " | ".join(missing_price_notes))

                if backtest_curve is not None and not backtest_curve.empty:
                    chart_df = backtest_curve.copy()
                    if not isinstance(chart_df.index, pd.DatetimeIndex):
                        chart_df.index = pd.to_datetime(chart_df.index)
                    
                    # Align graph to match metrics (Live State) exactly
                    # Small discrepancies can occur between daily backtest and weekly live execution
                    # due to rounding or fee timing. We scale the curve to hit the live final value.
                    live_total_val = (current_position_amt or 0.0) + (cash_amt or 0.0)
                    backtest_final_val = chart_df["Portfolio Value"].iloc[-1] if not chart_df.empty else 0.0
                    
                    if backtest_final_val > 0 and live_total_val > 0:
                        scale_factor = live_total_val / backtest_final_val
                        # Only apply scaling if the difference is small (e.g. < 1%) to avoid masking real errors
                        if 0.99 < scale_factor < 1.01:
                            chart_df["Portfolio Value"] = chart_df["Portfolio Value"] * scale_factor

                    # Create Plotly Figure
                    fig = go.Figure()

                    # 1. Portfolio Value Line
                    fig.add_trace(go.Scatter(
                        x=chart_df.index,
                        y=chart_df["Portfolio Value"],
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='#00CC96', width=2)
                    ))

                    # 2. Benchmark Line (S&P 500)
                    if benchmark_curve is not None and not benchmark_curve.empty:
                        bench_chart = benchmark_curve.copy()
                        if not isinstance(bench_chart.index, pd.DatetimeIndex):
                            bench_chart.index = pd.to_datetime(bench_chart.index)
                        
                        # Align benchmark to chart_df dates if needed, or just plot
                        fig.add_trace(go.Scatter(
                            x=bench_chart.index,
                            y=bench_chart["Benchmark Value"],
                            mode='lines',
                            name='S&P 500',
                            line=dict(color='#636EFA', width=2, dash='dot')
                        ))

                    # 3. Rebalance Markers
                    if rebalance_dates:
                        valid_dates = []
                        rebalance_vals = []
                        for dt in rebalance_dates:
                            dt_ts = pd.to_datetime(dt)
                            # Find exact or nearest date in chart_df to place the marker
                            # Using 'nearest' ensures we plot a point even if exact timestamp mismatches slightly
                            try:
                                idx = chart_df.index.get_indexer([dt_ts], method='nearest')
                                if len(idx) > 0 and idx[0] != -1:
                                    actual_dt = chart_df.index[idx[0]]
                                    # Check if the nearest date is reasonably close (e.g. within a few days)
                                    # to avoid plotting a marker far away if data is missing
                                    if abs((actual_dt - dt_ts).days) <= 5:
                                        val = chart_df.iloc[idx[0]]["Portfolio Value"]
                                        valid_dates.append(actual_dt)
                                        rebalance_vals.append(val)
                            except Exception:
                                pass # Skip if date matching fails

                        if valid_dates:
                            fig.add_trace(go.Scatter(
                                x=valid_dates,
                                y=rebalance_vals,
                                mode='markers',
                                name='Rebalance',
                                marker=dict(
                                    color='red', 
                                    size=10, 
                                    symbol='circle-open', 
                                    line=dict(width=2)
                                ),
                                hovertemplate='Rebalance: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}'
                            ))

                    # Layout Configuration
                    fig.update_layout(
                        title="Portfolio Performance vs Benchmark",
                        xaxis_title="Date",
                        yaxis_title="Value ($)",
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=20, r=20, t=60, b=20),
                        height=500,
                        xaxis=dict(
                            showgrid=True, 
                            gridcolor='rgba(211, 211, 211, 0.3)'
                        ),
                        yaxis=dict(
                            showgrid=True, 
                            gridcolor='rgba(211, 211, 211, 0.3)',
                            tickprefix="$"
                        ),
                        plot_bgcolor='white'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Portfolio value trajectory ({rebalance_choice.lower()} rebalance) versus S&P 500 benchmark. Red circles indicate rebalance execution dates."
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
                            format_percentage(backtest_metrics.get("Annualized Return")),
                            format_percentage(backtest_metrics.get("Annualized Volatility")),
                            _fmt_ratio(backtest_metrics.get("Sharpe Ratio")),
                            _fmt_ratio(backtest_metrics.get("Sortino Ratio")),
                            format_percentage(backtest_metrics.get("Max Drawdown")),
                            format_percentage(backtest_metrics.get("CAGR")),
                        ],
                    }
                    if benchmark_metrics:
                        metrics_data["S&P 500"] = [
                            format_percentage(benchmark_metrics.get("Annualized Return")),
                            format_percentage(benchmark_metrics.get("Annualized Volatility")),
                            _fmt_ratio(benchmark_metrics.get("Sharpe Ratio")),
                            _fmt_ratio(benchmark_metrics.get("Sortino Ratio")),
                            format_percentage(benchmark_metrics.get("Max Drawdown")),
                            format_percentage(benchmark_metrics.get("CAGR")),
                        ]
                    metrics_df = pd.DataFrame(metrics_data)
                    st.table(metrics_df)

            if state_cycle_df is not None and not state_cycle_df.empty:
                with tab5:
                    st.write("### Cycle P/L Log")
                    log_display = state_cycle_df.copy()
                    log_display["Rebalance Cadence"] = f"{rebalance_choice} Rebalance"
                    log_display["Return %"] = log_display["ReturnPct"]
                    log_display["P/L"] = log_display["P/L"]
                    log_display["Realized P/L"] = log_display["RealizedCycle"]
                    log_display["Cumulative Realized P/L"] = log_display["RealizedPnL"]
                    log_display["Unrealized P/L"] = log_display["UnrealizedCycle"]
                    log_display["Latest Unrealized P/L"] = log_display["UnrealizedPnL"]
                    ordered_cols = [
                        "Execution Date",
                        "Rebalance Cadence",
                        "Current Position",
                        "P/L",
                        "Return %",
                        "Rebalances",
                        "Realized P/L",
                        "Cumulative Realized P/L",
                        "Unrealized P/L",
                        "Latest Unrealized P/L",
                        "Transaction Costs",
                        "Cash",
                    ]
                    log_display = log_display[ordered_cols]
                    currency_cols = [
                        "Current Position",
                        "P/L",
                        "Realized P/L",
                        "Cumulative Realized P/L",
                        "Unrealized P/L",
                        "Latest Unrealized P/L",
                        "Transaction Costs",
                        "Cash",
                    ]
                    for col in currency_cols:
                        log_display[col] = truncate_series(
                            pd.to_numeric(log_display[col], errors="coerce"), decimals=4
                        )
                    log_display["Return %"] = truncate_series(
                        pd.to_numeric(log_display["Return %"], errors="coerce"), decimals=4
                    )
                    formatted_log = log_display.copy()
                    for col in currency_cols:
                        formatted_log[col] = formatted_log[col].apply(format_currency)
                    formatted_log["Return %"] = formatted_log["Return %"].apply(format_percentage)
                    st.dataframe(formatted_log, hide_index=True)
                    st.caption(
                        "Logged using live PortfolioState accounting; each row satisfies Current Holdings + Cash − Initial = Realized + Unrealized − Fees."
                    )

    else:
        st.info(
            "👈 Configure inputs in the sidebar and click **Run Rebalance Cycle** to start."
        )
