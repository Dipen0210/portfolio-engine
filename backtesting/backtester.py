# backtesting/backtester.py
import math

import numpy as np
import pandas as pd

from execution.transaction_cost import calc_commission, apply_slippage

from .metrics import compute_performance_metrics


def _prepare_price_series(df: pd.DataFrame, ticker: str, column: str) -> pd.Series | None:
    """
    Ensure the OHLCV DataFrame is indexed by datetime and return the requested price series.
    """
    if column not in df.columns:
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
    return working[column].rename(ticker)


def backtest_portfolio(
    price_data_dict,
    weights_df: pd.DataFrame,
    trade_log_df: pd.DataFrame | None = None,
    rebalance_freq: str = "M",
    initial_capital: float = 100000,
    benchmark_df: pd.DataFrame | None = None,
    start_date=None,
    end_date=None,
    commission_per_trade: float = 1.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 5.0,
):
    """
    Simulate a rebalancing backtest with explicit transaction logging.

    Parameters
    ----------
    price_data_dict : dict[str, pd.DataFrame]
        Mapping of ticker to OHLCV DataFrame (must include 'Close').
    weights_df : pd.DataFrame
        Optimised weights with columns ['Ticker', 'Weight'] or snapshot history
        with optional 'Date' column.
    trade_log_df : pd.DataFrame, optional
        Trade signal log used for annotating transactions (Date, Ticker, Signal,
        Old_Weight, New_Weight, Reason, Timestamp).
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
    commission_per_trade : float, optional
        Flat commission applied to every order (default $1).
    commission_bps : float, optional
        Additional commission expressed in basis points of notional.
    slippage_bps : float, optional
        Slippage applied to fills (default 5 bps).

    Returns
    -------
    dict
        Contains portfolio curve, metrics, benchmark comparison, transaction
        history, and rebalance diagnostics.
    """

    if weights_df.empty:
        raise ValueError("Backtest failed: no weights provided for simulation.")

    if "Weight" not in weights_df.columns:
        raise ValueError("Backtest failed: weights DataFrame must include a 'Weight' column.")

    working_weights = weights_df.copy()
    working_weights["Weight"] = (
        pd.to_numeric(working_weights["Weight"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    working_weights = working_weights.dropna(subset=["Weight"])
    if working_weights.empty:
        raise ValueError("Backtest failed: weights must contain valid numeric allocations.")

    has_dated_weights = "Date" in working_weights.columns
    weight_snapshots: list[dict] = []

    if has_dated_weights:
        working_weights["Date"] = pd.to_datetime(working_weights["Date"], errors="coerce")
        if "OriginalDate" in working_weights.columns:
            working_weights["OriginalDate"] = pd.to_datetime(
                working_weights["OriginalDate"], errors="coerce"
            )
            working_weights.loc[
                working_weights["OriginalDate"].isna(), "OriginalDate"
            ] = working_weights.loc[working_weights["OriginalDate"].isna(), "Date"]
        else:
            working_weights["OriginalDate"] = working_weights["Date"]

        working_weights["Date"] = working_weights["Date"].dt.normalize()
        working_weights = working_weights.dropna(subset=["Date"])
        if working_weights.empty:
            raise ValueError("Backtest failed: dated weights have invalid timestamps.")
        for dt, group in working_weights.groupby("Date", sort=True):
            series = (
                group.set_index("Ticker")["Weight"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            series = series[series > 0]
            if not series.empty:
                original_series = pd.to_datetime(group["OriginalDate"], errors="coerce").dropna()
                original_ts = original_series.iloc[-1] if not original_series.empty else pd.Timestamp(dt)
                weight_snapshots.append(
                    {
                        "requested_date": pd.Timestamp(dt),
                        "original_date": pd.Timestamp(original_ts),
                        "weights": series,
                    }
                )
        if not weight_snapshots:
            raise ValueError("Backtest failed: no usable weight snapshots after filtering.")
    else:
        series = (
            working_weights.set_index("Ticker")["Weight"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        series = series[series > 0]
        if series.empty:
            raise ValueError("Backtest failed: weights must contain positive allocations.")
        weight_snapshots.append(
            {
                "requested_date": None,
                "original_date": None,
                "weights": series,
            }
        )

    all_tickers = sorted({ticker for snap in weight_snapshots for ticker in snap["weights"].index})
    if not all_tickers:
        raise ValueError("Backtest failed: no tickers found in the supplied weights.")

    trade_log_lookup = None
    if trade_log_df is not None and not trade_log_df.empty:
        log_work = trade_log_df.copy()
        log_work["Date"] = pd.to_datetime(log_work["Date"], errors="coerce")
        log_work = log_work.dropna(subset=["Date"])
        log_work["Ticker"] = log_work["Ticker"].astype(str)
        if "Timestamp" in log_work.columns:
            log_work["Timestamp"] = pd.to_datetime(log_work["Timestamp"], errors="coerce")
        else:
            log_work["Timestamp"] = log_work["Date"]
        log_work["LookupDate"] = log_work["Date"].dt.normalize()
        log_work = log_work.dropna(subset=["LookupDate", "Ticker"])
        log_work = log_work.sort_values(["LookupDate", "Ticker", "Timestamp"])
        trade_log_lookup = log_work.groupby(["LookupDate", "Ticker"]).last()

    close_series = []
    open_series = []
    for ticker in all_tickers:
        df = price_data_dict.get(ticker)
        if df is None or df.empty:
            continue
        close_ser = _prepare_price_series(df, ticker, "Close")
        open_ser = _prepare_price_series(df, ticker, "Open")
        if close_ser is not None and not close_ser.empty:
            close_series.append(close_ser)
        if open_ser is not None and not open_ser.empty:
            open_series.append(open_ser)

    if not close_series:
        raise ValueError("Backtest failed: no valid price history for the selected tickers.")

    close_prices = pd.concat(close_series, axis=1)
    close_prices = close_prices.sort_index()
    close_prices = close_prices.loc[~close_prices.index.duplicated()]
    close_prices = close_prices.dropna(axis=1, how="all")

    open_prices = None
    if open_series:
        open_prices = pd.concat(open_series, axis=1)
        open_prices = open_prices.sort_index()
        open_prices = open_prices.loc[~open_prices.index.duplicated()]
        open_prices = open_prices.dropna(axis=1, how="all")

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

    available_tickers = [t for t in close_prices.columns if t in all_tickers]
    if not available_tickers:
        raise ValueError("Backtest failed: no overlapping price history within the selected window.")

    close_prices = close_prices[available_tickers]
    if open_prices is not None:
        open_prices = open_prices.reindex(close_prices.index)
        open_prices = open_prices[available_tickers]
        open_prices = open_prices.ffill()

    trading_days = close_prices.index
    if len(trading_days) == 0:
        raise ValueError("Backtest failed: no trading days available in the window.")

    def _align_to_trading_day(
        candidate: pd.Timestamp,
    ) -> tuple[pd.Timestamp | None, str | None]:
        """
        Align a requested timestamp to an available trading day.

        Returns a tuple of (aligned_date, alignment_direction) where
        alignment_direction is one of:
            * None: no adjustment needed (same day)
            * "forward": shifted to the next available trading day
            * "backward": shifted to the prior available trading day (price data missing)
        """
        if len(trading_days) == 0:
            return None, None

        ts = pd.Timestamp(candidate).normalize()

        try:
            exact_loc = trading_days.get_loc(ts)
            return trading_days[exact_loc], None
        except KeyError:
            pass

        idx = trading_days.searchsorted(ts, side="left")
        if idx < len(trading_days):
            aligned = trading_days[idx]
            direction = None
            if aligned.normalize() != ts:
                direction = "forward"
            return aligned, direction

        # Candidate occurs after the last available trading day -> fall back to most recent day.
        aligned = trading_days[-1]
        direction = "backward"
        if aligned.normalize() == ts:
            direction = None
        return aligned, direction

    snapshot_schedule_map: dict[pd.Timestamp, dict] = {}
    calendar_notes: list[str] = []
    missing_price_notes: list[str] = []
    share_precision = 4

    def _round_down_shares(value: float) -> float:
        if value <= 0:
            return 0.0
        factor = 10 ** share_precision
        return math.floor(value * factor + 1e-12) / factor

    def _round_to_shares(value: float) -> float:
        factor = 10 ** share_precision
        if value >= 0:
            return math.floor(value * factor + 0.5) / factor
        return -math.floor(-value * factor + 0.5) / factor
    for snap in weight_snapshots:
        weights_series = snap["weights"]
        requested_date = snap.get("requested_date")
        original_ts = snap.get("original_date")

        if requested_date is None:
            aligned_date = trading_days[0]
            normalized_requested = aligned_date
            alignment_direction = None
        else:
            candidate = pd.Timestamp(requested_date)
            normalized_requested = candidate.normalize()
            aligned_date, alignment_direction = _align_to_trading_day(candidate)
            if aligned_date is None:
                aligned_date, alignment_direction = _align_to_trading_day(normalized_requested)
            if aligned_date is None:
                continue
        if requested_date is None:
            alignment_direction = None

        reindexed = weights_series.reindex(available_tickers).fillna(0.0)
        total_weight = reindexed.sum()
        if total_weight <= 0:
            continue

        holiday_note = None
        if requested_date is not None and aligned_date.normalize() != normalized_requested.normalize():
            if alignment_direction == "backward":
                holiday_note = (
                    f"Signal dated {normalized_requested.date()} executed on prior trading day "
                    f"{aligned_date.date()} (latest price data unavailable on signal date)."
                )
            else:
                holiday_note = (
                    f"Signal dated {normalized_requested.date()} executed on next trading day {aligned_date.date()}."
                )
            calendar_notes.append(holiday_note)

        snapshot_schedule_map[aligned_date] = {
            "weights": (reindexed / total_weight),
            "requested_date": requested_date,
            "original_date": original_ts if original_ts is not None else aligned_date,
            "holiday_note": holiday_note,
        }

    snapshot_schedule = [
        {"aligned_date": date, **meta}
        for date, meta in sorted(snapshot_schedule_map.items())
        if date in trading_days
    ]
    if not snapshot_schedule:
        raise ValueError("Backtest failed: no valid weight snapshots aligned to trading calendar.")

    first_snapshot_day = snapshot_schedule[0]["aligned_date"]
    if trading_days[0] < first_snapshot_day:
        close_prices = close_prices.loc[close_prices.index >= first_snapshot_day]
        trading_days = close_prices.index
        if len(trading_days) == 0:
            raise ValueError("Backtest failed: insufficient price history after aligning to weight snapshots.")

    rebalance_rule = (rebalance_freq or "").strip().upper()
    rebalance_candidate_set: set[pd.Timestamp] = set()
    if not has_dated_weights and rebalance_rule:
        schedule = [trading_days[0]]
        raw_schedule = pd.date_range(
            start=trading_days[0],
            end=trading_days[-1],
            freq=rebalance_rule,
        )
        for candidate in raw_schedule[1:]:
            aligned, _ = _align_to_trading_day(candidate)
            if aligned is not None and aligned != schedule[-1]:
                schedule.append(aligned)
        rebalance_candidate_set = set(schedule[1:])

    holdings = pd.Series(0.0, index=close_prices.columns, dtype=float)
    holdings_cost = pd.Series(0.0, index=close_prices.columns, dtype=float)
    holdings_entry_date = pd.Series(pd.NaT, index=close_prices.columns, dtype="datetime64[ns]")
    holdings_cycle_open_price = pd.Series(np.nan, index=close_prices.columns, dtype=float)
    cash_balance = float(initial_capital)
    transactions: list[dict] = []
    portfolio_records: list[dict] = []
    executed_rebalance_dates: list[pd.Timestamp] = []
    realized_pnl = 0.0
    unrealized_pnl = 0.0
    total_transaction_cost = 0.0

    prev_target_weights = pd.Series(0.0, index=holdings.index, dtype=float)
    current_target_weights: pd.Series | None = None
    next_snapshot_idx = 0

    next_day_price: pd.Series | None = None
    for idx, day in enumerate(trading_days):
        next_day_price = None
        if idx + 1 < len(trading_days):
            next_day_price = close_prices.loc[trading_days[idx + 1]]
        close_prices_today = close_prices.loc[day]
        open_prices_today = None
        if open_prices is not None:
            try:
                open_prices_today = open_prices.loc[day]
            except KeyError:
                open_prices_today = None

        weight_update_today = False
        plan_old_weights = prev_target_weights.copy()
        plan_new_weights = current_target_weights.copy() if current_target_weights is not None else None
        schedule_meta_today = None
        holiday_note_today: str | None = None

        if next_snapshot_idx < len(snapshot_schedule) and day == snapshot_schedule[next_snapshot_idx]["aligned_date"]:
            weight_update_today = True
            schedule_meta_today = snapshot_schedule[next_snapshot_idx]
            plan_old_weights = current_target_weights.copy() if current_target_weights is not None else prev_target_weights.copy()
            current_target_weights = schedule_meta_today["weights"].copy()
            plan_new_weights = current_target_weights.copy()
            holiday_note_today = schedule_meta_today.get("holiday_note")
            next_snapshot_idx += 1

        if current_target_weights is None:
            # No weights available yet; skip until first allocation snapshot.
            portfolio_value = float((holdings * close_prices_today).sum() + cash_balance)
            portfolio_records.append({"Date": day, "Portfolio Value": portfolio_value})
            continue

        should_rebalance_today = False
        event_label = None
        if weight_update_today:
            should_rebalance_today = True
            event_label = (
                "Initial Allocation"
                if plan_old_weights.sum() <= 1e-12
                else "Allocation Update"
            )
        elif day in rebalance_candidate_set:
            should_rebalance_today = True
            event_label = "Scheduled Rebalance"

        base_value = float((holdings * close_prices_today).sum() + cash_balance)
        day_open_price_cache: dict[str, float] = {}

        if holiday_note_today:
            transactions.append(
                {
                    "Date": day,
                    "Event": "Calendar Adjustment",
                    "Ticker": "-",
                    "Action": "HOLD",
                    "Signal": "HOLD",
                    "Shares": 0.0,
                    "Price": 0.0,
                    "TradeValue": 0.0,
                    "CashFlow": 0.0,
                    "Old_Weight": 0.0,
                    "New_Weight": 0.0,
                    "Net_Weight": 0.0,
                    "Reason": holiday_note_today,
                    "Timestamp": schedule_meta_today.get("original_date") if schedule_meta_today else pd.NaT,
                }
            )

        if should_rebalance_today and plan_new_weights is not None and base_value > 0:
            plan_old_aligned = plan_old_weights.reindex(holdings.index, fill_value=0.0)
            plan_new_aligned = plan_new_weights.reindex(holdings.index, fill_value=0.0)

            target_shares_map: dict[str, float] = {}
            for ticker in holdings.index:
                price = close_prices_today[ticker]
                if pd.isna(price) or price <= 0:
                    target_shares_map[ticker] = float(holdings.at[ticker])
                    continue
                weight_plan = max(float(plan_new_aligned.at[ticker]), 0.0)
                desired_value = weight_plan * base_value
                desired_shares = desired_value / price if price > 0 else float(holdings.at[ticker])
                target_shares_map[ticker] = _round_down_shares(desired_shares)

            day_transactions: list[dict] = []
            cash_pointer = cash_balance
            processing_meta: list[tuple[str, float, float]] = []
            for ticker in holdings.index:
                target_value = max(target_shares_map.get(ticker, float(holdings.at[ticker])), 0.0)
                processing_meta.append((ticker, target_value, float(holdings.at[ticker])))

            selling_meta = [
                meta for meta in processing_meta if meta[1] + 1e-9 < meta[2]
            ]
            buying_meta = [
                meta for meta in processing_meta if meta not in selling_meta
            ]

            for ticker, target_shares, old_shares in selling_meta + buying_meta:
                close_price = close_prices_today[ticker]
                if pd.isna(close_price) or close_price <= 0:
                    missing_price_notes.append(
                        f"{pd.Timestamp(day).date()} · {ticker}: missing close price; trade skipped."
                    )
                    continue
                execution_open = None
                if open_prices_today is not None and ticker in open_prices_today.index:
                    execution_open = open_prices_today[ticker]
                if pd.isna(execution_open) or execution_open is None or execution_open <= 0:
                    execution_open = close_price
                if pd.isna(execution_open) or execution_open <= 0:
                    missing_price_notes.append(
                        f"{pd.Timestamp(day).date()} · {ticker}: missing open/close price; trade skipped."
                    )
                    continue
                day_open_price_cache[ticker] = float(execution_open)

                target_shares = max(target_shares_map.get(ticker, float(holdings.at[ticker])), 0.0)
                target_shares = _round_to_shares(target_shares)
                old_shares = float(holdings.at[ticker])
                delta_shares = target_shares - old_shares
                shares_sell = max(old_shares - target_shares, 0.0)
                shares_hold = min(old_shares, target_shares)

                prev_open = holdings_cycle_open_price.at[ticker]
                if shares_hold > 0 and not pd.isna(prev_open):
                    unrealized_pnl += (execution_open - prev_open) * shares_hold
                if shares_sell > 0 and not pd.isna(prev_open):
                    realized_pnl += (execution_open - prev_open) * shares_sell

                if abs(delta_shares) <= 1e-9:
                    if target_shares > 0:
                        holdings_cycle_open_price.at[ticker] = float(execution_open)
                    else:
                        holdings_cycle_open_price.at[ticker] = np.nan
                    continue

                action = "BUY" if delta_shares > 0 else "SELL"
                shares_traded = abs(delta_shares)
                fill_price = apply_slippage(execution_open, action, slippage_bps=slippage_bps)
                trade_value = shares_traded * fill_price
                if action == "BUY":
                    while shares_traded > 0:
                        commission = calc_commission(
                            order_value=fill_price * shares_traded,
                            per_trade=commission_per_trade,
                            bps=commission_bps,
                        )
                        total_cost = fill_price * shares_traded + commission
                        if total_cost <= cash_pointer + 1e-8:
                            break
                        shares_traded -= 1
                    if shares_traded <= 0:
                        holdings_cycle_open_price.at[ticker] = float(execution_open)
                        continue
                    trade_value = shares_traded * fill_price
                commission = calc_commission(
                    order_value=trade_value,
                    per_trade=commission_per_trade,
                    bps=commission_bps,
                )
                total_transaction_cost += commission
                cash_flow = -trade_value if action == "BUY" else trade_value
                transaction_cost_signed = -commission
                cash_pointer += cash_flow + transaction_cost_signed

                actual_old_weight = (old_shares * close_price) / base_value if base_value > 0 else 0.0
                actual_new_weight = (target_shares * close_price) / base_value if base_value > 0 else 0.0

                if action == "BUY":
                    holdings_cost.at[ticker] += trade_value
                    if old_shares <= 1e-9:
                        holdings_entry_date.at[ticker] = pd.Timestamp(day)
                else:
                    existing_cost = float(holdings_cost.at[ticker])
                    cost_per_share = existing_cost / old_shares if old_shares > 0 else 0.0
                    cost_removed = min(existing_cost, cost_per_share * shares_traded)
                    holdings_cost.at[ticker] = max(existing_cost - cost_removed, 0.0)
                if action == "BUY":
                    target_shares = old_shares + shares_traded
                actual_new_weight = (target_shares * close_price) / base_value if base_value > 0 else 0.0
                if target_shares > 0:
                    holdings_cycle_open_price.at[ticker] = float(execution_open)
                else:
                    holdings_cycle_open_price.at[ticker] = np.nan
                if holdings.at[ticker] <= 0 or holdings_cost.at[ticker] < 1e-9:
                    holdings_cost.at[ticker] = 0.0
                    holdings_entry_date.at[ticker] = pd.NaT

                if weight_update_today:
                    old_weight_plan = float(plan_old_aligned.at[ticker])
                    prev_plan_sum = plan_old_aligned.sum()
                    if prev_plan_sum <= 1e-12:
                        signal = "BUY"
                        reason = "Initial portfolio allocation"
                    elif old_weight_plan <= 1e-12 and target_shares > 0:
                        signal = "BUY"
                        reason = "Newly added to optimized portfolio"
                    elif target_shares <= 1e-12 and old_weight_plan > 0:
                        signal = "SELL"
                        reason = "Removed from new optimized portfolio"
                    else:
                        target_weight_plan = float(plan_new_aligned.at[ticker])
                        drift = abs(target_weight_plan - old_weight_plan) / old_weight_plan if old_weight_plan > 0 else 0.0
                        signal = "REBALANCE"
                        reason = f"Weight drifted by {drift:.2%}"
                    old_weight_to_log = old_weight_plan
                else:
                    signal = "REBALANCE"
                    target_weight_plan = float(plan_new_aligned.at[ticker])
                    comparison_base = actual_old_weight if actual_old_weight > 0 else target_weight_plan
                    if comparison_base > 0:
                        drift = abs(target_weight_plan - actual_old_weight) / comparison_base
                    else:
                        drift = float("inf")
                    reason = (
                        f"Weight drifted by {drift:.2%}"
                        if np.isfinite(drift)
                        else "Weight reset during rebalance"
                    )
                    old_weight_to_log = actual_old_weight

                recorded_signal = signal
                recorded_reason = reason
                recorded_old_weight = float(old_weight_to_log)
                recorded_new_weight = float(actual_new_weight)
                recorded_timestamp = None
                log_entry = None
                if trade_log_lookup is not None:
                    day_key = pd.Timestamp(day).normalize()
                    try:
                        log_entry = trade_log_lookup.loc[(day_key, str(ticker))]
                    except KeyError:
                        log_entry = None
                if log_entry is not None:
                    if isinstance(log_entry, pd.DataFrame):
                        log_entry = log_entry.iloc[-1]
                    if "Signal" in log_entry and pd.notna(log_entry["Signal"]):
                        recorded_signal = str(log_entry["Signal"])
                    if "Reason" in log_entry and pd.notna(log_entry["Reason"]):
                        recorded_reason = log_entry["Reason"]
                    if "Old_Weight" in log_entry and pd.notna(log_entry["Old_Weight"]):
                        recorded_old_weight = float(log_entry["Old_Weight"])
                    if "New_Weight" in log_entry and pd.notna(log_entry["New_Weight"]):
                        recorded_new_weight = float(log_entry["New_Weight"])
                    if "Timestamp" in log_entry and pd.notna(log_entry["Timestamp"]):
                        recorded_timestamp = pd.to_datetime(log_entry["Timestamp"])

                day_transactions.append(
                    {
                        "Date": day,
                        "Event": event_label or "Rebalance",
                        "Ticker": ticker,
                        "Action": action,
                        "Signal": recorded_signal,
                        "Shares": shares_traded,
                        "Price": fill_price,
                        "TradeValue": trade_value,
                        "CashFlow": cash_flow,
                        "Transaction Cost": transaction_cost_signed,
                        "RemainingCash": cash_pointer,
                        "Old_Weight": recorded_old_weight,
                        "New_Weight": recorded_new_weight,
                        "Net_Weight": recorded_new_weight - recorded_old_weight,
                        "Reason": recorded_reason,
                        "Timestamp": recorded_timestamp,
                    }
                )
                holdings.at[ticker] = max(target_shares, 0.0)
                if holdings.at[ticker] <= 0:
                    holdings_entry_date.at[ticker] = pd.NaT
                if holdings.at[ticker] <= 0 or holdings_cost.at[ticker] < 1e-9:
                    holdings_cost.at[ticker] = 0.0

            if day_transactions:
                total_cash_flow = sum(
                    tx["CashFlow"] + tx.get("Transaction Cost", 0.0)
                    for tx in day_transactions
                )
                cash_balance += total_cash_flow
                if abs(cash_balance) <= 1e-9:
                    cash_balance = 0.0
                transactions.extend(day_transactions)
                executed_rebalance_dates.append(day)
        if should_rebalance_today:
            for ticker in holdings.index:
                current_shares = float(holdings.at[ticker])
                if current_shares > 0:
                    base_px = day_open_price_cache.get(ticker)
                    if base_px is None and open_prices_today is not None and ticker in open_prices_today.index:
                        base_px = open_prices_today[ticker]
                    if base_px is None or pd.isna(base_px):
                        base_px = close_prices_today.get(ticker, np.nan)
                    holdings_cycle_open_price.at[ticker] = float(base_px) if base_px is not None else np.nan
                else:
                    holdings_cycle_open_price.at[ticker] = np.nan
        if weight_update_today and current_target_weights is not None:
            prev_target_weights = current_target_weights.copy()

        if next_day_price is not None:
            valuation_prices = next_day_price.copy()
            if open_prices is not None:
                try:
                    next_day_open = open_prices.loc[trading_days[idx + 1]]
                    valuation_prices = next_day_open.reindex(valuation_prices.index).fillna(valuation_prices)
                except KeyError:
                    pass
            valuation_prices = valuation_prices.reindex(holdings.index).fillna(close_prices_today.reindex(holdings.index))
        else:
            valuation_prices = pd.Series(index=holdings.index, dtype=float)
            for ticker in holdings.index:
                price = close_prices_today.get(ticker, np.nan)
                shares = float(holdings.at[ticker])
                entry_ts = holdings_entry_date.at[ticker]
                if shares > 0 and not pd.isna(entry_ts) and entry_ts == day:
                    cost_basis = float(holdings_cost.at[ticker])
                    price = cost_basis / shares if shares > 0 else price
                valuation_prices.at[ticker] = price
        portfolio_value = float((holdings * valuation_prices).sum() + cash_balance)
        portfolio_records.append({"Date": day, "Portfolio Value": portfolio_value})

    portfolio_df = pd.DataFrame(portfolio_records).set_index("Date")
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    portfolio_df["Portfolio Return"] = (
        portfolio_df["Portfolio Value"].pct_change().fillna(0.0)
    )

    benchmark_curve = None
    benchmark_metrics = None
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_series = _prepare_price_series(benchmark_df, "Benchmark", "Close")
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
        for col in [
            "Shares",
            "Price",
            "TradeValue",
            "CashFlow",
            "Transaction Cost",
            "RemainingCash",
            "Old_Weight",
            "New_Weight",
            "Net_Weight",
        ]:
            transactions_df[col] = pd.to_numeric(transactions_df[col], errors="coerce")
        if "Timestamp" in transactions_df.columns:
            transactions_df["Timestamp"] = pd.to_datetime(transactions_df["Timestamp"])
        transactions_df = transactions_df.sort_values(["Date", "Event", "Ticker"]).reset_index(drop=True)

    final_day = trading_days[-1]
    final_prices = close_prices.loc[final_day]

    holdings_snapshot = (
        holdings[holdings.abs() > 1e-9]
        .to_frame(name="Shares")
        .reset_index()
        .rename(columns={"index": "Ticker"})
    )
    if not holdings_snapshot.empty:
        holdings_snapshot["Price"] = holdings_snapshot["Ticker"].map(final_prices)
        holdings_snapshot = holdings_snapshot.dropna(subset=["Price"])
        holdings_snapshot["MarketValue"] = holdings_snapshot["Shares"] * holdings_snapshot["Price"]
        holdings_snapshot["Shares"] = holdings_snapshot["Shares"].round(6)
        holdings_snapshot["Price"] = holdings_snapshot["Price"].round(6)
        holdings_snapshot["MarketValue"] = holdings_snapshot["MarketValue"].round(4)
    else:
        holdings_snapshot = pd.DataFrame(columns=["Ticker", "Shares", "Price", "MarketValue"])
    if abs(cash_balance) > 1e-6:
        cash_row = pd.DataFrame(
            [
                {
                    "Ticker": "CASH",
                    "Shares": np.nan,
                    "Price": np.nan,
                    "MarketValue": round(float(cash_balance), 4),
                }
            ]
        )
        holdings_snapshot = pd.concat([holdings_snapshot, cash_row], ignore_index=True)
    holdings_snapshot = holdings_snapshot.reindex(columns=["Ticker", "Shares", "Price", "MarketValue"])

    if calendar_notes:
        calendar_notes = list(dict.fromkeys(calendar_notes))
    if missing_price_notes:
        missing_price_notes = list(dict.fromkeys(missing_price_notes))

    effective_final_value = float(initial_capital + realized_pnl + unrealized_pnl)
    effective_return_amount = float(realized_pnl + unrealized_pnl)
    effective_return_pct = (
        effective_return_amount / float(initial_capital) if initial_capital else float("nan")
    )

    if not portfolio_df.empty:
        portfolio_df.at[portfolio_df.index[-1], "Portfolio Value"] = effective_final_value
        portfolio_df["Portfolio Return"] = portfolio_df["Portfolio Value"].pct_change().fillna(0.0)

    portfolio_returns = portfolio_df["Portfolio Return"]
    metrics = compute_performance_metrics(portfolio_returns) if not portfolio_returns.empty else {}

    summary = {
        "initial_capital": float(initial_capital),
        "invested_capital": float(initial_capital - total_transaction_cost),
        "final_value": effective_final_value,
        "return_amount": effective_return_amount,
        "return_pct": effective_return_pct,
        "rebalance_count": len(executed_rebalance_dates),
        "start_date": portfolio_df.index[0],
        "end_date": portfolio_df.index[-1],
        "ending_cash": float(cash_balance),
        "calendar_notes": calendar_notes,
        "missing_price_notes": missing_price_notes,
        "transaction_cost_total": float(total_transaction_cost),
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
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
        "holdings": holdings_snapshot,
    }
