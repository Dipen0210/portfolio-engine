"""
strategies/registry.py
----------------------
Utility helpers to resolve strategy runners based on the user's selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd

from strategies.momentum import run_momentum_strategy
from strategies.mean_reversion import run_mean_reversion_strategy

StrategyRunner = Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


def _normalize_strategy_output(
    df: pd.DataFrame, strategy_name: str
) -> pd.DataFrame:
    """
    Ensure every strategy returns the standard schema required downstream.
    """
    if df.empty:
        return df

    required_columns = {"Ticker", "Strategy_Score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Strategy '{strategy_name}' result missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["Strategy_Score"] = df["Strategy_Score"].astype(float).fillna(0.0)
    df = df.sort_values("Strategy_Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


@dataclass(frozen=True)
class StrategySpec:
    """Metadata describing a portfolio construction strategy."""

    name: str
    runner: StrategyRunner
    lookback_window: int


STRATEGY_REGISTRY: Dict[str, StrategySpec] = {
    "Momentum": StrategySpec(
        name="Momentum",
        runner=run_momentum_strategy,
        lookback_window=60,
    ),
    "Mean Reversion": StrategySpec(
        name="Mean Reversion",
        runner=run_mean_reversion_strategy,
        lookback_window=120,
    ),
    # Additional strategies can be registered here with their preferred lookback.
}


def get_strategy_spec(strategy_name: str) -> StrategySpec:
    """
    Resolve the metadata describing the given strategy.

    Raises:
        NotImplementedError: when the strategy is not registered.
    """
    normalized = strategy_name.strip() if strategy_name else ""
    spec = STRATEGY_REGISTRY.get(normalized)
    if not spec:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise NotImplementedError(
            f"Strategy '{strategy_name}' is not implemented. "
            f"Available strategies: {available}"
        )

    return spec


def resolve_strategy(strategy_name: str) -> tuple[StrategySpec, StrategyRunner]:
    """
    Retrieve both the metadata and the normalized runner for a strategy.
    """
    spec = get_strategy_spec(strategy_name)

    def runner_wrapper(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        raw_df = spec.runner(stock_data_dict)
        return _normalize_strategy_output(raw_df, spec.name or "Unknown")

    return spec, runner_wrapper


def get_strategy_runner(strategy_name: str) -> StrategyRunner:
    """
    Backwards-compatible helper returning only the normalized runner.
    """
    _, runner = resolve_strategy(strategy_name)
    return runner
