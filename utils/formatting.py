from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def truncate_value(value, decimals: int = 4):
    """
    Truncate a scalar numeric to the specified decimal precision without rounding up.
    Non-numeric inputs are returned unchanged.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(numeric):
        return numeric
    factor = 10 ** decimals
    return math.trunc(numeric * factor) / factor


def truncate_series(series: pd.Series, decimals: int = 4) -> pd.Series:
    if series is None or series.empty:
        return series
    factor = 10 ** decimals
    numeric = pd.to_numeric(series, errors="coerce")
    truncated = np.trunc(numeric.to_numpy(dtype=float) * factor) / factor
    result = series.copy()
    result.loc[:] = truncated
    result[numeric.isna()] = np.nan
    return result


def truncate_dataframe(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if not len(numeric_cols):
        return df
    result = df.copy()
    factor = 10 ** decimals
    result[numeric_cols] = np.trunc(result[numeric_cols].astype(float) * factor) / factor
    return result


def format_currency(value, decimals: int = 4) -> str:
    truncated = truncate_value(value, decimals)
    if truncated is None:
        return "N/A"
    return f"${truncated:,.{decimals}f}"


def format_percentage(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    truncated = truncate_value(float(value) * 100.0, decimals)
    if truncated is None:
        return "N/A"
    return f"{truncated:.{decimals}f}%"


def normalize_and_truncate_weights(
    df: pd.DataFrame,
    weight_col: str = "Weight",
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Normalize weights to sum to 1.0 and truncate to the specified precision.
    """
    if df is None or df.empty or weight_col not in df.columns:
        return df
    working = df.copy()
    working[weight_col] = pd.to_numeric(working[weight_col], errors="coerce")
    working = working.dropna(subset=[weight_col])
    if working.empty:
        return working
    weights = working[weight_col].to_numpy(dtype=float)
    weights[weights < 0] = 0.0
    total = weights.sum()
    if total <= 0:
        working[weight_col] = weights
        return working

    normalized = weights / total
    factor = 10 ** decimals
    scaled = normalized * factor
    truncated_units = np.floor(scaled + 1e-12).astype(int)
    remainder_units = int(max(0, math.floor(factor - truncated_units.sum() + 1e-9)))
    if remainder_units:
        fractional = scaled - truncated_units
        order = np.argsort(-fractional)
        for idx in order[:remainder_units]:
            truncated_units[idx] += 1
    working[weight_col] = truncated_units / factor
    return working


def truncate_numeric_iterable(values: Iterable[float], decimals: int = 4) -> list[float]:
    factor = 10 ** decimals
    return [math.trunc(float(val) * factor) / factor for val in values]
