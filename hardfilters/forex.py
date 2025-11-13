"""
hardfilters/forex.py
--------------------
Helpers for working with the FX universe: loading base/quote currency options,
validating available pairs, and constructing tickers for downstream pricing
pipelines (e.g., yfinance `XXXYYY=X` symbols).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pandas as pd

FOREX_CSV_DEFAULT = Path("data/raw/forex.csv")


def _coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"CODE", "BASE_CURRENCY", "QUOTE_CURRENCY"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Forex universe missing required columns: {sorted(missing)}")
    normalized = df.copy()
    for col in required_cols:
        normalized[col] = normalized[col].astype(str).str.strip().str.upper()
    return normalized


@lru_cache(maxsize=1)
def load_forex_universe(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Read the forex universe CSV (defaults to data/raw/forex.csv) and normalize case.
    """
    path = Path(csv_path) if csv_path else FOREX_CSV_DEFAULT
    if not path.exists():
        raise FileNotFoundError(f"Forex universe file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    return _coerce_dataframe(df)


def available_base_currencies(data: pd.DataFrame | None = None) -> list[str]:
    """
    Return the sorted list of unique base currencies.
    """
    df = _coerce_dataframe(data if data is not None else load_forex_universe())
    return sorted(df["BASE_CURRENCY"].unique())


def available_quote_currencies(
    base_currency: str,
    data: pd.DataFrame | None = None,
) -> list[str]:
    """
    Return the sorted list of quote currencies compatible with the chosen base.
    """
    if not base_currency:
        return []
    df = _coerce_dataframe(data if data is not None else load_forex_universe())
    base_mask = df["BASE_CURRENCY"] == base_currency.strip().upper()
    return sorted(df.loc[base_mask, "QUOTE_CURRENCY"].unique())


def resolve_pair_code(
    base_currency: str,
    quote_currency: str,
    data: pd.DataFrame | None = None,
) -> str:
    """
    Resolve and return the canonical pair code (e.g., EURUSD) for the provided
    base/quote combination, raising ValueError if the pair is unsupported.
    """
    if not base_currency or not quote_currency:
        raise ValueError("Both base and quote currencies are required.")
    df = _coerce_dataframe(data if data is not None else load_forex_universe())
    mask = (
        (df["BASE_CURRENCY"] == base_currency.strip().upper())
        & (df["QUOTE_CURRENCY"] == quote_currency.strip().upper())
    )
    match = df.loc[mask, "CODE"].dropna()
    if match.empty:
        raise ValueError(f"Unsupported FX pair: {base_currency}/{quote_currency}")
    return match.iloc[0]


def to_yfinance_symbol(pair_code: str) -> str:
    """
    Convert a raw pair code (e.g., EURUSD) into the yfinance symbol (EURUSD=X).
    """
    if not pair_code:
        raise ValueError("Pair code is required to build a yfinance symbol.")
    return f"{pair_code.strip().upper()}=X"


def build_yfinance_symbol_from_inputs(
    base_currency: str,
    quote_currency: str,
    data: pd.DataFrame | None = None,
) -> str:
    """
    Helper wrapper to go from base/quote selections straight to the yfinance code.
    """
    pair_code = resolve_pair_code(base_currency, quote_currency, data=data)
    return to_yfinance_symbol(pair_code)


def list_pairs_for_bases(
    base_currencies: Sequence[str],
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of all FX pairs whose base currency is within the provided list.
    Includes a `YF_SYMBOL` column for convenience.
    """
    if not base_currencies:
        return pd.DataFrame(columns=["CODE", "BASE_CURRENCY", "QUOTE_CURRENCY", "YF_SYMBOL"])

    df = _coerce_dataframe(data if data is not None else load_forex_universe())
    bases = {b.strip().upper() for b in base_currencies if b}
    if not bases:
        return pd.DataFrame(columns=["CODE", "BASE_CURRENCY", "QUOTE_CURRENCY", "YF_SYMBOL"])

    subset = df[df["BASE_CURRENCY"].isin(bases)].drop_duplicates(subset=["CODE"]).copy()
    if subset.empty:
        return pd.DataFrame(columns=["CODE", "BASE_CURRENCY", "QUOTE_CURRENCY", "YF_SYMBOL"])

    subset["YF_SYMBOL"] = subset["CODE"].apply(to_yfinance_symbol)
    return subset.sort_values(["BASE_CURRENCY", "QUOTE_CURRENCY"]).reset_index(drop=True)
