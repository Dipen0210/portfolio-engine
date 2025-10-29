"""
signals/filters.py
-------------------
Filters for initial stock selection based on sector and market cap.
MVP version — uses only company.csv (no OHLCV or fundamentals yet).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

RISK_PERCENTILE_WINDOWS = {
    # Larger caps -> lower risk; smaller caps -> higher risk
    "low": (0.66, 1.0),
    "medium": (0.33, 0.66),
    "high": (0.0, 0.33),
}


def _canonicalize_risk_level(risk_level: str) -> str:
    if not risk_level:
        return "medium"
    risk_level = risk_level.lower()
    return risk_level if risk_level in RISK_PERCENTILE_WINDOWS else "medium"


def _percentile_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile ranks for market caps inside a sector.
    """
    return series.rank(method="max", pct=True)


def filter_by_sector(df: pd.DataFrame, selected_sector: str):
    """
    Filter stocks by selected sector.
    """
    if not selected_sector or selected_sector.lower() == "all":
        return df
    return df[df["sector"].str.lower() == selected_sector.lower()]


def filter_by_market_cap(
    df: pd.DataFrame, min_cap: float | None = None, max_cap: float | None = None
):
    """
    Filter by market capitalization range (optional).
    Args:
        min_cap: Minimum market cap (in USD)
        max_cap: Maximum market cap (in USD)
    """
    df = df.copy()
    if min_cap is not None:
        df = df[df["market cap"] >= min_cap]
    if max_cap is not None:
        df = df[df["market cap"] <= max_cap]
    return df


def select_top_n(df: pd.DataFrame, n: int = 5):
    """
    Select top N stocks by market cap.
    """
    df_sorted = df.sort_values(by="market cap", ascending=False)
    return df_sorted.head(n)


def apply_filters(company_df, sector=None, n=5, min_cap=None, max_cap=None):
    """
    Apply sector and market cap filters, then select top N.
    Returns filtered DataFrame and list of tickers.
    """
    filtered = filter_by_sector(company_df, sector)
    filtered = filter_by_market_cap(filtered, min_cap, max_cap)
    selected = select_top_n(filtered, n)

    tickers = selected["ticker"].tolist()
    return selected, tickers


def _resolve_sectors(df: pd.DataFrame, sectors: Iterable[str] | None) -> list[str]:
    if not sectors:
        return sorted(df["sector"].dropna().unique().tolist())

    available = df["sector"].dropna().unique().tolist()
    lower_map = {sector.lower(): sector for sector in available}

    resolved = [lower_map[s.lower()] for s in sectors if s.lower() in lower_map]
    return resolved or sorted(available)


def filter_universe_by_risk_and_sector(
    company_df: pd.DataFrame,
    sectors: list[str] | None,
    risk_level: str,
    top_n_per_sector: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Apply sector filter, risk-based market cap percentile bands, and pick the top N.

    Returns a dict mapping sector -> DataFrame of the filtered top N rows.
    """
    canonical_risk = _canonicalize_risk_level(risk_level)
    lower_pct, upper_pct = RISK_PERCENTILE_WINDOWS[canonical_risk]
    available_sectors = _resolve_sectors(company_df, sectors)

    sector_slices: dict[str, pd.DataFrame] = {}
    for sector in available_sectors:
        sector_df = filter_by_sector(company_df, sector)
        if sector_df.empty:
            continue

        working = sector_df.dropna(subset=["market cap"]).copy()
        if working.empty:
            continue

        working["market_cap_percentile"] = _percentile_rank(working["market cap"])

        # Apply percentile window inclusive of the upper bound for continuity.
        mask = (working["market_cap_percentile"] > lower_pct) & (
            working["market_cap_percentile"] <= upper_pct
        )
        filtered = working.loc[mask]
        if filtered.empty:
            continue

        top_selection = (
            filtered.sort_values(by="market cap", ascending=False)
            .head(top_n_per_sector)
            .reset_index(drop=True)
        )
        if top_selection.empty:
            continue

        # Preserve canonical sector name from the dataset when possible
        canonical_sector = top_selection["sector"].iloc[0]
        top_selection = top_selection.copy()
        top_selection["sector"] = canonical_sector
        sector_slices[canonical_sector] = top_selection

    return sector_slices
