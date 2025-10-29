"""
hardfilters/data_utils.py
-------------------------
Utility helpers to load and clean the company universe dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# Columns that we keep after cleaning the raw dataset
OUTPUT_COLUMNS = ("ticker", "company name", "market cap", "sector")


def _coerce_market_cap(series: pd.Series) -> pd.Series:
    """Convert a market cap column to numeric and drop non-positive entries."""
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.where(coerced > 0)


def clean_company_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw company universe dataset.

    Steps:
    * Keep the relevant columns.
    * Drop rows with missing tickers, sectors, or market caps.
    * Standardise sector strings.
    """
    missing_columns = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    cleaned = df.loc[:, OUTPUT_COLUMNS].copy()
    cleaned["ticker"] = cleaned["ticker"].str.strip().str.upper()
    cleaned["sector"] = cleaned["sector"].astype(str).str.strip()
    cleaned["market cap"] = _coerce_market_cap(cleaned["market cap"])

    cleaned = cleaned.dropna(subset=["ticker", "sector", "market cap"])
    cleaned = cleaned.drop_duplicates(subset=["ticker"])

    # Re-order columns for readability
    cleaned = cleaned.loc[:, ["ticker", "company name", "sector", "market cap"]]
    cleaned = cleaned.sort_values(by="market cap", ascending=False).reset_index(drop=True)
    return cleaned


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Company universe file not found at {path}")
    return pd.read_csv(path)


def load_company_universe(
    raw_path: str | Path = "data/raw/companies.csv",
    processed_path: str | Path = "data/processed/cleaned_companies.csv",
    *,
    prefer_processed: bool = True,
    persist_processed: bool = True,
) -> pd.DataFrame:
    """
    Load the company universe, cleaning the raw CSV when needed.

    By default we prefer a cached processed file to save time during repeated
    Streamlit runs, but fall back to cleaning the raw dataset when necessary.
    """
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    if prefer_processed and processed_path.exists():
        processed = _read_csv(processed_path)
        # Basic validation to ensure the processed file is usable.
        if set(OUTPUT_COLUMNS).issubset(processed.columns):
            return processed

    raw_df = _read_csv(raw_path)
    cleaned = clean_company_dataframe(raw_df)

    if persist_processed:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(processed_path, index=False)

    return cleaned


def ensure_sectors_exist(
    df: pd.DataFrame, sectors: Iterable[str] | None
) -> list[str]:
    """
    Validate user sector selections against the dataset.

    Returns the list of sectors to use (unique, case-sensitive as in the dataset).
    """
    available_sectors = sorted(df["sector"].dropna().unique().tolist())
    if not sectors:
        return available_sectors

    normalized_map = {sector.lower(): sector for sector in available_sectors}
    matched = []
    for sector in sectors:
        lower = sector.lower()
        if lower in normalized_map:
            matched.append(normalized_map[lower])

    return matched or available_sectors
