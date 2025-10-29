# risk_management/exposure_limits.py
import numpy as np
import pandas as pd


def enforce_weight_bounds(weights: pd.Series, min_w=0.0, max_w=0.2) -> pd.Series:
    """Clamp individual weights then renormalize."""
    w = weights.clip(lower=min_w, upper=max_w)
    s = w.sum()
    return w / s if s > 0 else w


def sector_exposure_caps(weights: pd.Series, meta_df: pd.DataFrame, sector_col="Sector", max_sector=0.5) -> pd.Series:
    """
    Ensure no sector exceeds max_sector. Proportional rescale if exceeded.
    meta_df must map index=Ticker to sector.
    """
    w = weights.copy()
    sectors = meta_df.loc[w.index, sector_col]
    sector_sums = w.groupby(sectors).sum()
    for sec, total in sector_sums.items():
        if total > max_sector:
            # scale down that sector block so it equals max_sector, redistribute surplus to others
            scale = max_sector / total
            mask = (sectors == sec)
            w.loc[mask] *= scale
            # renormalize others to absorb residual
            residual = 1.0 - w.sum()
            if residual > 0:
                w.loc[~mask] *= (1 + residual / w.loc[~mask].sum())
    # final normalize
    return w / w.sum()


def turnover_limit(old_weights: pd.Series, new_weights: pd.Series, max_turnover: float = 0.3) -> pd.Series:
    """
    Constrain L1 turnover: sum(|w_new - w_old|) <= max_turnover
    If exceeded, blend new weights back toward old weights to respect the limit.
    """
    old_w = old_weights.reindex(new_weights.index).fillna(0.0)
    delta = (new_weights - old_w).abs().sum()
    if delta <= max_turnover or delta == 0:
        return new_weights
    # Blend factor
    blend = max_turnover / delta
    w = old_w + blend * (new_weights - old_w)
    return w / w.sum()


def basic_sanity_checks(weights: pd.Series):
    assert np.isfinite(weights.values).all(), "Non-finite weights detected."
    assert abs(weights.sum() - 1.0) < 1e-6, "Weights do not sum to 1."
    assert (weights >= -1e-8).all(), "Negative weight found (long-only expected)."
