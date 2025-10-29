# optimization/constraints.py

import numpy as np


def weight_sum_constraint(weights):
    """Constraint: all weights must sum to 1 (fully invested portfolio)."""
    return np.sum(weights) - 1.0


def non_negative_constraint(weights):
    """Constraint: no short-selling allowed (weights >= 0)."""
    return weights


def max_weight_constraint(weights, max_weight=0.3):
    """Constraint: individual stock cannot exceed given weight."""
    return max_weight - np.max(weights)


def sector_weight_constraint(weights, sector_map, max_sector_weight=0.4):
    """
    Constraint: limit total weight in any single sector.

    Parameters
    ----------
    weights : np.array
        Current portfolio weights.
    sector_map : list or array
        Sector label for each asset (aligned with weights).
    max_sector_weight : float
        Maximum allowed sector exposure (e.g., 0.4 for 40%).
    """
    unique_sectors = set(sector_map)
    max_exposure = max(sum(w for w, s in zip(weights, sector_map) if s == sector)
                       for sector in unique_sectors)
    return max_sector_weight - max_exposure
