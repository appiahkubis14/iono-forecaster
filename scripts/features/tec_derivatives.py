"""
features/tec_derivatives.py — IonoForecaster
Compute ROT (Rate of TEC change) and ROTI (ROT Index) from TEC time series.

ROT  = ΔTEC / Δt  (TECU/min)
ROTI = std(ROT)   over rolling window (proxy for scintillation)

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("features.tec_derivatives")


def compute_rot(
    df: pd.DataFrame,
    tec_col: str = "tec",
    time_col: str = "timestamp",
    group_col: Optional[str] = "station_id",
) -> pd.DataFrame:
    """
    Compute Rate of TEC change (ROT) per station.

    ROT = (TEC[t] - TEC[t-1]) / Δt_minutes  (TECU/min)

    Parameters
    ----------
    df : pd.DataFrame
    tec_col, time_col, group_col : str

    Returns
    -------
    pd.DataFrame with new column 'rot'
    """
    df = df.copy().sort_values([group_col, time_col] if group_col else [time_col])

    def _rot(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values(time_col)
        dt_min = grp[time_col].diff().dt.total_seconds() / 60.0
        grp["rot"] = grp[tec_col].diff() / dt_min
        return grp

    if group_col and group_col in df.columns:
        return df.groupby(group_col, group_keys=False).apply(_rot).reset_index(drop=True)
    return _rot(df)


def compute_roti(
    df: pd.DataFrame,
    window_steps: int = 12,
    rot_col: str = "rot",
    group_col: Optional[str] = "station_id",
) -> pd.DataFrame:
    """
    Compute ROTI (Rate of TEC Index) as rolling std of ROT.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'rot' column (compute with compute_rot first).
    window_steps : int
        Rolling window length in timesteps.
    rot_col : str
    group_col : str

    Returns
    -------
    pd.DataFrame with new column 'roti_computed'
    """
    df = df.copy()

    def _roti(grp: pd.DataFrame) -> pd.DataFrame:
        grp["roti_computed"] = (
            grp[rot_col]
            .rolling(window=window_steps, min_periods=max(2, window_steps // 4))
            .std()
        )
        return grp

    if group_col and group_col in df.columns:
        return df.groupby(group_col, group_keys=False).apply(_roti).reset_index(drop=True)
    return _roti(df)
