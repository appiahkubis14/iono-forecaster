"""
preprocess/gap_fill.py — IonoForecaster
Linear interpolation gap-filling for short gaps in time series.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("preprocess.gap_fill")


def fill_gaps(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    max_gap_minutes: int = 120,
    method: str = "linear",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fill short gaps in time series using interpolation.

    Gaps longer than max_gap_minutes are left as NaN (not interpolated).

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    value_cols : list of str
    max_gap_minutes : int
        Maximum gap length to interpolate.
    method : str
        Interpolation method: 'linear', 'time', 'cubic'.
    group_col : str, optional
        If provided, fill gaps within each group (e.g. station_id).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    def _fill_group(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values(time_col).set_index(time_col)
        for col in value_cols:
            if col not in grp.columns:
                continue
            series = grp[col].copy()
            # Identify gaps
            is_nan = series.isna()
            if not is_nan.any():
                continue
            # Only interpolate short gaps
            limit = max_gap_minutes // 30  # assuming 30-min resolution
            series_filled = series.interpolate(method=method, limit=limit, limit_direction="both")
            grp[col] = series_filled
        return grp.reset_index()

    if group_col and group_col in df.columns:
        result = df.groupby(group_col, group_keys=False).apply(_fill_group)
    else:
        result = _fill_group(df)

    return result.reset_index(drop=True)
