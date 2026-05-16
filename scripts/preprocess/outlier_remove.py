"""
preprocess/outlier_remove.py — IonoForecaster
IQR-based outlier detection and removal for GNSS/solar wind data.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("preprocess.outlier_remove")


def remove_outliers_iqr(
    df: pd.DataFrame,
    value_cols: List[str],
    iqr_threshold: float = 3.0,
    group_col: Optional[str] = None,
    replace_with: str = "nan",
) -> pd.DataFrame:
    """
    Detect and remove outliers using the IQR method.

    Outliers are defined as values outside:
        [Q1 - threshold * IQR, Q3 + threshold * IQR]

    Parameters
    ----------
    df : pd.DataFrame
    value_cols : list of str
    iqr_threshold : float
        Multiplier for IQR bounds (typically 1.5 normal, 3.0 conservative).
    group_col : str, optional
        Apply IQR per group.
    replace_with : str
        'nan' to replace with NaN, 'clip' to clip to bounds.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    def _process(grp: pd.DataFrame) -> pd.DataFrame:
        for col in value_cols:
            if col not in grp.columns:
                continue
            series = grp[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_threshold * iqr
            upper = q3 + iqr_threshold * iqr

            n_before = series.notna().sum()
            if replace_with == "clip":
                grp[col] = series.clip(lower, upper)
            else:
                mask = (series < lower) | (series > upper)
                n_outliers = mask.sum()
                if n_outliers > 0:
                    logger.debug(
                        "Column %s: %d outliers removed (%.1f%% of values)",
                        col, n_outliers, 100 * n_outliers / max(n_before, 1),
                    )
                grp.loc[mask, col] = np.nan
        return grp

    if group_col and group_col in df.columns:
        return df.groupby(group_col, group_keys=False).apply(_process).reset_index(drop=True)
    return _process(df)
