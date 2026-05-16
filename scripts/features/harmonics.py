"""
features/harmonics.py — IonoForecaster
Generate cyclical (sin/cos) time encodings for diurnal and seasonal patterns.

Cyclical encoding avoids discontinuities at midnight/year-end.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("features.harmonics")


def add_time_harmonics(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    lon_col: str = "lon",
    n_diurnal: int = 2,
    n_seasonal: int = 2,
) -> pd.DataFrame:
    """
    Add diurnal and seasonal harmonic features.

    Features added:
        sin_hour_N, cos_hour_N  : N-th Fourier component of local time
        sin_doy_N, cos_doy_N    : N-th Fourier component of day of year
        lt_hour                 : local solar time (hours)

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    lon_col : str
        Longitude column for local time calculation.
    n_diurnal : int
        Number of diurnal harmonics (1 = fundamental, 2 = + semi-diurnal).
    n_seasonal : int
        Number of seasonal harmonics.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    ts = pd.to_datetime(df[time_col])

    # Local solar time
    lon = df[lon_col] if lon_col in df.columns else pd.Series(np.zeros(len(df)))
    lt_offset_h = lon / 15.0
    lt_hour = (ts.dt.hour + ts.dt.minute / 60.0 + lt_offset_h) % 24
    df["lt_hour"] = lt_hour.astype(np.float32)

    # Diurnal harmonics
    for k in range(1, n_diurnal + 1):
        angle = 2 * np.pi * k * lt_hour / 24.0
        df[f"sin_hour_{k}"] = np.sin(angle).astype(np.float32)
        df[f"cos_hour_{k}"] = np.cos(angle).astype(np.float32)

    # Seasonal harmonics
    doy = ts.dt.day_of_year
    for k in range(1, n_seasonal + 1):
        angle = 2 * np.pi * k * doy / 365.25
        df[f"sin_doy_{k}"] = np.sin(angle).astype(np.float32)
        df[f"cos_doy_{k}"] = np.cos(angle).astype(np.float32)

    # Post-sunset flag (18–22 LT) — prime scintillation window
    df["post_sunset"] = ((lt_hour >= 18) & (lt_hour <= 22)).astype(np.float32)

    return df
