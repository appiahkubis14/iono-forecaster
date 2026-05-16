"""
preprocess/resample.py — IonoForecaster
Resample all data sources to a common 30-minute temporal grid.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("preprocess.resample")


def resample_gnss(df: pd.DataFrame, resolution_minutes: int = 30) -> pd.DataFrame:
    """
    Resample GNSS station data to target resolution.

    Uses max S4 (worst scintillation) per window; mean for other variables.

    Parameters
    ----------
    df : pd.DataFrame
        Must have: timestamp, station_id, s4, sigma_phi, roti, tec
    resolution_minutes : int

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    rule = f"{resolution_minutes}min"
    groups = df.groupby("station_id")
    resampled_list = []

    for sid, grp in groups:
        rs = grp.resample(rule, label="left", closed="left")
        agg = rs.agg({
            "s4":       "max",
            "sigma_phi": "max",
            "roti":     "max",
            "tec":      "mean",
            "lat":      "first",
            "lon":      "first",
        })
        agg["station_id"] = sid
        resampled_list.append(agg)

    if not resampled_list:
        return df.reset_index()

    combined = pd.concat(resampled_list)
    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    return combined.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def resample_gim(df: pd.DataFrame, resolution_minutes: int = 30) -> pd.DataFrame:
    """
    Resample GIM TEC maps to target resolution by linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have: epoch, lat, lon, tec
    resolution_minutes : int

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["epoch"] = pd.to_datetime(df["epoch"])

    lats = df["lat"].unique()
    lons = df["lon"].unique()

    target_times = pd.date_range(
        df["epoch"].min(),
        df["epoch"].max(),
        freq=f"{resolution_minutes}min",
    )

    records = []
    for lat in lats:
        for lon in lons:
            subset = df[(df["lat"] == lat) & (df["lon"] == lon)].copy()
            subset = subset.set_index("epoch").sort_index()
            subset = subset["tec"].reindex(
                subset.index.union(target_times)
            ).interpolate("time").reindex(target_times)

            for ts, tec in subset.items():
                records.append({"epoch": ts, "lat": lat, "lon": lon, "tec": tec})

    result = pd.DataFrame(records)
    return result.sort_values(["epoch", "lat", "lon"]).reset_index(drop=True)


def resample_scalar(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    resolution_minutes: int = 30,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Generic resampler for scalar time-series (solar wind, geomagnetic, aurora).

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    value_cols : list of str
    resolution_minutes : int
    agg : str
        Aggregation method: 'mean', 'max', 'min'.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)[value_cols].sort_index()
    rs = df.resample(f"{resolution_minutes}min", label="left", closed="left")
    result = getattr(rs, agg)().reset_index()
    return result
