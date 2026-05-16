"""
features/storm_time.py — IonoForecaster
Compute storm-related temporal features.

Features:
    hours_since_storm_onset : time elapsed since last geomagnetic storm (Dst < -50 nT)
    storm_phase             : 0=quiet, 1=sudden commencement, 2=main, 3=recovery
    cumulative_dst          : 24-hour cumulative Dst (energy proxy)

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("features.storm_time")

# Storm onset threshold (nT)
STORM_DST_THRESHOLD = -50.0
# Sudden commencement: rapid increase in H-component (proxy: Dst change > 20 nT/hour)
SSC_DST_CHANGE_THRESHOLD = 20.0


def add_storm_features(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    dst_col: str = "dst",
    kp_col: str = "kp",
) -> pd.DataFrame:
    """
    Add storm-phase timing features.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by time_col and contain dst_col.
    time_col, dst_col, kp_col : str

    Returns
    -------
    pd.DataFrame with new storm feature columns.
    """
    df = df.copy().sort_values(time_col).reset_index(drop=True)
    n = len(df)

    if dst_col not in df.columns:
        logger.warning("'%s' column not found; setting storm features to 0.", dst_col)
        df["hours_since_storm_onset"] = 0.0
        df["storm_phase"] = 0
        df["cumulative_dst"] = 0.0
        df["storm_intensity"] = 0.0
        return df

    dst = df[dst_col].values
    ts = pd.to_datetime(df[time_col])

    # Identify storm onsets: Dst crosses below threshold
    in_storm = dst <= STORM_DST_THRESHOLD
    onset_idx = np.where(np.diff(in_storm.astype(int)) == 1)[0] + 1
    onset_times = ts.iloc[onset_idx].values if len(onset_idx) > 0 else np.array([])

    # hours_since_storm_onset
    hours_since = np.full(n, np.nan)
    for i, t in enumerate(ts):
        past_onsets = onset_times[onset_times <= np.datetime64(t)]
        if len(past_onsets) > 0:
            dt = (np.datetime64(t) - past_onsets[-1]) / np.timedelta64(1, "h")
            hours_since[i] = float(dt)

    # Fill pre-first-storm with large value (72h = 3 days of "quiet")
    hours_since = np.where(np.isnan(hours_since), 72.0, hours_since)
    df["hours_since_storm_onset"] = hours_since.astype(np.float32)

    # Storm phase (simplified)
    # 0=quiet, 1=SSC, 2=main phase, 3=recovery
    dst_change = np.gradient(dst)
    phase = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if dst[i] > STORM_DST_THRESHOLD:
            if dst_change[i] > SSC_DST_CHANGE_THRESHOLD:
                phase[i] = 1  # sudden commencement
            else:
                phase[i] = 0  # quiet
        elif dst_change[i] < 0:
            phase[i] = 2  # main phase
        else:
            phase[i] = 3  # recovery
    df["storm_phase"] = phase

    # 24-hour cumulative Dst (energy input proxy)
    window_24h = max(1, 24 * 60 // 30)  # 30-min resolution → 48 steps
    df["cumulative_dst"] = (
        pd.Series(dst).rolling(window=window_24h, min_periods=1).mean().values.astype(np.float32)
    )

    # Storm intensity classification (0=none, 1=moderate, 2=intense, 3=super)
    intensity = np.zeros(n, dtype=np.int32)
    intensity = np.where(dst < -30, 1, intensity)
    intensity = np.where(dst < -100, 2, intensity)
    intensity = np.where(dst < -200, 3, intensity)
    df["storm_intensity"] = intensity.astype(np.float32)

    return df
