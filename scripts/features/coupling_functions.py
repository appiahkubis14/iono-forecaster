"""
features/coupling_functions.py — IonoForecaster
Compute solar wind–magnetosphere coupling functions.

Key coupling functions:
  ε = v * Bz (simplified Akasofu parameter proxy, nT km/s)
  dΦ/dt ≈ v * Bt * sin²(θ/2)  (Newell et al. 2007 reconnection rate)

These drive prompt penetration electric fields (PPEFs) that modulate
equatorial ionospheric irregularities.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("features.coupling_functions")


def compute_coupling_functions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add solar wind coupling function columns to DataFrame.

    Required input columns: bz, v_sw, n_p (optional: bt)

    New columns:
        epsilon   : simplified Akasofu coupling (v * |Bz| when Bz < 0)
        newell    : Newell reconnection rate proxy
        ey        : dawn-dusk electric field (Ey = v * Bz, mV/m)
        beta_sw   : solar wind dynamic pressure proxy (n_p * v_sw²)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    bz = df.get("bz", pd.Series(np.zeros(len(df)), index=df.index))
    v_sw = df.get("v_sw", pd.Series(np.full(len(df), 400.0), index=df.index))
    n_p = df.get("n_p", pd.Series(np.full(len(df), 5.0), index=df.index))
    bt = df.get("bt", bz.abs())

    # ε: southward Bz coupling (zero when Bz ≥ 0)
    bz_south = np.where(bz < 0, np.abs(bz), 0.0)
    df["epsilon"] = (v_sw * bz_south).astype(np.float32)

    # Newell reconnection rate: v^(4/3) * Bt^(2/3) * sin^(8/3)(θ/2)
    # θ = IMF clock angle; for southward Bz, sin²(θ/2) ≈ 1
    clock_angle = np.arctan2(bz.abs(), bt.replace(0, 1e-9))
    sin8_3 = np.sin(clock_angle / 2) ** (8 / 3)
    df["newell"] = (
        v_sw.abs() ** (4 / 3) * bt.abs() ** (2 / 3) * sin8_3
    ).astype(np.float32)

    # Ey convection electric field (mV/m): negative = anti-sunward = geoeffective
    df["ey"] = (-v_sw * bz / 1000.0).astype(np.float32)  # km/s * nT → mV/m

    # Solar wind dynamic pressure proxy (arbitrary units)
    df["beta_sw"] = (n_p * v_sw ** 2).astype(np.float32)

    return df
