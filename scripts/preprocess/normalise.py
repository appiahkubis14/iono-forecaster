"""
preprocess/normalise.py — IonoForecaster
Feature normalisation using MinMaxScaler (per feature, globally fitted).

Scalers are saved as JSON so inference-time normalisation is consistent.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("preprocess.normalise")

# Physical bounds for each feature (used as fallback range)
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "s4":        (0.0, 1.0),
    "sigma_phi": (0.0, 3.14159),
    "roti":      (0.0, 5.0),
    "tec":       (0.0, 100.0),
    "bz":        (-50.0, 20.0),
    "v_sw":      (200.0, 900.0),
    "n_p":       (0.0, 50.0),
    "t_p":       (0.0, 1e6),
    "kp":        (0.0, 9.0),
    "dst":       (-250.0, 50.0),
    "hp_north":  (0.0, 300.0),
    "hp_south":  (0.0, 300.0),
    "ne":        (0.0, 1e7),
}


class IonoScaler:
    """
    Fit-transform normaliser that stores min/max per feature.

    Parameters
    ----------
    feature_range : tuple
        Target range (default 0–1).
    use_physical_bounds : bool
        If True, use known physical bounds instead of data-driven bounds.
        Recommended for small datasets to avoid overfitting normalisation.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        use_physical_bounds: bool = True,
    ) -> None:
        self.feature_range = feature_range
        self.use_physical_bounds = use_physical_bounds
        self._params: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "IonoScaler":
        """Compute and store normalisation parameters from training data."""
        for col in columns:
            if col not in df.columns:
                continue
            if self.use_physical_bounds and col in FEATURE_BOUNDS:
                lo, hi = FEATURE_BOUNDS[col]
            else:
                lo = float(df[col].min())
                hi = float(df[col].max())

            if hi == lo:
                hi = lo + 1.0  # prevent division by zero

            self._params[col] = {"min": lo, "max": hi}
        logger.info("Scaler fitted for %d features.", len(self._params))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalisation to fitted columns."""
        df = df.copy()
        lo_t, hi_t = self.feature_range
        for col, params in self._params.items():
            if col not in df.columns:
                continue
            lo, hi = params["min"], params["max"]
            df[col] = lo_t + (df[col] - lo) / (hi - lo) * (hi_t - lo_t)
            df[col] = df[col].clip(lo_t, hi_t)
        return df

    def inverse_transform(
        self, arr: np.ndarray, feature: str
    ) -> np.ndarray:
        """
        Reverse normalisation for a single feature.

        Parameters
        ----------
        arr : np.ndarray
            Normalised values.
        feature : str

        Returns
        -------
        np.ndarray
            Values in original units.
        """
        if feature not in self._params:
            raise KeyError(f"Feature '{feature}' not fitted.")
        lo_t, hi_t = self.feature_range
        lo, hi = self._params[feature]["min"], self._params[feature]["max"]
        return lo + (arr - lo_t) / (hi_t - lo_t) * (hi - lo)

    def save(self, path: str | Path) -> None:
        """Persist scaler parameters as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"feature_range": list(self.feature_range), "params": self._params}, fh, indent=2)
        logger.info("Scaler saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "IonoScaler":
        """Load scaler from JSON file."""
        with open(path) as fh:
            data = json.load(fh)
        scaler = cls(feature_range=tuple(data["feature_range"]))
        scaler._params = data["params"]
        logger.info("Scaler loaded ← %s", path)
        return scaler

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit then transform."""
        return self.fit(df, columns).transform(df)
