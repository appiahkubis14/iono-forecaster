"""
dataset/iono_dataset.py — IonoForecaster
PyTorch Dataset for spatio-temporal GNSS scintillation sequences.

Each sample is a sliding window of shape (T, N, F):
  T = input_timesteps (12 = 6 hours at 30-min resolution)
  N = number of GNSS stations (nodes)
  F = number of features

Target: (N, 1) S4 index at t+1.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.utils import get_logger

logger = get_logger("dataset")

# Feature columns used in the model
# Order matters — must be consistent between train and inference
FEATURE_COLS = [
    # GNSS / scintillation
    "s4", "sigma_phi", "roti", "tec",
    "rot", "roti_computed",
    # Solar wind
    "bz", "v_sw", "n_p", "t_p",
    "epsilon", "newell", "ey", "beta_sw",
    # Geomagnetic
    "kp", "dst",
    # Aurora
    "hp_north", "hp_south",
    # Time harmonics
    "sin_hour_1", "cos_hour_1", "sin_hour_2", "cos_hour_2",
    "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
    "post_sunset", "lt_hour",
    # Storm
    "hours_since_storm_onset", "storm_phase", "cumulative_dst", "storm_intensity",
]
TARGET_COL = "s4"


class IonoDataset(Dataset):
    """
    Sliding-window spatio-temporal dataset for ST-GNN training.

    Parameters
    ----------
    data : np.ndarray
        Shape (T_total, N, F) — full feature array, normalised.
    targets : np.ndarray
        Shape (T_total, N) — S4 values (may be same as data[:, :, s4_idx]).
    input_steps : int
        Number of input timesteps per sample.
    forecast_steps : int
        How many steps ahead to predict (1 for one-step, 12 for 6-hour roll).
    mode : str
        'train', 'val', 'test'.
    station_mask : np.ndarray, optional
        Boolean mask of shape (N,) — if provided, mask out held-out stations
        in the target (set to NaN).
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        input_steps: int = 12,
        forecast_steps: int = 1,
        mode: str = "train",
        station_mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.data = torch.from_numpy(data).float()          # (T, N, F)
        self.targets = torch.from_numpy(targets).float()    # (T, N)
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.mode = mode
        self.station_mask = station_mask

        T = self.data.shape[0]
        self.valid_indices = list(range(input_steps, T - forecast_steps + 1))
        logger.info(
            "IonoDataset [%s]: %d samples | %d timesteps | %d nodes | %d features",
            mode, len(self.valid_indices), T, data.shape[1], data.shape[2],
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            x       : (T, N, F) input window
            y       : (N, 1) target S4 (next step)
            mask    : (N,) valid-station mask (True = include in loss)
        """
        t = self.valid_indices[idx]
        x = self.data[t - self.input_steps: t]              # (T, N, F)
        y = self.targets[t: t + self.forecast_steps]        # (steps, N)
        # For single-step: take first step only
        y = y[0].unsqueeze(-1)                              # (N, 1)

        # NaN mask
        mask = ~torch.isnan(y.squeeze(-1))                  # (N,)
        if self.station_mask is not None:
            mask = mask & torch.from_numpy(self.station_mask)

        y = torch.nan_to_num(y, nan=0.0)
        x = torch.nan_to_num(x, nan=0.0)

        return {"x": x, "y": y, "mask": mask}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    cfg: Dict,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> Tuple[IonoDataset, IonoDataset, IonoDataset]:
    """
    Split feature array into train/val/test IonoDatasets using temporal splits.

    Parameters
    ----------
    feature_array : np.ndarray
        (T, N, F) normalised features.
    target_array : np.ndarray
        (T, N) S4 targets.
    cfg : dict
    timestamps : pd.DatetimeIndex, optional
        Timestamps corresponding to T dimension.

    Returns
    -------
    train_ds, val_ds, test_ds : IonoDataset
    """
    input_steps = cfg["model"].get("input_timesteps", 12)
    resolution = cfg["temporal"].get("resolution_minutes", 30)

    T = feature_array.shape[0]

    if timestamps is not None:
        train_end = pd.Timestamp(cfg["temporal"]["train_end"])
        val_end = pd.Timestamp(cfg["temporal"]["val_end"])

        train_mask = timestamps <= train_end
        val_mask = (timestamps > train_end) & (timestamps <= val_end)
        test_mask = timestamps > val_end

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
    else:
        # Default 70/15/15 split
        n_train = int(T * 0.70)
        n_val = int(T * 0.15)
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, T)

    def _slice(arr, idx):
        return arr[idx]

    train_ds = IonoDataset(
        _slice(feature_array, train_idx),
        _slice(target_array, train_idx),
        input_steps=input_steps,
        mode="train",
    )
    val_ds = IonoDataset(
        _slice(feature_array, val_idx),
        _slice(target_array, val_idx),
        input_steps=input_steps,
        mode="val",
    )
    test_ds = IonoDataset(
        _slice(feature_array, test_idx),
        _slice(target_array, test_idx),
        input_steps=input_steps,
        mode="test",
    )
    return train_ds, val_ds, test_ds
