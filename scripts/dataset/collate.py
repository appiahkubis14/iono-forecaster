"""
dataset/collate.py — IonoForecaster
Custom collate function for batching spatio-temporal graph samples.

PyTorch DataLoader uses this to assemble individual samples from
IonoDataset into a mini-batch tensor, handling variable-length masks.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Dict, List

import torch


def iono_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of IonoDataset samples into a batched dict.

    Each sample has keys: x (T, N, F), y (N, 1), mask (N,).

    Parameters
    ----------
    batch : list of dict

    Returns
    -------
    dict
        Keys: x (B, T, N, F), y (B, N, 1), mask (B, N)
    """
    x = torch.stack([sample["x"] for sample in batch], dim=0)       # (B, T, N, F)
    y = torch.stack([sample["y"] for sample in batch], dim=0)       # (B, N, 1)
    mask = torch.stack([sample["mask"] for sample in batch], dim=0) # (B, N)
    return {"x": x, "y": y, "mask": mask}
