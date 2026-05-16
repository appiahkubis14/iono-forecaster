"""
models/losses.py — IonoForecaster
Combined MSE + MAE loss for scintillation index regression.

loss = α * MSE + (1-α) * MAE

Justification:
  - MSE penalises large errors (critical for missed severe scintillation events)
  - MAE provides robustness to noise / outliers in training data
  - α = 0.7 weighted towards MSE (per spec)

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Weighted combination of MSE and MAE losses.

    Parameters
    ----------
    mse_weight : float
        Weight for MSE component (default 0.7).
    mae_weight : float
        Weight for MAE component (default 0.3).
    reduction : str
        'mean' or 'sum'.
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        mae_weight: float = 0.3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : torch.Tensor (...,)
        y_true : torch.Tensor (...,)
        mask : torch.Tensor, optional
            Boolean mask; True = valid (not NaN). If None all elements used.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if mask is not None:
            y_pred = y_pred[mask]
            y_true = y_true[mask]

        if y_pred.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=y_pred.device)

        mse = F.mse_loss(y_pred, y_true, reduction=self.reduction)
        mae = F.l1_loss(y_pred, y_true, reduction=self.reduction)
        return self.mse_weight * mse + self.mae_weight * mae


class FocalScintillationLoss(nn.Module):
    """
    Focal-weighted variant: up-weights severe scintillation events (S4 > 0.7).

    Useful when training data is imbalanced (rare extreme events).

    Parameters
    ----------
    gamma : float
        Focusing parameter (higher = more focus on hard examples).
    severe_threshold : float
        S4 threshold for severe events.
    severe_weight : float
        Extra weight for severe event timesteps.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        severe_threshold: float = 0.7,
        severe_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.severe_threshold = severe_threshold
        self.severe_weight = severe_weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        error = (y_pred - y_true).abs()
        # Focal weight: harder examples get higher weight
        focal_w = (1 + error) ** self.gamma
        # Severe event up-weighting
        severe_mask = (y_true >= self.severe_threshold).float()
        severity_w = 1.0 + (self.severe_weight - 1.0) * severe_mask

        loss = (focal_w * severity_w * error ** 2).mean()
        return loss
