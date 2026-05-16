"""
train/temporal_cv.py — IonoForecaster
3-fold temporal cross-validation (non-overlapping, chronological splits).

Unlike k-fold CV, temporal CV respects the time ordering:
  Fold 1: train 2018-2020, val 2021
  Fold 2: train 2018-2021, val 2022
  Fold 3: train 2018-2022, val 2023

This prevents data leakage and simulates real operational deployment.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from scripts.models.metrics import compute_regression_metrics, compute_event_metrics
from scripts.utils import get_logger, set_seed

logger = get_logger("train.temporal_cv")


def temporal_cv_splits(
    timestamps: pd.DatetimeIndex,
    n_folds: int = 3,
    val_months: int = 12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate temporal CV split indices (train, val pairs).

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
    n_folds : int
        Number of CV folds.
    val_months : int
        Validation period length in months.

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    T = len(timestamps)
    splits = []

    # Divide time axis into n_folds validation windows, expanding training set
    # e.g. 6 years of data, 3 folds, 12-month val windows
    total_months = (
        (timestamps[-1].year - timestamps[0].year) * 12
        + timestamps[-1].month - timestamps[0].month
    )
    step_months = max(val_months, total_months // (n_folds + 1))

    start_ts = timestamps[0]

    for fold in range(1, n_folds + 1):
        val_start = start_ts + pd.DateOffset(months=fold * step_months)
        val_end = val_start + pd.DateOffset(months=val_months)

        if val_end > timestamps[-1]:
            logger.warning("Fold %d val_end %s beyond data range; skipping.", fold, val_end)
            continue

        train_mask = timestamps < val_start
        val_mask = (timestamps >= val_start) & (timestamps < val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))
        logger.info(
            "Fold %d: train [%s → %s, %d steps] | val [%s → %s, %d steps]",
            fold,
            timestamps[train_idx[0]].strftime("%Y-%m-%d"),
            timestamps[train_idx[-1]].strftime("%Y-%m-%d"),
            len(train_idx),
            timestamps[val_idx[0]].strftime("%Y-%m-%d"),
            timestamps[val_idx[-1]].strftime("%Y-%m-%d"),
            len(val_idx),
        )

    return splits


def run_temporal_cv(
    cfg: Dict,
    feature_array: np.ndarray,
    target_array: np.ndarray,
    timestamps: pd.DatetimeIndex,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    adj: Optional[torch.Tensor] = None,
    n_folds: int = 3,
    val_months: int = 12,
) -> Dict:
    """
    Execute 3-fold temporal cross-validation.

    Parameters
    ----------
    cfg : dict
    feature_array : np.ndarray (T, N, F)
    target_array : np.ndarray (T, N)
    timestamps : pd.DatetimeIndex (T,)
    edge_index, edge_weight : torch.Tensor
    adj : torch.Tensor, optional
    n_folds : int
    val_months : int

    Returns
    -------
    dict
        Aggregated CV results: mean/std of all metrics across folds.
    """
    from scripts.models.st_gnn import build_model
    from scripts.dataset.iono_dataset import IonoDataset, FEATURE_COLS
    from scripts.train.train import Trainer

    splits = temporal_cv_splits(timestamps, n_folds=n_folds, val_months=val_months)
    if not splits:
        logger.error("No valid temporal CV splits generated.")
        return {}

    n_features = feature_array.shape[2]
    n_nodes = feature_array.shape[1]
    input_steps = cfg["model"].get("input_timesteps", 12)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info("=" * 60)
        logger.info("Temporal CV Fold %d/%d", fold_idx + 1, len(splits))
        logger.info("=" * 60)

        set_seed(cfg.get("training", {}).get("seed", 42) + fold_idx)

        train_ds = IonoDataset(
            feature_array[train_idx], target_array[train_idx],
            input_steps=input_steps, mode="train",
        )
        val_ds = IonoDataset(
            feature_array[val_idx], target_array[val_idx],
            input_steps=input_steps, mode="val",
        )

        # Build fresh model for each fold
        model = build_model(cfg, n_features=n_features, n_nodes=n_nodes)

        # Override epochs to a smaller value for CV (speed)
        cv_cfg = copy.deepcopy(cfg)
        cv_cfg["training"]["epochs"] = max(20, cfg["training"].get("epochs", 100) // 3)
        cv_cfg["training"]["early_stopping_patience"] = 5

        trainer = Trainer(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            edge_index=edge_index,
            edge_weight=edge_weight,
            adj=adj,
            cfg=cv_cfg,
        )
        trainer.fit(resume="no")

        # Evaluate on validation fold
        val_metrics = _evaluate_fold(trainer, val_ds, edge_index, edge_weight, adj)
        fold_results.append(val_metrics)

        logger.info(
            "Fold %d results: MAE=%.4f | RMSE=%.4f | F1_warning=%.4f",
            fold_idx + 1,
            val_metrics.get("mae", float("nan")),
            val_metrics.get("rmse", float("nan")),
            val_metrics.get("f1_warning", float("nan")),
        )

    # Aggregate
    agg = _aggregate_fold_results(fold_results)
    _save_cv_results(agg, cfg)
    return agg


def _evaluate_fold(
    trainer,
    val_ds: "IonoDataset",
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    adj: Optional[torch.Tensor],
) -> Dict:
    """Run one-pass evaluation on a single fold."""
    from torch.utils.data import DataLoader
    import torch

    device = trainer.device
    loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    trainer.model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_pred = trainer.model(x, edge_index.to(device), edge_weight.to(device),
                                   adj.to(device) if adj is not None else None)
            all_pred.append(y_pred.squeeze(-1).cpu().numpy())
            all_true.append(y.squeeze(-1).cpu().numpy())

    y_pred_arr = np.concatenate([p.ravel() for p in all_pred])
    y_true_arr = np.concatenate([t.ravel() for t in all_true])

    reg = compute_regression_metrics(y_true_arr, y_pred_arr)
    evt = compute_event_metrics(y_true_arr, y_pred_arr)

    return {
        **reg,
        "f1_warning": evt.get("warning", {}).get("f1", float("nan")),
        "f1_severe": evt.get("severe", {}).get("f1", float("nan")),
        "auc_warning": evt.get("warning", {}).get("auc", float("nan")),
    }


def _aggregate_fold_results(fold_results: List[Dict]) -> Dict:
    """Compute mean ± std across folds."""
    agg = {}
    if not fold_results:
        return agg

    all_keys = fold_results[0].keys()
    for key in all_keys:
        vals = [f[key] for f in fold_results if not np.isnan(float(f.get(key, np.nan)))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    logger.info("Temporal CV Summary:")
    for k, v in agg.items():
        logger.info("  %s = %.4f", k, v)

    return agg


def _save_cv_results(results: Dict, cfg: Dict) -> None:
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "temporal_cv_results.json"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Temporal CV results saved → %s", path)
