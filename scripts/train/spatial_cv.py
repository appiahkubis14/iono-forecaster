"""
train/spatial_cv.py — IonoForecaster
Station hold-out cross-validation (spatial CV).

For each fold, one or more stations are withheld from training.
This tests the model's ability to generalise to unmonitored locations —
critical for operational forecasting across data-sparse equatorial Africa.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts.models.metrics import compute_regression_metrics, compute_event_metrics
from scripts.utils import get_logger, set_seed

logger = get_logger("train.spatial_cv")


def spatial_cv_splits(
    station_ids: List[str],
    n_folds: int = 5,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate spatial CV splits: (train_stations, val_stations) pairs.

    Parameters
    ----------
    station_ids : list of str
    n_folds : int
    holdout_fraction : float
        Fraction of stations held out per fold.
    seed : int

    Returns
    -------
    list of (train_station_ids, val_station_ids) tuples
    """
    rng = np.random.default_rng(seed)
    n = len(station_ids)
    n_holdout = max(1, int(n * holdout_fraction))

    indices = np.arange(n)
    splits = []

    for fold in range(n_folds):
        # Rotate which stations are held out
        start = (fold * n_holdout) % n
        holdout_idx = [(start + i) % n for i in range(n_holdout)]
        train_idx = [i for i in indices if i not in holdout_idx]

        train_sids = [station_ids[i] for i in train_idx]
        val_sids = [station_ids[i] for i in holdout_idx]
        splits.append((train_sids, val_sids))

        logger.info(
            "Spatial CV Fold %d: train=%s | holdout=%s",
            fold + 1, train_sids, val_sids,
        )

    return splits


def run_spatial_cv(
    cfg: Dict,
    feature_array: np.ndarray,   # (T, N, F)
    target_array: np.ndarray,    # (T, N)
    station_ids: List[str],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    adj: Optional[torch.Tensor] = None,
    n_folds: int = 5,
    holdout_fraction: float = 0.2,
) -> Dict:
    """
    Execute spatial station hold-out CV.

    For each fold:
      - Train on (T_train, N_train, F) — all time, subset of stations
      - Evaluate on held-out stations using all time

    Parameters
    ----------
    cfg : dict
    feature_array : np.ndarray (T, N, F)
    target_array : np.ndarray (T, N)
    station_ids : list of str (length N)
    edge_index, edge_weight : torch.Tensor
    adj : torch.Tensor, optional
    n_folds : int
    holdout_fraction : float

    Returns
    -------
    dict
        Aggregated spatial CV results.
    """
    from scripts.models.st_gnn import build_model
    from scripts.dataset.iono_dataset import IonoDataset
    from scripts.train.train import Trainer

    splits = spatial_cv_splits(
        station_ids, n_folds=n_folds,
        holdout_fraction=holdout_fraction,
        seed=cfg.get("training", {}).get("seed", 42),
    )

    n_features = feature_array.shape[2]
    n_nodes = len(station_ids)
    input_steps = cfg["model"].get("input_timesteps", 12)
    T = feature_array.shape[0]

    # Temporal split for train/val within each spatial fold
    n_train_t = int(T * 0.80)

    fold_results = []

    for fold_idx, (train_sids, val_sids) in enumerate(splits):
        logger.info("=" * 60)
        logger.info("Spatial CV Fold %d/%d — holdout: %s", fold_idx + 1, n_folds, val_sids)
        logger.info("=" * 60)

        set_seed(cfg.get("training", {}).get("seed", 42) + fold_idx * 100)

        # Station masks
        train_node_idx = [station_ids.index(s) for s in train_sids]
        val_node_idx = [station_ids.index(s) for s in val_sids]

        # Station mask for loss computation (exclude held-out from training loss)
        station_mask = np.zeros(n_nodes, dtype=bool)
        for i in train_node_idx:
            station_mask[i] = True

        # Datasets — use full station set but mask held-out in loss
        train_ds = IonoDataset(
            feature_array[:n_train_t],
            target_array[:n_train_t],
            input_steps=input_steps,
            mode="train",
            station_mask=station_mask,
        )
        val_ds = IonoDataset(
            feature_array[n_train_t:],
            target_array[n_train_t:],
            input_steps=input_steps,
            mode="val",
        )

        model = build_model(cfg, n_features=n_features, n_nodes=n_nodes)

        cv_cfg = copy.deepcopy(cfg)
        cv_cfg["training"]["epochs"] = max(15, cfg["training"].get("epochs", 100) // 4)
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

        # Evaluate ONLY on held-out stations
        val_metrics = _evaluate_holdout(
            trainer, val_ds, val_node_idx, edge_index, edge_weight, adj
        )
        val_metrics["holdout_stations"] = val_sids
        fold_results.append(val_metrics)

        logger.info(
            "Fold %d holdout metrics: MAE=%.4f | RMSE=%.4f | F1_warning=%.4f",
            fold_idx + 1,
            val_metrics.get("mae", float("nan")),
            val_metrics.get("rmse", float("nan")),
            val_metrics.get("f1_warning", float("nan")),
        )

    agg = _aggregate_spatial_results(fold_results)
    _save_spatial_cv_results(agg, fold_results, cfg)
    return agg


def _evaluate_holdout(
    trainer,
    val_ds,
    holdout_node_idx: List[int],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    adj: Optional[torch.Tensor],
) -> Dict:
    """Evaluate predictions only on held-out stations."""
    from torch.utils.data import DataLoader
    import torch

    device = trainer.device
    loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    trainer.model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)  # (B, N, 1)
            y_pred = trainer.model(
                x, edge_index.to(device), edge_weight.to(device),
                adj.to(device) if adj is not None else None,
            )
            # Extract held-out stations only
            y_ho = y[:, holdout_node_idx, :].squeeze(-1).cpu().numpy()
            y_pred_ho = y_pred[:, holdout_node_idx, :].squeeze(-1).cpu().numpy()
            all_pred.append(y_pred_ho.ravel())
            all_true.append(y_ho.ravel())

    y_pred_arr = np.concatenate(all_pred)
    y_true_arr = np.concatenate(all_true)

    reg = compute_regression_metrics(y_true_arr, y_pred_arr)
    evt = compute_event_metrics(y_true_arr, y_pred_arr)

    return {
        **reg,
        "f1_warning": evt.get("warning", {}).get("f1", float("nan")),
        "f1_severe": evt.get("severe", {}).get("f1", float("nan")),
        "n_holdout_predictions": len(y_pred_arr),
    }


def _aggregate_spatial_results(fold_results: List[Dict]) -> Dict:
    """Aggregate spatial CV metrics."""
    numeric_keys = [k for k in fold_results[0] if k != "holdout_stations"]
    agg = {}
    for key in numeric_keys:
        try:
            vals = [float(f[key]) for f in fold_results if not np.isnan(float(f.get(key, np.nan)))]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        except (TypeError, ValueError):
            pass

    logger.info("Spatial CV Summary:")
    for k, v in agg.items():
        logger.info("  %s = %.4f", k, v)
    return agg


def _save_spatial_cv_results(agg: Dict, fold_results: List[Dict], cfg: Dict) -> None:
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "spatial_cv_results.json"
    with open(path, "w") as fh:
        json.dump({"summary": agg, "folds": fold_results}, fh, indent=2, default=str)
    logger.info("Spatial CV results saved → %s", path)
