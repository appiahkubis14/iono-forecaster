"""
models/metrics.py — IonoForecaster
Evaluation metrics: MAE, RMSE, F1, R², skill score, and event-based metrics.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard regression metrics for S4 forecast evaluation.

    Parameters
    ----------
    y_true, y_pred : np.ndarray (N,)
    mask : np.ndarray bool, optional
        Valid-data mask.

    Returns
    -------
    dict with keys: mae, rmse, r2, mbe, skill_score
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    if len(y_true) == 0:
        return {}

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mbe = float(np.mean(y_pred - y_true))  # mean bias error

    # R² coefficient of determination
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-9))

    # Skill score vs persistence baseline
    # Persistence: y_pred_persist = y_true[:-1], shifted
    if len(y_true) > 1:
        persist_rmse = float(np.sqrt(np.mean((y_true[1:] - y_true[:-1]) ** 2)))
        skill = float(1.0 - rmse / (persist_rmse + 1e-9))
    else:
        skill = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mbe": mbe,
        "skill_score": skill,
    }


def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute event detection metrics at multiple S4 thresholds.

    Parameters
    ----------
    y_true, y_pred : np.ndarray (N,)
    thresholds : dict, optional
        {'watch': 0.3, 'warning': 0.4, 'severe': 0.7}

    Returns
    -------
    dict of dicts: {level: {f1, precision, recall, auc, pod, far, csi}}
    """
    if thresholds is None:
        thresholds = {"watch": 0.3, "warning": 0.4, "severe": 0.7}

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    results = {}

    for level, thresh in thresholds.items():
        y_t_bin = (y_true >= thresh).astype(int)
        y_p_bin = (y_pred >= thresh).astype(int)

        if y_t_bin.sum() == 0:
            results[level] = {"note": "no events in truth"}
            continue

        f1 = float(f1_score(y_t_bin, y_p_bin, zero_division=0))
        prec = float(precision_score(y_t_bin, y_p_bin, zero_division=0))
        rec = float(recall_score(y_t_bin, y_p_bin, zero_division=0))

        # AUC (uses continuous predictions)
        try:
            auc = float(roc_auc_score(y_t_bin, y_pred))
        except Exception:
            auc = float("nan")

        # Contingency table metrics
        tp = int(((y_t_bin == 1) & (y_p_bin == 1)).sum())
        fp = int(((y_t_bin == 0) & (y_p_bin == 1)).sum())
        fn = int(((y_t_bin == 1) & (y_p_bin == 0)).sum())

        pod = tp / (tp + fn + 1e-9)   # Probability of Detection (= Recall)
        far = fp / (tp + fp + 1e-9)   # False Alarm Rate
        csi = tp / (tp + fp + fn + 1e-9)  # Critical Success Index

        results[level] = {
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "pod": float(pod),
            "far": float(far),
            "csi": float(csi),
            "n_events": int(y_t_bin.sum()),
        }

    return results


def compute_lead_time_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.4,
    max_lead_minutes: int = 60,
    dt_minutes: int = 30,
) -> Dict[str, float]:
    """
    Compute alert lead time: time between prediction and actual onset.

    Parameters
    ----------
    y_true, y_pred : np.ndarray (N,)
    timestamps : np.ndarray (N,)
    threshold : float
    max_lead_minutes : int
        Maximum allowed lead time for it to count.
    dt_minutes : int
        Timestep resolution.

    Returns
    -------
    dict: mean_lead_minutes, median_lead_minutes, n_alerts_with_lead
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    n = len(y_true)
    lead_times = []

    # Find actual event onsets
    actual_events = np.where((y_true[1:] >= threshold) & (y_true[:-1] < threshold))[0] + 1

    for event_idx in actual_events:
        # Look backwards to find when prediction first exceeded threshold
        for look_back in range(max_lead_minutes // dt_minutes, 0, -1):
            pred_idx = event_idx - look_back
            if 0 <= pred_idx < n and y_pred[pred_idx] >= threshold:
                lead_times.append(look_back * dt_minutes)
                break

    if not lead_times:
        return {
            "mean_lead_minutes": 0.0,
            "median_lead_minutes": 0.0,
            "n_alerts_with_lead": 0,
            "n_events": len(actual_events),
        }

    return {
        "mean_lead_minutes": float(np.mean(lead_times)),
        "median_lead_minutes": float(np.median(lead_times)),
        "n_alerts_with_lead": len(lead_times),
        "n_events": len(actual_events),
    }
