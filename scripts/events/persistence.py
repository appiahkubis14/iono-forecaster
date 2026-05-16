"""
events/persistence.py — IonoForecaster
Persistence baseline forecast: y_hat(t+k) = y(t)

The persistence model is the simplest possible forecast and serves as
a minimum skill benchmark. Any operational model must outperform it.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.models.metrics import compute_regression_metrics, compute_event_metrics
from scripts.utils import get_logger

logger = get_logger("events.persistence")


def persistence_forecast(
    s4_array: np.ndarray,
    forecast_steps: int = 12,
) -> np.ndarray:
    """
    Generate persistence forecast by repeating last observed value.

    Parameters
    ----------
    s4_array : np.ndarray
        (T, N) observed S4 values (T timesteps, N stations).
    forecast_steps : int
        Number of steps to forecast (12 = 6 hours at 30-min resolution).

    Returns
    -------
    np.ndarray
        (T, forecast_steps, N) persistence predictions.
    """
    T, N = s4_array.shape
    preds = np.zeros((T, forecast_steps, N), dtype=np.float32)
    for t in range(T):
        for step in range(forecast_steps):
            # Persistence: last observed value
            preds[t, step, :] = s4_array[t]
    return preds


def evaluate_persistence(
    s4_array: np.ndarray,
    forecast_steps: int = 12,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Evaluate persistence forecast against actual values.

    This provides the baseline skill score that the ST-GNN must beat.

    Parameters
    ----------
    s4_array : np.ndarray
        (T, N) observed S4.
    forecast_steps : int
    thresholds : dict, optional

    Returns
    -------
    dict
        Metrics per lead time step: {step: {mae, rmse, r2, f1_warning, ...}}
    """
    if thresholds is None:
        thresholds = {"watch": 0.3, "warning": 0.4, "severe": 0.7}

    T, N = s4_array.shape
    results = {}

    for step in range(1, forecast_steps + 1):
        if T <= step:
            continue

        # True: values at t+step
        y_true = s4_array[step:].ravel()
        # Persistence: values at t (before shift)
        y_pred = s4_array[:T - step].ravel()

        # Remove NaN pairs
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[valid], y_pred[valid]

        if len(y_true) == 0:
            continue

        reg = compute_regression_metrics(y_true, y_pred)
        evt = compute_event_metrics(y_true, y_pred, thresholds)

        results[f"step_{step}"] = {
            "lead_time_min": step * 30,
            **reg,
            "f1_warning": evt.get("warning", {}).get("f1", float("nan")),
            "f1_severe": evt.get("severe", {}).get("f1", float("nan")),
        }

        logger.info(
            "Persistence step=%d (lead=%d min): MAE=%.4f | RMSE=%.4f | F1_warn=%.4f",
            step, step * 30,
            reg.get("mae", float("nan")),
            reg.get("rmse", float("nan")),
            evt.get("warning", {}).get("f1", float("nan")),
        )

    # Summary: average across all steps
    metric_keys = ["mae", "rmse", "r2", "f1_warning"]
    summary = {}
    for k in metric_keys:
        vals = [v[k] for v in results.values() if k in v and not np.isnan(v[k])]
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))

    results["summary"] = summary
    logger.info("Persistence baseline summary: %s", summary)
    return results


def compute_skill_score(
    model_metrics: Dict[str, float],
    persistence_metrics: Dict[str, float],
    metric: str = "rmse",
) -> float:
    """
    Compute skill score of model vs persistence baseline.

    Skill = 1 - (model_metric / persistence_metric)

    Score of 0 = same as persistence.
    Score of 1 = perfect (zero error).
    Score < 0 = worse than persistence.

    Parameters
    ----------
    model_metrics : dict
    persistence_metrics : dict
    metric : str
        Which metric to use (lower-is-better: 'mae', 'rmse').

    Returns
    -------
    float
    """
    model_val = model_metrics.get(metric, float("nan"))
    persist_val = persistence_metrics.get(metric, float("nan"))

    if np.isnan(model_val) or np.isnan(persist_val) or persist_val == 0:
        return float("nan")

    skill = 1.0 - (model_val / persist_val)
    logger.info(
        "Skill score (%s): %.4f (model=%.4f, persistence=%.4f)",
        metric, skill, model_val, persist_val,
    )
    return float(skill)
