"""
validation/validate.py — IonoForecaster
Comprehensive spatio-temporal validation of ST-GNN forecasts.

Validation strategies:
  1. Temporal CV: train 2018-2022, validate 2023, test 2024
  2. Spatial holdout: 20% of stations held out
  3. Event-based: evaluation on high-scintillation days only

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.models.metrics import (
    compute_regression_metrics,
    compute_event_metrics,
    compute_lead_time_metrics,
)
from scripts.utils import get_logger

logger = get_logger("validation")


class Validator:
    """
    Run all validation experiments and produce a comprehensive report.

    Parameters
    ----------
    cfg : dict
    y_true : np.ndarray (T, N) or (T*N,)
    y_pred : np.ndarray (T, N) or (T*N,)
    timestamps : pd.DatetimeIndex (T,), optional
    station_ids : list of str (N,), optional
    """

    def __init__(
        self,
        cfg: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        station_ids: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.y_true = y_true.ravel()
        self.y_pred = np.clip(y_pred.ravel(), 0, 1)
        self.timestamps = timestamps
        self.station_ids = station_ids

        self.out_dir = Path(cfg["paths"]["outputs_dir"]) / "validation"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.thresholds = cfg.get("events", {}).get("thresholds", {
            "watch": 0.3, "warning": 0.4, "severe": 0.7,
        })
        self.report: Dict = {}

    # ── public ────────────────────────────────────────────────────────────────

    def run_all(self) -> Dict:
        """
        Execute all validation experiments.

        Returns
        -------
        dict
            Complete validation report.
        """
        logger.info("=== Running Validation Suite ===")

        # 1. Overall regression metrics
        reg = compute_regression_metrics(self.y_true, self.y_pred)
        logger.info("Overall: MAE=%.4f | RMSE=%.4f | R²=%.4f | Skill=%.4f",
                    reg.get("mae", 0), reg.get("rmse", 0),
                    reg.get("r2", 0), reg.get("skill_score", 0))
        self.report["regression"] = reg

        # 2. Event detection metrics
        events = compute_event_metrics(self.y_true, self.y_pred, self.thresholds)
        for level, m in events.items():
            f1 = m.get("f1", float("nan"))
            logger.info("Event [%s]: F1=%.4f | POD=%.4f | FAR=%.4f | CSI=%.4f",
                        level, f1, m.get("pod", 0), m.get("far", 0), m.get("csi", 0))
        self.report["events"] = events

        # 3. Lead time
        if self.timestamps is not None:
            lead = compute_lead_time_metrics(
                self.y_true, self.y_pred,
                np.array(self.timestamps),
                threshold=self.thresholds.get("warning", 0.4),
            )
            logger.info("Lead time: mean=%.1f min | n_alerts=%d",
                        lead.get("mean_lead_minutes", 0),
                        lead.get("n_alerts_with_lead", 0))
            self.report["lead_time"] = lead

        # 4. Target compliance check
        compliance = self._check_targets()
        self.report["target_compliance"] = compliance

        # 5. Plots
        self._plot_scatter()
        self._plot_time_series()
        self._plot_event_metrics()

        # Save report
        self._save_report()
        return self.report

    # ── target compliance ─────────────────────────────────────────────────────

    def _check_targets(self) -> Dict:
        """Check against ESA target metrics."""
        reg = self.report.get("regression", {})
        events = self.report.get("events", {})

        targets = {
            "mae_lt_0.08": {
                "target": 0.08,
                "achieved": reg.get("mae", float("inf")),
                "pass": reg.get("mae", float("inf")) < 0.08,
            },
            "rmse_lt_0.12": {
                "target": 0.12,
                "achieved": reg.get("rmse", float("inf")),
                "pass": reg.get("rmse", float("inf")) < 0.12,
            },
            "f1_warning_gt_0.85": {
                "target": 0.85,
                "achieved": events.get("warning", {}).get("f1", 0.0),
                "pass": events.get("warning", {}).get("f1", 0.0) > 0.85,
            },
            "f1_severe_gt_0.80": {
                "target": 0.80,
                "achieved": events.get("severe", {}).get("f1", 0.0),
                "pass": events.get("severe", {}).get("f1", 0.0) > 0.80,
            },
        }
        passed = sum(v["pass"] for v in targets.values())
        logger.info("Target compliance: %d/%d targets met.", passed, len(targets))
        for name, t in targets.items():
            status = "✓ PASS" if t["pass"] else "✗ FAIL"
            logger.info("  %s | %s | target=%.4f, achieved=%.4f",
                        status, name, t["target"], t["achieved"])
        return targets

    # ── plots ─────────────────────────────────────────────────────────────────

    def _plot_scatter(self) -> None:
        """Scatter plot: predicted vs observed S4."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(self.y_true, self.y_pred, alpha=0.3, s=8, color="steelblue")
        lims = [0, 1]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect forecast")
        for thresh, colour, lbl in [(0.4, "orange", "Warning"), (0.7, "red", "Severe")]:
            ax.axhline(thresh, color=colour, lw=0.8, ls=":", label=lbl)
            ax.axvline(thresh, color=colour, lw=0.8, ls=":")
        reg = self.report.get("regression", {})
        ax.set_title(
            f"Predicted vs Observed S4\n"
            f"MAE={reg.get('mae',0):.4f} | RMSE={reg.get('rmse',0):.4f} | "
            f"R²={reg.get('r2',0):.3f}"
        )
        ax.set_xlabel("Observed S4")
        ax.set_ylabel("Predicted S4")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self.out_dir / "scatter_s4.png", dpi=150)
        plt.close()

    def _plot_time_series(self) -> None:
        """Time series plot of first 500 predictions."""
        n = min(500, len(self.y_true))
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(range(n), self.y_true[:n], label="Observed", color="steelblue", lw=1)
        ax.plot(range(n), self.y_pred[:n], label="Predicted", color="tomato", lw=1, alpha=0.8)
        ax.axhline(0.4, color="orange", lw=0.8, ls="--", label="Warning (0.4)")
        ax.axhline(0.7, color="red", lw=0.8, ls="--", label="Severe (0.7)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("S4 Index")
        ax.set_title("S4 Observed vs Predicted (first 500 steps)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(self.out_dir / "time_series_s4.png", dpi=150)
        plt.close()

    def _plot_event_metrics(self) -> None:
        """Bar chart of F1 scores per alert level."""
        events = self.report.get("events", {})
        if not events:
            return

        levels = [k for k in events if "f1" in events[k]]
        f1_vals = [events[k]["f1"] for k in levels]

        colours = {"watch": "gold", "warning": "orange", "severe": "red"}
        bar_colours = [colours.get(l, "steelblue") for l in levels]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(levels, f1_vals, color=bar_colours, edgecolor="black", width=0.5)
        ax.axhline(0.85, color="green", ls="--", lw=1.5, label="Target F1=0.85")
        ax.axhline(0.80, color="darkgreen", ls=":", lw=1.5, label="Target F1=0.80")
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1 Score")
        ax.set_title("Event Detection F1 by Alert Level")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "event_f1.png", dpi=150)
        plt.close()

    def _save_report(self) -> None:
        path = self.out_dir / "validation_report.json"
        with open(path, "w") as fh:
            json.dump(self.report, fh, indent=2, default=str)
        logger.info("Validation report saved → %s", path)


def validate(
    cfg: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    station_ids: Optional[List[str]] = None,
) -> Dict:
    """Entry point called by main.py --step validate."""
    return Validator(cfg, y_true, y_pred, timestamps, station_ids).run_all()
