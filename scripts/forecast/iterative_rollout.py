"""
forecast/iterative_rollout.py — IonoForecaster
6-hour iterative recursive forecasting pipeline.

Strategy:
  1. Load trained model (best checkpoint)
  2. For each station, roll forward 12 steps (30-min × 12 = 6 hours)
  3. At each step: use previous prediction as input for next step
  4. Monte Carlo Dropout for uncertainty quantification
  5. 5-model ensemble for robustness

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from scripts.models.st_gnn import STGNN
from scripts.utils import CheckpointManager, get_logger

logger = get_logger("forecast.rollout")


class IterativeForecast:
    """
    6-hour iterative recursive scintillation forecast.

    Rolls the ST-GNN forward horizon_steps (12) times, feeding each
    prediction back as the S4 input for the next step.

    Parameters
    ----------
    model : STGNN
    edge_index : torch.Tensor
    edge_weight : torch.Tensor
    adj : torch.Tensor, optional
    cfg : dict
    """

    def __init__(
        self,
        model: STGNN,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        adj: Optional[torch.Tensor],
        cfg: Dict,
    ) -> None:
        self.model = model
        self.cfg = cfg
        forecast_cfg = cfg.get("forecast", {})
        self.horizon_steps = forecast_cfg.get("horizon_steps", 12)
        self.mc_samples = forecast_cfg.get("mc_dropout_samples", 30)

        self.device = next(model.parameters()).device
        self.edge_index = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)
        self.adj = adj.to(self.device) if adj is not None else None

        # Index of S4 in the feature vector (must match FEATURE_COLS order)
        from scripts.dataset.iono_dataset import FEATURE_COLS
        self.s4_idx = FEATURE_COLS.index("s4") if "s4" in FEATURE_COLS else 0

    # ── public ────────────────────────────────────────────────────────────────

    def forecast(
        self,
        x_init: torch.Tensor,
        timestamps_init: pd.DatetimeIndex,
        station_ids: List[str],
    ) -> pd.DataFrame:
        """
        Generate 6-hour ahead S4 forecast.

        Parameters
        ----------
        x_init : torch.Tensor
            (1, T, N, F) initial input window (most recent T steps).
        timestamps_init : pd.DatetimeIndex
            Timestamps corresponding to the T input steps.
        station_ids : list of str

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, station_id, s4_forecast, s4_lower, s4_upper, s4_std
        """
        self.model.eval()
        x = x_init.clone().to(self.device)  # (1, T, N, F)
        T = x.shape[1]

        resolution_min = self.cfg["temporal"]["resolution_minutes"]
        last_ts = timestamps_init[-1]
        forecast_timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=resolution_min),
            periods=self.horizon_steps,
            freq=f"{resolution_min}min",
        )

        all_means = []
        all_stds = []

        logger.info("Running iterative forecast: %d steps (%d min horizon)",
                    self.horizon_steps, self.horizon_steps * resolution_min)

        for step in range(self.horizon_steps):
            mean_pred, std_pred = self.model.predict_with_dropout(
                x,
                self.edge_index,
                self.edge_weight,
                self.adj,
                n_samples=self.mc_samples,
            )
            # (1, N, 1) → (N,)
            mean_s4 = mean_pred.squeeze().cpu().numpy()  # (N,)
            std_s4 = std_pred.squeeze().cpu().numpy()    # (N,)

            all_means.append(mean_s4)
            all_stds.append(std_s4)

            # Update x: shift window by 1, update S4 channel with prediction
            x_new = x[:, 1:, :, :].clone()  # drop oldest step
            x_last = x[:, -1:, :, :].clone()  # copy last step features
            # Replace S4 in last step with prediction
            x_last[0, 0, :, self.s4_idx] = torch.from_numpy(mean_s4).float().to(self.device)
            x = torch.cat([x_new, x_last], dim=1)

        # Build output DataFrame
        records = []
        for step, (ts, mean_s4, std_s4) in enumerate(
            zip(forecast_timestamps, all_means, all_stds)
        ):
            for i, sid in enumerate(station_ids):
                mean_val = float(np.clip(mean_s4[i], 0, 1))
                std_val = float(std_s4[i])
                records.append({
                    "timestamp": ts,
                    "station_id": sid,
                    "forecast_step": step + 1,
                    "lead_time_min": (step + 1) * resolution_min,
                    "s4_forecast": mean_val,
                    "s4_lower": float(np.clip(mean_val - 1.96 * std_val, 0, 1)),
                    "s4_upper": float(np.clip(mean_val + 1.96 * std_val, 0, 1)),
                    "s4_std": std_val,
                })

        df = pd.DataFrame(records)
        logger.info("Forecast complete: %d predictions across %d stations.",
                    len(df), len(station_ids))
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble forecast
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleForecast:
    """
    5-model ensemble forecast for additional robustness.

    Loads multiple model checkpoints (trained with different seeds)
    and averages their predictions.

    Parameters
    ----------
    model_class : class (STGNN)
    model_kwargs : dict
    checkpoint_paths : list of Path
    edge_index, edge_weight, adj
    cfg : dict
    """

    def __init__(
        self,
        model_class,
        model_kwargs: Dict,
        checkpoint_paths: List[Path],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        adj: Optional[torch.Tensor],
        cfg: Dict,
    ) -> None:
        self.cfg = cfg
        self.models = []
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = device
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)
        self.adj = adj.to(device) if adj is not None else None

        for path in checkpoint_paths:
            if not Path(path).exists():
                logger.warning("Ensemble checkpoint not found: %s", path)
                continue
            model = model_class(**model_kwargs).to(device)
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

        if not self.models:
            raise RuntimeError(
                "No ensemble checkpoints found. Train with multiple seeds first."
            )
        logger.info("Ensemble loaded: %d models.", len(self.models))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction at a single timestep.

        Parameters
        ----------
        x : torch.Tensor (1, T, N, F)

        Returns
        -------
        mean_s4 : np.ndarray (N,)
        std_s4 : np.ndarray (N,)
        """
        x = x.to(self.device)
        preds = []
        for model in self.models:
            pred = model(x, self.edge_index, self.edge_weight, self.adj)
            preds.append(pred.squeeze().cpu().numpy())

        preds = np.stack(preds, axis=0)  # (n_models, N)
        return preds.mean(0), preds.std(0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast(
    cfg: Dict,
    model: STGNN,
    x_init: torch.Tensor,
    timestamps_init: pd.DatetimeIndex,
    station_ids: List[str],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    adj: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """
    Entry point called by main.py --step forecast.

    Saves forecast CSV to data/outputs/forecasts/.
    """
    forecaster = IterativeForecast(
        model=model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adj=adj,
        cfg=cfg,
    )
    df = forecaster.forecast(x_init, timestamps_init, station_ids)

    out_dir = Path(cfg["paths"]["outputs_dir"]) / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"forecast_{ts_str}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Forecast saved → %s", out_path)

    # Also save uncertainty
    unc_dir = out_dir / "uncertainty"
    unc_dir.mkdir(exist_ok=True)
    unc = df[["timestamp", "station_id", "s4_std", "s4_lower", "s4_upper"]]
    unc.to_json(unc_dir / f"uncertainty_{ts_str}.json", orient="records", indent=2)

    return df
