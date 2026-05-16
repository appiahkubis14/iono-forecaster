"""
train/train.py — IonoForecaster
Training loop for the ST-GNN with:
  - Mixed-precision (AMP)
  - Gradient clipping
  - ReduceLROnPlateau scheduler
  - Early stopping
  - TensorBoard logging
  - Checkpoint/resume support

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripts.dataset.iono_dataset import IonoDataset
from scripts.models.losses import CombinedLoss
from scripts.models.metrics import compute_regression_metrics, compute_event_metrics
from scripts.models.st_gnn import STGNN
from scripts.utils import CheckpointManager, get_logger, set_seed

logger = get_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages ST-GNN training with all ESA-grade features.

    Parameters
    ----------
    model : STGNN
    train_ds, val_ds : IonoDataset
    edge_index : torch.Tensor (2, E)
    edge_weight : torch.Tensor (E,)
    adj : torch.Tensor (N, N), optional
    cfg : dict
    """

    def __init__(
        self,
        model: STGNN,
        train_ds: IonoDataset,
        val_ds: IonoDataset,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        adj: Optional[torch.Tensor],
        cfg: Dict,
    ) -> None:
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.train_cfg = cfg.get("training", {})

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info("Training on device: %s", self.device)

        self.model = self.model.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)
        self.adj = adj.to(self.device) if adj is not None else None

        # Loss, optimiser, scheduler
        self.criterion = CombinedLoss(
            mse_weight=self.train_cfg.get("loss_mse_weight", 0.7),
            mae_weight=self.train_cfg.get("loss_mae_weight", 0.3),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.get("learning_rate", 1e-3),
            weight_decay=self.train_cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.train_cfg.get("scheduler_patience", 10),
            factor=0.5,
            verbose=True,
        )
        self.scaler = GradScaler(enabled=self.train_cfg.get("mixed_precision", True)
                                 and self.device.type == "cuda")

        # Checkpoint manager
        models_dir = Path(cfg["paths"]["models_dir"])
        self._ckpt = CheckpointManager(models_dir, "training")

        # TensorBoard
        log_dir = Path(cfg["paths"]["logs_dir"])
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # State
        self.history: Dict = {
            "train_loss": [], "val_loss": [],
            "val_mae": [], "val_rmse": [], "val_f1_warning": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0

    # ── public ────────────────────────────────────────────────────────────────

    def fit(self, resume: str = "last") -> Dict:
        """
        Train the model.

        Parameters
        ----------
        resume : str
            'last' = resume from checkpoint if available; 'no' = train from scratch.

        Returns
        -------
        dict
            Training history.
        """
        set_seed(self.train_cfg.get("seed", 42))

        if resume == "last":
            self._try_resume()

        batch_size = self.train_cfg.get("batch_size", 32)
        epochs = self.train_cfg.get("epochs", 100)
        early_stop = self.train_cfg.get("early_stopping_patience", 20)

        train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            "Starting training: epochs=%d | batch=%d | early_stop=%d",
            epochs, batch_size, early_stop,
        )

        for epoch in range(self.start_epoch, epochs):
            t0 = time.perf_counter()

            train_loss = self._train_epoch(train_loader, epoch)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            elapsed = time.perf_counter() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_metrics.get("mae", float("nan")))
            self.history["val_rmse"].append(val_metrics.get("rmse", float("nan")))
            self.history["lr"].append(lr)

            # TensorBoard
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalar("Metrics/MAE", val_metrics.get("mae", 0), epoch)
            self.writer.add_scalar("Metrics/RMSE", val_metrics.get("rmse", 0), epoch)
            self.writer.add_scalar("LR", lr, epoch)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "MAE=%.4f | RMSE=%.4f | lr=%.6f | %.1fs",
                epoch + 1, epochs, train_loss, val_loss,
                val_metrics.get("mae", float("nan")),
                val_metrics.get("rmse", float("nan")),
                lr, elapsed,
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._ckpt.save_model(self.model.state_dict(), tag="best")
                self.patience_counter = 0
                logger.info("  ✓ New best model saved (val_loss=%.4f)", val_loss)
            else:
                self.patience_counter += 1

            # Save last checkpoint every N epochs
            freq = self.cfg.get("logging", {}).get("checkpoint_frequency", 1)
            if (epoch + 1) % freq == 0:
                self._ckpt.save_model(self.model.state_dict(), tag="last")
                self._ckpt.save_meta({
                    "epoch": epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    "history": self.history,
                })

            # Early stopping
            if self.patience_counter >= early_stop:
                logger.info(
                    "Early stopping triggered at epoch %d (no improvement for %d epochs).",
                    epoch + 1, early_stop,
                )
                break

        self.writer.close()
        self._save_history()
        logger.info("Training complete. Best val_loss: %.4f", self.best_val_loss)
        return self.history

    # ── private ───────────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """One training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        clip_norm = self.train_cfg.get("gradient_clip_norm", 1.0)
        use_amp = self.train_cfg.get("mixed_precision", True) and self.device.type == "cuda"

        for batch in loader:
            x = batch["x"].to(self.device)     # (B, T, N, F)
            y = batch["y"].to(self.device)     # (B, N, 1)
            mask = batch["mask"].to(self.device)  # (B, N)

            self.optimizer.zero_grad()

            with autocast(enabled=use_amp):
                y_pred = self.model(x, self.edge_index, self.edge_weight, self.adj)
                # Flatten for loss
                loss = self.criterion(
                    y_pred.squeeze(-1)[mask],
                    y.squeeze(-1)[mask],
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(
        self, loader: DataLoader
    ) -> Tuple[float, Dict]:
        """One validation epoch with full metrics."""
        self.model.eval()
        total_loss = 0.0
        all_pred, all_true = [], []
        n_batches = 0

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["mask"].to(self.device)

            y_pred = self.model(x, self.edge_index, self.edge_weight, self.adj)
            loss = self.criterion(
                y_pred.squeeze(-1)[mask],
                y.squeeze(-1)[mask],
            )
            total_loss += loss.item()
            n_batches += 1

            all_pred.append(y_pred.squeeze(-1).cpu().numpy())
            all_true.append(y.squeeze(-1).cpu().numpy())

        val_loss = total_loss / max(n_batches, 1)

        y_pred_arr = np.concatenate([p.ravel() for p in all_pred])
        y_true_arr = np.concatenate([t.ravel() for t in all_true])

        metrics = compute_regression_metrics(y_true_arr, y_pred_arr)
        return val_loss, metrics

    def _try_resume(self) -> None:
        """Attempt to resume training from last checkpoint."""
        meta = self._ckpt.load_meta()
        if meta:
            weights = self._ckpt.load_model("last")
            if weights:
                self.model.load_state_dict(weights)
                self.start_epoch = meta.get("epoch", 0)
                self.best_val_loss = meta.get("best_val_loss", float("inf"))
                self.history = meta.get("history", self.history)
                logger.info(
                    "Resumed from epoch %d (best_val_loss=%.4f).",
                    self.start_epoch, self.best_val_loss,
                )

    def _save_history(self) -> None:
        """Persist training history as JSON."""
        path = Path(self.cfg["paths"]["models_dir"]) / "training_history.json"
        with open(path, "w") as fh:
            json.dump(self.history, fh, indent=2, default=str)
        logger.info("Training history saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: Dict,
    model: STGNN,
    train_ds: IonoDataset,
    val_ds: IonoDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    adj: Optional[torch.Tensor] = None,
    resume: str = "last",
) -> Dict:
    """Entry point called by main.py --step train."""
    trainer = Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adj=adj,
        cfg=cfg,
    )
    return trainer.fit(resume=resume)
