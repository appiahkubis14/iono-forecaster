"""
utils.py — IonoForecaster
Centralised logging, configuration loading, checkpointing, and helper utilities.

Author: Samuel Appiah Kubi
Institution: Paris Lodron University Salzburg
Project: IonoForecaster — Copernicus Master's in Digital Earth
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate the YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve environment-variable overrides for credentials
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """Apply environment variable overrides for sensitive credentials."""
    env_map = {
        ("data_sources", "swarm", "username"): "ESA_SWARM_USER",
        ("data_sources", "swarm", "password"): "ESA_SWARM_PASS",
        ("alerts", "telegram_bot_token"): "TELEGRAM_BOT_TOKEN",
        ("alerts", "telegram_chat_id"): "TELEGRAM_CHAT_ID",
        ("alerts", "email_smtp"): "SMTP_SERVER",
        ("alerts", "email_from"): "EMAIL_FROM",
        ("alerts", "email_to"): "EMAIL_TO",
    }
    for key_path, env_var in env_map.items():
        val = os.environ.get(env_var)
        if val:
            node = cfg
            for k in key_path[:-1]:
                node = node.setdefault(k, {})
            node[key_path[-1]] = val


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "data/logs",
    name: str = "iono_forecaster",
) -> logging.Logger:
    """
    Configure root logger with console and rotating file handlers.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_dir : str or Path
        Directory where log files are stored.
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    logger.info("Logging initialised → %s", log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the iono_forecaster namespace."""
    return logging.getLogger(f"iono_forecaster.{name}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages save/load of pipeline step checkpoints and model weights.

    Checkpoints are stored as JSON (metadata) + optional .pt (model weights).
    """

    def __init__(self, checkpoint_dir: str | Path, step: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.step = step
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._meta_file = self.checkpoint_dir / f"{step}_checkpoint.json"
        self.logger = get_logger("checkpoint")

    # ── metadata ──────────────────────────────────────────────────────────────

    def save_meta(self, data: Dict[str, Any]) -> None:
        """Persist arbitrary JSON-serialisable metadata for a step."""
        payload = {
            "step": self.step,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }
        with open(self._meta_file, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        self.logger.info("Checkpoint saved: %s", self._meta_file)

    def load_meta(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint metadata if it exists."""
        if not self._meta_file.exists():
            return None
        with open(self._meta_file) as fh:
            data = json.load(fh)
        self.logger.info("Checkpoint loaded: %s", self._meta_file)
        return data

    def exists(self) -> bool:
        """Return True if a checkpoint exists for this step."""
        return self._meta_file.exists()

    def delete(self) -> None:
        """Remove checkpoint (start fresh)."""
        if self._meta_file.exists():
            self._meta_file.unlink()
            self.logger.info("Checkpoint deleted: %s", self._meta_file)

    # ── model weights ─────────────────────────────────────────────────────────

    def model_path(self, tag: str = "last") -> Path:
        """Return path for model weight file."""
        return self.checkpoint_dir / f"{self.step}_{tag}.pt"

    def save_model(self, state_dict: Dict, tag: str = "last") -> Path:
        """Save PyTorch state dict (lazy import to avoid hard dependency)."""
        import torch
        path = self.model_path(tag)
        torch.save(state_dict, path)
        self.logger.info("Model weights saved → %s", path)
        return path

    def load_model(self, tag: str = "last") -> Optional[Dict]:
        """Load PyTorch state dict if it exists."""
        import torch
        path = self.model_path(tag)
        if not path.exists():
            self.logger.warning("Model weights not found: %s", path)
            return None
        state = torch.load(path, map_location="cpu")
        self.logger.info("Model weights loaded ← %s", path)
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for Python, NumPy, and PyTorch (if available).

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create all directories listed in config paths section."""
    paths = cfg.get("paths", {})
    for key, path_str in paths.items():
        Path(path_str).mkdir(parents=True, exist_ok=True)

    # Also ensure output sub-directories
    out = Path(paths.get("outputs_dir", "data/outputs"))
    for sub in ["forecasts", "forecasts/uncertainty", "events", "alerts", "stac", "dashboard"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> None:
    """Copy file, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager that records elapsed wall time."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger = get_logger("timer")
        logger.info("%s completed in %.2f s", self.name, self.elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Data validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataframe(df: "pd.DataFrame", required_cols: list[str], name: str = "") -> None:
    """
    Assert that a DataFrame has required columns and is non-empty.

    Parameters
    ----------
    df : pd.DataFrame
    required_cols : list of str
    name : str
        Label for error messages.

    Raises
    ------
    ValueError
    """
    if df.empty:
        raise ValueError(f"DataFrame '{name}' is empty.")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame '{name}' missing columns: {missing}")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates in decimal degrees.

    Returns
    -------
    float
        Distance in kilometres.
    """
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 5.0, exceptions: tuple = (Exception,)):
    """
    Decorator that retries a function on failure.

    Parameters
    ----------
    max_attempts : int
    delay : float
        Seconds between attempts.
    exceptions : tuple
        Exception types to catch.
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("retry")
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs…",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator
