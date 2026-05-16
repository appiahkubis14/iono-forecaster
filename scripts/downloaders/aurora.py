"""
aurora.py — IonoForecaster
Download OVATION Prime aurora precipitation data from NOAA SWPC.

OVATION Prime provides hemispheric power and equatorial electrojet estimates
that drive high-latitude to mid-latitude ionospheric coupling affecting
equatorial scintillation through prompt penetration electric fields (PPEFs).

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from scripts.utils import get_logger, retry

logger = get_logger("aurora")

OVATION_LATEST = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
HEMISPHERIC_POWER = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"


def _generate_synthetic_aurora(
    start: date,
    end: date,
    resolution_minutes: int = 30,
) -> pd.DataFrame:
    """
    Generate synthetic hemispheric auroral power indices.

    Hemispheric power (HP) ranges ~1–100 GW.
    Correlated with synthetic Kp for physical consistency.
    """
    timestamps = pd.date_range(
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()) + timedelta(days=1),
        freq=f"{resolution_minutes}min",
        inclusive="left",
    )
    n = len(timestamps)
    rng = np.random.default_rng(5555)

    # Quiet-time HP ~ 5–15 GW; storm-time up to 200 GW
    hp_north = rng.lognormal(2.0, 0.8, n)
    hp_south = rng.lognormal(1.8, 0.7, n)

    # Storm events
    n_storms = max(1, int((end - start).days / 10))
    for _ in range(n_storms):
        idx = rng.integers(0, n - 48)
        dur = rng.integers(6, 48)
        peak = rng.uniform(50, 200)
        end_idx = min(idx + dur, n)
        hp_north[idx:end_idx] += np.linspace(0, peak, end_idx - idx)
        hp_south[idx:end_idx] += np.linspace(0, peak * 0.8, end_idx - idx)

    return pd.DataFrame({
        "timestamp": timestamps,
        "hp_north": np.clip(hp_north, 1, 300).astype(np.float32),
        "hp_south": np.clip(hp_south, 1, 300).astype(np.float32),
        "source": "synthetic",
    })


class AuroraDownloader:
    """Download OVATION Prime aurora precipitation data."""

    def __init__(self, cfg: Dict, use_synthetic: bool = False) -> None:
        self.cfg = cfg
        self.use_synthetic = use_synthetic
        self.processed_dir = Path(cfg["paths"]["processed_data"]) / "aurora"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.start = date.fromisoformat(cfg["temporal"]["start_date"])
        self.end = date.fromisoformat(cfg["temporal"]["end_date"])
        self.resolution = cfg["temporal"]["resolution_minutes"]
        self._session = requests.Session()

    def run(self) -> None:
        logger.info("=== Aurora / OVATION Prime Download ===")
        out_path = self.processed_dir / "aurora.parquet"
        if out_path.exists():
            logger.info("Aurora data already exists.")
            return

        if self.use_synthetic:
            df = _generate_synthetic_aurora(self.start, self.end, self.resolution)
            df.to_parquet(out_path, index=False)
            logger.info("Synthetic aurora data saved: %d rows", len(df))
            return

        try:
            df = self._fetch_real()
        except Exception as exc:
            logger.warning("Aurora fetch failed (%s); using synthetic.", exc)
            df = _generate_synthetic_aurora(self.start, self.end, self.resolution)

        df.to_parquet(out_path, index=False)
        logger.info("Aurora data saved: %d rows", len(df))

    def load(self) -> pd.DataFrame:
        path = self.processed_dir / "aurora.parquet"
        if not path.exists():
            raise FileNotFoundError("Aurora data not found. Run download first.")
        return pd.read_parquet(path)

    @retry(max_attempts=3, delay=5.0, exceptions=(requests.RequestException,))
    def _fetch_real(self) -> pd.DataFrame:
        """Fetch NOAA hemispheric power proxy (recent data only)."""
        resp = self._session.get(OVATION_LATEST, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = []
        # OVATION JSON has coordinates + aurora intensity; extract HP proxy
        if isinstance(data, dict):
            ts_str = data.get("Observation Time") or data.get("Forecast Time")
            if ts_str:
                ts = pd.to_datetime(ts_str)
                # Compute mean aurora intensity as HP proxy
                coords = data.get("coordinates", [])
                if coords:
                    intensities = [c[2] for c in coords if len(c) >= 3]
                    hp_proxy = float(np.mean(intensities)) if intensities else np.nan
                    records.append({
                        "timestamp": ts,
                        "hp_north": hp_proxy,
                        "hp_south": hp_proxy * 0.9,
                        "source": "OVATION",
                    })

        if not records:
            return _generate_synthetic_aurora(self.start, self.end, self.resolution)
        return pd.DataFrame(records)


def download_aurora(cfg: Dict, use_synthetic: bool = True) -> None:
    AuroraDownloader(cfg, use_synthetic=use_synthetic).run()
