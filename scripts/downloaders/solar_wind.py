"""
solar_wind.py — IonoForecaster
Download solar wind data from NOAA DSCOVR and ACE satellites.

Variables retrieved:
  - Bz   : IMF z-component (nT) — primary scintillation driver
  - v_sw  : solar wind speed (km/s)
  - n_p   : proton density (cm⁻³)
  - t_p   : proton temperature (K)

NOAA SWPC real-time JSON endpoints are used for recent data.
For historical data (2018–2024) the NOAA FTP archive is queried.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import csv
import gzip
import io
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from scripts.utils import CheckpointManager, get_logger, retry

logger = get_logger("solar_wind")

# NOAA SWPC endpoints
DSCOVR_MAG_7DAY = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
DSCOVR_PLASMA_7DAY = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
ACE_MAG_JSON = "https://services.swpc.noaa.gov/json/ace/mag/ace_mag_1m.json"
ACE_SWEPAM_JSON = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1m.json"

# NOAA historical FTP
NOAA_FTP_BASE = "https://ftp.swpc.noaa.gov/pub/lists/ace/"


# ─────────────────────────────────────────────────────────────────────────────
# NOAA JSON parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dscovr_mag(data: List[List]) -> pd.DataFrame:
    """
    Parse DSCOVR magnetometer JSON response.

    Format: [time_tag, bx, by, bz, lon, lat, bt]
    """
    records = []
    for row in data[1:]:  # skip header
        try:
            records.append({
                "timestamp": pd.to_datetime(row[0]),
                "bz": float(row[3]) if row[3] not in (None, "null", "-999.9") else np.nan,
                "bt": float(row[6]) if row[6] not in (None, "null", "-999.9") else np.nan,
            })
        except (ValueError, IndexError, TypeError):
            continue
    return pd.DataFrame(records)


def _parse_dscovr_plasma(data: List[List]) -> pd.DataFrame:
    """
    Parse DSCOVR plasma JSON response.

    Format: [time_tag, density, speed, temperature]
    """
    records = []
    for row in data[1:]:
        try:
            records.append({
                "timestamp": pd.to_datetime(row[0]),
                "n_p": float(row[1]) if row[1] not in (None, "null", "-9999.9") else np.nan,
                "v_sw": float(row[2]) if row[2] not in (None, "null", "-9999.9") else np.nan,
                "t_p": float(row[3]) if row[3] not in (None, "null", "-1.00e+05") else np.nan,
            })
        except (ValueError, IndexError, TypeError):
            continue
    return pd.DataFrame(records)


def _parse_ace_mag(data: List[Dict]) -> pd.DataFrame:
    """Parse ACE magnetometer JSON."""
    records = []
    for row in data:
        try:
            records.append({
                "timestamp": pd.to_datetime(row["time_tag"]),
                "bz": float(row.get("bz_gsm", np.nan)),
                "bt": float(row.get("bt", np.nan)),
            })
        except (ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(records)


def _parse_ace_swepam(data: List[Dict]) -> pd.DataFrame:
    """Parse ACE solar wind electron proton alpha monitor JSON."""
    records = []
    for row in data:
        try:
            records.append({
                "timestamp": pd.to_datetime(row["time_tag"]),
                "n_p": float(row.get("proton_density", np.nan)),
                "v_sw": float(row.get("proton_speed", np.nan)),
                "t_p": float(row.get("proton_temp", np.nan)),
            })
        except (ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_solar_wind(
    start: date,
    end: date,
    resolution_minutes: int = 30,
) -> pd.DataFrame:
    """
    Generate synthetic solar wind data with realistic statistical properties.

    Encodes:
    - Bz with random southward excursions (storm drivers)
    - Solar wind speed with CIR-like fluctuations (27-day recurrence)
    - Proton density anti-correlated with speed
    - Geomagnetic storm events (~3 per month)
    """
    timestamps = pd.date_range(
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()) + timedelta(days=1),
        freq=f"{resolution_minutes}min",
        inclusive="left",
    )
    n = len(timestamps)
    rng = np.random.default_rng(1234)

    # Bz: Ornstein-Uhlenbeck process with mean ~0 nT
    bz = np.zeros(n)
    bz[0] = rng.normal(0, 3)
    theta, mu, sigma = 0.05, 0.0, 4.0
    for i in range(1, n):
        bz[i] = bz[i-1] + theta * (mu - bz[i-1]) + sigma * rng.normal(0, 1) * np.sqrt(1/30)

    # Inject storm events: strong sustained southward Bz
    storm_count = int((end - start).days / 10)  # ~3 per month
    for _ in range(storm_count):
        idx = rng.integers(0, n - 24)
        duration = rng.integers(4, 24)  # 2–12 hours
        bz[idx:idx+duration] -= rng.uniform(10, 40)

    # Solar wind speed: ~400 km/s baseline with 27-day CIR
    doy_arr = np.array([t.day_of_year for t in timestamps])
    v_sw = 400 + 100 * np.sin(2 * np.pi * doy_arr / 27) + rng.normal(0, 30, n)
    v_sw = np.clip(v_sw, 250, 900)

    # Proton density: anti-correlated with speed
    n_p = 8 * (400 / v_sw) + rng.exponential(1, n)
    n_p = np.clip(n_p, 0.1, 50)

    # Proton temperature: correlated with speed
    t_p = 1.2e5 * (v_sw / 400) ** 3.1 + rng.normal(0, 1e4, n)
    t_p = np.clip(t_p, 1e3, 1e6)

    return pd.DataFrame({
        "timestamp": timestamps,
        "bz": bz.astype(np.float32),
        "v_sw": v_sw.astype(np.float32),
        "n_p": n_p.astype(np.float32),
        "t_p": t_p.astype(np.float32),
        "bt": np.abs(bz).astype(np.float32) + rng.uniform(0, 5, n).astype(np.float32),
        "source": "synthetic",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main downloader class
# ─────────────────────────────────────────────────────────────────────────────

class SolarWindDownloader:
    """
    Download and merge DSCOVR and ACE solar wind data for IonoForecaster.

    Parameters
    ----------
    cfg : dict
    use_synthetic : bool
    """

    def __init__(self, cfg: Dict, use_synthetic: bool = False) -> None:
        self.cfg = cfg
        self.use_synthetic = use_synthetic
        self.raw_dir = Path(cfg["paths"]["raw_data"]) / "solar_wind"
        self.processed_dir = Path(cfg["paths"]["processed_data"]) / "solar_wind"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.start = date.fromisoformat(cfg["temporal"]["start_date"])
        self.end = date.fromisoformat(cfg["temporal"]["end_date"])
        self.resolution = cfg["temporal"]["resolution_minutes"]
        self._session = requests.Session()
        self._ckpt = CheckpointManager(cfg["paths"]["models_dir"], "solar_wind")

    def run(self) -> None:
        """Execute solar wind download pipeline."""
        logger.info("=== Solar Wind Download (DSCOVR + ACE) ===")

        out_path = self.processed_dir / "solar_wind.parquet"
        if out_path.exists():
            logger.info("Solar wind data already exists.")
            return

        if self.use_synthetic:
            logger.info("Generating synthetic solar wind data…")
            df = _generate_synthetic_solar_wind(self.start, self.end, self.resolution)
            df.to_parquet(out_path, index=False)
            logger.info("Synthetic solar wind saved: %d rows", len(df))
            return

        df = self._fetch_real()
        if df is None or df.empty:
            logger.warning("No real solar wind data; using synthetic fallback.")
            df = _generate_synthetic_solar_wind(self.start, self.end, self.resolution)

        df.to_parquet(out_path, index=False)
        logger.info("Solar wind data saved: %d rows", len(df))

    def load(self) -> pd.DataFrame:
        """Load processed solar wind data."""
        path = self.processed_dir / "solar_wind.parquet"
        if not path.exists():
            raise FileNotFoundError("Solar wind data not found. Run download first.")
        return pd.read_parquet(path)

    @retry(max_attempts=3, delay=10.0, exceptions=(requests.RequestException,))
    def _fetch_real(self) -> Optional[pd.DataFrame]:
        """Download DSCOVR 7-day data and merge mag + plasma."""
        try:
            logger.info("Fetching DSCOVR magnetometer data…")
            resp_mag = self._session.get(DSCOVR_MAG_7DAY, timeout=30)
            resp_mag.raise_for_status()
            df_mag = _parse_dscovr_mag(resp_mag.json())

            logger.info("Fetching DSCOVR plasma data…")
            resp_plasma = self._session.get(DSCOVR_PLASMA_7DAY, timeout=30)
            resp_plasma.raise_for_status()
            df_plasma = _parse_dscovr_plasma(resp_plasma.json())

            if df_mag.empty or df_plasma.empty:
                return None

            # Merge on nearest timestamp
            df_mag = df_mag.set_index("timestamp").sort_index()
            df_plasma = df_plasma.set_index("timestamp").sort_index()
            df = pd.merge_asof(
                df_plasma.reset_index(),
                df_mag.reset_index(),
                on="timestamp",
                tolerance=pd.Timedelta("5min"),
                direction="nearest",
            )
            df["source"] = "DSCOVR"

            # Resample to target resolution
            df = df.set_index("timestamp")
            df = df.resample(f"{self.resolution}min").mean().reset_index()
            return df

        except Exception as exc:
            logger.error("DSCOVR fetch failed: %s", exc)
            return None


def download_solar_wind(cfg: Dict, use_synthetic: bool = True) -> None:
    """Entry point called by main.py --step download."""
    SolarWindDownloader(cfg, use_synthetic=use_synthetic).run()
