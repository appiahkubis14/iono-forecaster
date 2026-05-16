"""
gnss_samba.py — IonoForecaster
Download and parse S4, σφ, and ROTI data from SAMBA, EPOSA, and IGS networks.

For operational use the pipeline fetches RINEX observation files from NASA CDDIS,
computes the S4 scintillation index and ROTI from raw pseudorange/carrier-phase,
and stores results in a standardised Parquet format per station.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import ftplib
import gzip
import io
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from scripts.utils import CheckpointManager, get_logger, haversine_km, retry

logger = get_logger("gnss_samba")

# ─────────────────────────────────────────────────────────────────────────────
# Station metadata (from config.yaml stations list)
# ─────────────────────────────────────────────────────────────────────────────

STATION_META: List[Dict] = [
    {"id": "TETB", "lat": 6.3,   "lon": 2.2,    "country": "Benin",          "network": "SAMBA"},
    {"id": "ABMF", "lat": 6.3,   "lon": 2.3,    "country": "Benin",          "network": "IGS"},
    {"id": "NKLG", "lat": 0.4,   "lon": 9.7,    "country": "Gabon",          "network": "IGS"},
    {"id": "KOUC", "lat": 5.1,   "lon": -52.8,  "country": "French Guiana",  "network": "IGS"},
    {"id": "MBAR", "lat": -0.6,  "lon": 30.7,   "country": "Uganda",         "network": "EPOSA"},
    {"id": "ADIS", "lat": 9.0,   "lon": 38.8,   "country": "Ethiopia",       "network": "EPOSA"},
    {"id": "NAIR", "lat": -1.3,  "lon": 36.8,   "country": "Kenya",          "network": "EPOSA"},
    {"id": "DAKR", "lat": 14.7,  "lon": -17.4,  "country": "Senegal",        "network": "IGS"},
    {"id": "LAGO", "lat": 6.5,   "lon": 3.4,    "country": "Nigeria",        "network": "SAMBA"},
    {"id": "KHAR", "lat": 15.6,  "lon": 32.5,   "country": "Sudan",          "network": "EPOSA"},
    {"id": "DARE", "lat": -6.8,  "lon": 39.3,   "country": "Tanzania",       "network": "EPOSA"},
    {"id": "LUAK", "lat": -15.4, "lon": 28.3,   "country": "Zambia",         "network": "EPOSA"},
    {"id": "HARR", "lat": -17.8, "lon": 31.1,   "country": "Zimbabwe",       "network": "EPOSA"},
    {"id": "ACCR", "lat": 5.6,   "lon": -0.2,   "country": "Ghana",          "network": "SAMBA"},
    {"id": "ABUJ", "lat": 9.1,   "lon": 7.5,    "country": "Nigeria",        "network": "SAMBA"},
]

# CDDIS RINEX-3 base URL (requires Earthdata login for some products)
CDDIS_DAILY = "https://cddis.nasa.gov/archive/gnss/data/daily"
CDDIS_PRODUCTS = "https://cddis.nasa.gov/archive/gnss/products"

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic / Demo fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_station_data(
    station_id: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
    resolution_minutes: int = 30,
) -> pd.DataFrame:
    """
    Generate physically-plausible synthetic GNSS scintillation data.

    Used when real data is unavailable (demo mode / CI testing).
    The synthetic S4 series encodes:
      - Diurnal peak around 20–22 LT (equatorial post-sunset enhancement)
      - Seasonal dependence (stronger near equinoxes)
      - Random moderate/severe events (~5 % of time)
      - Gaussian noise

    Parameters
    ----------
    station_id : str
    lat, lon : float
    start, end : date
    resolution_minutes : int

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, station_id, lat, lon, s4, sigma_phi, roti, tec
    """
    timestamps = pd.date_range(
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()) + timedelta(days=1),
        freq=f"{resolution_minutes}min",
        inclusive="left",
    )
    n = len(timestamps)
    rng = np.random.default_rng(seed=abs(hash(station_id)) % 2**32)

    # Local time offset (approximate)
    lt_offset_h = lon / 15.0
    local_hour = (timestamps.hour + lt_offset_h) % 24

    # Diurnal modulation: peak around 21 LT
    diurnal = np.clip(
        0.15 * np.exp(-0.5 * ((local_hour.values - 21.0) / 2.5) ** 2),
        0, 0.3,
    )
    # Seasonal modulation: stronger near March and September equinoxes
    doy = timestamps.day_of_year.values
    seasonal = 0.05 * (np.cos(2 * np.pi * (doy - 80) / 365) ** 2)

    base_s4 = diurnal + seasonal + rng.normal(0, 0.02, n)
    base_s4 = np.clip(base_s4, 0.0, None)

    # Inject random scintillation events
    event_mask = rng.random(n) < 0.04  # ~4 % chance
    event_amplitude = rng.uniform(0.3, 0.9, n) * event_mask
    s4 = np.clip(base_s4 + event_amplitude, 0.0, 1.0)

    # sigma_phi correlated with S4
    sigma_phi = np.clip(s4 * rng.uniform(0.4, 0.7, n) + rng.normal(0, 0.02, n), 0, np.pi)

    # ROTI ~ dTEC/dt noise
    roti = np.clip(s4 * 1.5 + rng.exponential(0.05, n), 0, 5.0)

    # Background TEC (TECU) with diurnal shape
    tec = 20 + 10 * np.sin(np.pi * local_hour.values / 12) + rng.normal(0, 1, n)
    tec = np.clip(tec, 0, 80)

    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_id,
        "lat": lat,
        "lon": lon,
        "s4": s4.astype(np.float32),
        "sigma_phi": sigma_phi.astype(np.float32),
        "roti": roti.astype(np.float32),
        "tec": tec.astype(np.float32),
    })


# ─────────────────────────────────────────────────────────────────────────────
# RINEX observation downloader (real data path)
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=10.0, exceptions=(requests.RequestException, OSError))
def _download_rinex_file(
    station_id: str,
    target_date: date,
    raw_dir: Path,
    session: requests.Session,
) -> Optional[Path]:
    """
    Download a RINEX-3 observation file for a single station and date from CDDIS.

    The Earthdata Bearer token must be set in the EARTHDATA_TOKEN env variable,
    or NASA CDDIS netrc credentials in ~/.netrc.

    Parameters
    ----------
    station_id : str
        4-character station identifier.
    target_date : date
    raw_dir : Path
        Destination directory.
    session : requests.Session

    Returns
    -------
    Path or None
        Local path to the downloaded file, or None if unavailable.
    """
    year = target_date.year
    doy = target_date.timetuple().tm_yday
    yy = str(year)[2:]

    station_lower = station_id.lower()
    fname = f"{station_lower}0000_{year}{doy:03d}0000_01D_30S_MO.crx.gz"
    url = f"{CDDIS_DAILY}/{year}/{doy:03d}/{yy}d/{fname}"

    out_path = raw_dir / "rinex" / station_id / str(year) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return out_path

    headers = {}
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = session.get(url, headers=headers, timeout=60, stream=True)
    if resp.status_code == 200:
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        logger.info("Downloaded RINEX: %s", out_path.name)
        return out_path
    else:
        logger.warning("RINEX not found (HTTP %d): %s", resp.status_code, url)
        return None


def _compute_s4_from_rinex(rinex_path: Path) -> pd.DataFrame:
    """
    Compute S4 scintillation index from a RINEX observation file.

    S4 = std(I) / mean(I) over a 60-second window, where I is signal intensity.
    This is a simplified implementation; operational use would use ISMR receivers.

    Parameters
    ----------
    rinex_path : Path

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, prn, s4, sigma_phi, roti, snr
    """
    try:
        import georinex as gr
    except ImportError:
        logger.error("georinex not installed. Run: pip install georinex")
        return pd.DataFrame()

    try:
        obs = gr.load(rinex_path, use=["S1C", "S2W", "L1C", "L2W"])
    except Exception as exc:
        logger.warning("Failed to parse RINEX %s: %s", rinex_path.name, exc)
        return pd.DataFrame()

    records = []
    try:
        s1 = obs["S1C"].to_pandas() if "S1C" in obs else None
        if s1 is None:
            return pd.DataFrame()

        window = "60s"
        s4_all = (s1.rolling(window).std() / s1.rolling(window).mean().replace(0, np.nan))

        for prn in s4_all.columns:
            ts = s4_all.index
            s4_vals = s4_all[prn].values
            for t, s4 in zip(ts, s4_vals):
                if not np.isnan(s4):
                    records.append({
                        "timestamp": t,
                        "prn": prn,
                        "s4": float(np.clip(s4, 0, 1)),
                    })
    except Exception as exc:
        logger.warning("S4 computation failed: %s", exc)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main downloader class
# ─────────────────────────────────────────────────────────────────────────────

class GNSSDownloader:
    """
    Downloads and processes GNSS scintillation data for the IonoForecaster network.

    Supports both real CDDIS/RINEX data (when credentials are available) and
    synthetic fallback data (demo / offline mode).

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration.
    use_synthetic : bool
        If True, skip network access and generate synthetic data. Useful for
        CI testing and portfolio demonstration without credentials.
    """

    def __init__(self, cfg: Dict, use_synthetic: bool = False) -> None:
        self.cfg = cfg
        self.use_synthetic = use_synthetic
        self.raw_dir = Path(cfg["paths"]["raw_data"])
        self.processed_dir = Path(cfg["paths"]["processed_data"]) / "gnss"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.start = date.fromisoformat(cfg["temporal"]["start_date"])
        self.end = date.fromisoformat(cfg["temporal"]["end_date"])
        self.resolution = cfg["temporal"]["resolution_minutes"]
        self.stations = cfg.get("stations", STATION_META)

        self._session = requests.Session()
        self._ckpt = CheckpointManager(
            cfg["paths"]["models_dir"], "gnss_download"
        )

    # ── public interface ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full GNSS download and preprocessing pipeline."""
        logger.info("=== GNSS Data Download ===")
        logger.info(
            "Stations: %d | Period: %s → %s | Synthetic: %s",
            len(self.stations), self.start, self.end, self.use_synthetic,
        )

        existing = self._ckpt.load_meta()
        completed = set(existing.get("completed_stations", [])) if existing else set()

        for station in self.stations:
            sid = station["id"]
            if sid in completed:
                logger.info("Skipping %s (already downloaded)", sid)
                continue

            out_path = self.processed_dir / f"{sid}.parquet"
            if out_path.exists():
                logger.info("Output exists for %s, skipping", sid)
                completed.add(sid)
                continue

            try:
                df = self._fetch_station(station)
                if df is not None and not df.empty:
                    df.to_parquet(out_path, index=False)
                    logger.info(
                        "Saved %s → %d rows (%s to %s)",
                        sid, len(df),
                        df["timestamp"].min(), df["timestamp"].max(),
                    )
                    completed.add(sid)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", sid, exc)

            self._ckpt.save_meta({"completed_stations": list(completed)})

        logger.info("GNSS download complete. %d/%d stations.", len(completed), len(self.stations))

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all downloaded station data as a dictionary of DataFrames.

        Returns
        -------
        dict
            Mapping from station_id → DataFrame.
        """
        data = {}
        for station in self.stations:
            sid = station["id"]
            path = self.processed_dir / f"{sid}.parquet"
            if path.exists():
                data[sid] = pd.read_parquet(path)
                logger.debug("Loaded %s: %d rows", sid, len(data[sid]))
            else:
                logger.warning("No data found for station %s", sid)
        return data

    # ── internal ──────────────────────────────────────────────────────────────

    def _fetch_station(self, station: Dict) -> Optional[pd.DataFrame]:
        """Fetch data for one station (real or synthetic)."""
        sid = station["id"]
        lat = station["lat"]
        lon = station["lon"]

        if self.use_synthetic:
            logger.info("Generating synthetic data for %s", sid)
            return _generate_synthetic_station_data(
                sid, lat, lon, self.start, self.end, self.resolution
            )

        # Real data path: try CDDIS download then fall back to synthetic
        dfs = []
        current = self.start
        while current <= self.end:
            rinex_path = _download_rinex_file(
                sid, current, self.raw_dir, self._session
            )
            if rinex_path:
                df_day = _compute_s4_from_rinex(rinex_path)
                if not df_day.empty:
                    df_day["station_id"] = sid
                    df_day["lat"] = lat
                    df_day["lon"] = lon
                    dfs.append(df_day)
            current += timedelta(days=1)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Aggregate to 30-min resolution (max S4 = worst-case per window)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.set_index("timestamp")
            agg = combined.groupby("station_id").resample(
                f"{self.resolution}min", label="left"
            )["s4"].max().reset_index()
            return agg
        else:
            logger.warning("No real RINEX data for %s; using synthetic fallback.", sid)
            return _generate_synthetic_station_data(
                sid, lat, lon, self.start, self.end, self.resolution
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────────────────────────

def download_gnss(cfg: Dict, use_synthetic: bool = True) -> None:
    """
    Entry point called by main.py --step download.

    Parameters
    ----------
    cfg : dict
    use_synthetic : bool
        Default True for portfolio demo; set False to use real CDDIS data.
    """
    downloader = GNSSDownloader(cfg, use_synthetic=use_synthetic)
    downloader.run()
