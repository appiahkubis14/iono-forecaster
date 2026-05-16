"""
igs_gim.py — IonoForecaster
Download and parse IGS Global Ionosphere Maps (GIM) in IONEX format from NASA CDDIS.

IGS GIM provides global TEC maps on a 2.5° lat × 5° lon grid at 1-hour resolution.
For equatorial Africa the relevant latitude band is roughly -20° to +20°.

IONEX format reference: ftp://igs.org/pub/data/format/ionex1.pdf

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import gzip
import io
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from scripts.utils import CheckpointManager, get_logger, retry

logger = get_logger("igs_gim")

# CDDIS IONEX archive — requires Earthdata account (env: EARTHDATA_TOKEN)
CDDIS_IONEX_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"
# IGS combined product code
IGS_PRODUCT = "igsg"  # IGS final combined GIM

# Equatorial Africa bounding box
LAT_MIN, LAT_MAX = -20.0, 20.0
LON_MIN, LON_MAX = -30.0, 60.0


# ─────────────────────────────────────────────────────────────────────────────
# IONEX parser
# ─────────────────────────────────────────────────────────────────────────────

class IONEXParser:
    """
    Parse an IONEX file and return TEC maps as an xarray-compatible structure.

    The IONEX format stores TEC maps as blocks of integers (TEC × 10).
    """

    def __init__(self, content: str) -> None:
        self.content = content
        self._header: Dict = {}
        self.tec_maps: List[Dict] = []

    def parse(self) -> "IONEXParser":
        """Parse header and all TEC maps."""
        lines = self.content.splitlines()
        self._parse_header(lines)
        self._parse_maps(lines)
        return self

    def _parse_header(self, lines: List[str]) -> None:
        """Extract IONEX header metadata."""
        for line in lines:
            if "EPOCH OF FIRST MAP" in line:
                parts = line.split()[:6]
                self._header["epoch_first"] = datetime(
                    int(parts[0]), int(parts[1]), int(parts[2]),
                    int(parts[3]), int(parts[4]), int(parts[5]),
                )
            elif "EPOCH OF LAST MAP" in line:
                parts = line.split()[:6]
                self._header["epoch_last"] = datetime(
                    int(parts[0]), int(parts[1]), int(parts[2]),
                    int(parts[3]), int(parts[4]), int(parts[5]),
                )
            elif "INTERVAL" in line and "MAP" not in line:
                self._header["interval_s"] = int(line.split()[0])
            elif "EXPONENT" in line:
                self._header["exponent"] = int(line.split()[0])
            elif "END OF HEADER" in line:
                break

        self._header.setdefault("exponent", -1)  # TEC × 10^-1 = TECU

    def _parse_maps(self, lines: List[str]) -> None:
        """Extract individual TEC epoch maps."""
        exponent = self._header.get("exponent", -1)
        scale = 10 ** exponent  # TECU = raw * scale

        in_map = False
        current_epoch: Optional[datetime] = None
        current_lat: Optional[float] = None
        lat_data: Dict[float, List[float]] = {}
        lon_range: Optional[Tuple[float, float, float]] = None  # start, end, step

        for line in lines:
            if "START OF TEC MAP" in line:
                in_map = True
                lat_data = {}
                current_epoch = None
                lon_range = None

            elif "END OF TEC MAP" in line and in_map:
                if current_epoch and lat_data:
                    self.tec_maps.append({
                        "epoch": current_epoch,
                        "lat_data": dict(lat_data),
                        "lon_range": lon_range,
                    })
                in_map = False

            elif in_map:
                if "EPOCH OF CURRENT MAP" in line:
                    parts = line.split()[:6]
                    current_epoch = datetime(
                        int(parts[0]), int(parts[1]), int(parts[2]),
                        int(parts[3]), int(parts[4]), int(parts[5]),
                    )
                elif "LAT/LON1/LON2/DLON/H" in line:
                    parts = line.split()
                    current_lat = float(parts[0])
                    lon_start = float(parts[1])
                    lon_end = float(parts[2])
                    dlon = float(parts[3])
                    lon_range = (lon_start, lon_end, dlon)
                    lat_data[current_lat] = []
                elif current_lat is not None and in_map:
                    # TEC values — may span multiple lines
                    raw_vals = line.strip().split()
                    # Skip header-like lines
                    try:
                        vals = [int(v) * scale for v in raw_vals]
                        lat_data[current_lat].extend(vals)
                    except ValueError:
                        pass  # continuation of non-numeric line

    def to_dataframe(
        self,
        lat_min: float = LAT_MIN,
        lat_max: float = LAT_MAX,
        lon_min: float = LON_MIN,
        lon_max: float = LON_MAX,
    ) -> pd.DataFrame:
        """
        Convert parsed TEC maps to a long-format DataFrame, clipped to AOI.

        Returns
        -------
        pd.DataFrame
            Columns: epoch, lat, lon, tec (TECU)
        """
        records = []
        for tmap in self.tec_maps:
            epoch = tmap["epoch"]
            lon_range = tmap["lon_range"]
            if lon_range is None:
                continue

            lon_start, lon_end, dlon = lon_range
            lons = np.arange(lon_start, lon_end + dlon / 2, dlon)

            for lat, tec_vals in tmap["lat_data"].items():
                if not (lat_min <= lat <= lat_max):
                    continue
                if len(tec_vals) != len(lons):
                    continue
                for lon, tec in zip(lons, tec_vals):
                    if lon_min <= lon <= lon_max:
                        records.append({
                            "epoch": epoch,
                            "lat": float(lat),
                            "lon": float(lon),
                            "tec": float(tec),
                        })

        if not records:
            return pd.DataFrame(columns=["epoch", "lat", "lon", "tec"])
        df = pd.DataFrame(records)
        df["epoch"] = pd.to_datetime(df["epoch"])
        return df.sort_values(["epoch", "lat", "lon"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# GIM downloader
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=15.0, exceptions=(requests.RequestException, OSError))
def _fetch_ionex_day(
    target_date: date,
    raw_dir: Path,
    session: requests.Session,
) -> Optional[Path]:
    """
    Download the IGS combined IONEX file for a given date from CDDIS.

    Parameters
    ----------
    target_date : date
    raw_dir : Path
    session : requests.Session

    Returns
    -------
    Path or None
    """
    year = target_date.year
    doy = target_date.timetuple().tm_yday
    yy = str(year)[2:]

    # IONEX filename: igsgDDD0.YYi.Z  (old) or igsg0opsDDD0_YYYY_01D.INX.gz (new)
    fname_new = f"{IGS_PRODUCT}0ops{doy:03d}0_{year}_01D_01H_GIM.INX.gz"
    fname_old = f"{IGS_PRODUCT}{doy:03d}0.{yy}i.Z"

    out_dir = raw_dir / "ionex" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname_new

    if out_path.exists():
        return out_path

    headers = {}
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Try new filename format first
    url = f"{CDDIS_IONEX_BASE}/{year}/{doy:03d}/{fname_new}"
    resp = session.get(url, headers=headers, timeout=120, stream=True)

    if resp.status_code != 200:
        # Fall back to old format
        url = f"{CDDIS_IONEX_BASE}/{year}/{doy:03d}/{fname_old}"
        out_path = out_dir / fname_old
        resp = session.get(url, headers=headers, timeout=120, stream=True)

    if resp.status_code == 200:
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        logger.info("Downloaded IONEX: %s", out_path.name)
        return out_path
    else:
        logger.warning("IONEX not available (HTTP %d): %s", resp.status_code, url)
        return None


def _read_ionex(path: Path) -> str:
    """Read IONEX file (handles .gz and .Z compression)."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="latin-1", errors="replace") as fh:
            return fh.read()
    elif suffix == ".z":
        # .Z (compress) — use subprocess
        import subprocess
        result = subprocess.run(["zcat", str(path)], capture_output=True)
        if result.returncode == 0:
            return result.stdout.decode("latin-1", errors="replace")
        # Fallback: try gzip
        try:
            with gzip.open(path, "rt", encoding="latin-1") as fh:
                return fh.read()
        except Exception:
            logger.error("Cannot decompress %s", path)
            return ""
    else:
        with open(path, encoding="latin-1", errors="replace") as fh:
            return fh.read()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic GIM fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_gim(
    start: date,
    end: date,
    resolution_minutes: int = 60,
) -> pd.DataFrame:
    """
    Generate synthetic TEC maps for the equatorial Africa region.

    Encodes:
    - Equatorial ionisation anomaly (EIA) — dual peaks at ±15° latitude
    - Diurnal variation (peak ~14 LT)
    - Seasonal modulation
    - Gaussian spatial structure

    Returns
    -------
    pd.DataFrame
        Columns: epoch, lat, lon, tec
    """
    lats = np.arange(LAT_MIN, LAT_MAX + 2.5, 2.5)
    lons = np.arange(LON_MIN, LON_MAX + 5.0, 5.0)
    epochs = pd.date_range(
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()) + timedelta(days=1),
        freq=f"{resolution_minutes}min",
        inclusive="left",
    )

    records = []
    rng = np.random.default_rng(2024)

    for epoch in epochs:
        lt_offset = lons / 15.0
        hour_ut = epoch.hour + epoch.minute / 60.0
        doy = epoch.day_of_year

        for lat in lats:
            for lon, lt_off in zip(lons, lt_offset):
                local_time = (hour_ut + lt_off) % 24
                # Diurnal: peak at 14 LT
                diurnal = 20 + 15 * np.exp(-0.5 * ((local_time - 14) / 3) ** 2)
                # EIA: dual peaks at ±12°
                eia = 8 * np.exp(-0.5 * ((abs(lat) - 12) / 5) ** 2)
                # Seasonal: stronger March equinox
                seasonal = 5 * np.cos(2 * np.pi * (doy - 80) / 365)
                tec = max(0, diurnal + eia + seasonal + rng.normal(0, 1))
                records.append({
                    "epoch": epoch,
                    "lat": float(lat),
                    "lon": float(lon),
                    "tec": float(tec),
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main GIM downloader class
# ─────────────────────────────────────────────────────────────────────────────

class IGSGIMDownloader:
    """
    Download and process IGS Global Ionosphere Maps for equatorial Africa.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration.
    use_synthetic : bool
        Use synthetic GIM data (demo / offline mode).
    """

    def __init__(self, cfg: Dict, use_synthetic: bool = False) -> None:
        self.cfg = cfg
        self.use_synthetic = use_synthetic
        self.raw_dir = Path(cfg["paths"]["raw_data"])
        self.processed_dir = Path(cfg["paths"]["processed_data"]) / "gim"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.start = date.fromisoformat(cfg["temporal"]["start_date"])
        self.end = date.fromisoformat(cfg["temporal"]["end_date"])
        self._session = requests.Session()
        self._ckpt = CheckpointManager(cfg["paths"]["models_dir"], "gim_download")

    # ── public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute GIM download pipeline."""
        logger.info("=== IGS GIM Download ===")

        out_path = self.processed_dir / "tec_maps.parquet"
        if out_path.exists():
            logger.info("GIM data already exists: %s", out_path)
            return

        if self.use_synthetic:
            logger.info("Generating synthetic GIM data…")
            df = _generate_synthetic_gim(self.start, self.end)
            df.to_parquet(out_path, index=False)
            logger.info("Synthetic GIM saved: %d rows", len(df))
            return

        # Real data: download day by day
        existing = self._ckpt.load_meta()
        completed_days = set(existing.get("completed_days", [])) if existing else set()

        all_dfs: List[pd.DataFrame] = []
        current = self.start
        while current <= self.end:
            day_str = current.isoformat()
            if day_str not in completed_days:
                ionex_path = _fetch_ionex_day(current, self.raw_dir, self._session)
                if ionex_path:
                    try:
                        content = _read_ionex(ionex_path)
                        parser = IONEXParser(content).parse()
                        df_day = parser.to_dataframe()
                        if not df_day.empty:
                            all_dfs.append(df_day)
                            completed_days.add(day_str)
                    except Exception as exc:
                        logger.error("IONEX parse error for %s: %s", day_str, exc)
            current += timedelta(days=1)

            # Flush every 30 days
            if len(all_dfs) >= 30:
                self._flush(all_dfs, out_path)
                all_dfs = []
                self._ckpt.save_meta({"completed_days": list(completed_days)})

        if all_dfs:
            self._flush(all_dfs, out_path)

        if not out_path.exists():
            logger.warning("No real GIM data obtained; using synthetic fallback.")
            df = _generate_synthetic_gim(self.start, self.end)
            df.to_parquet(out_path, index=False)

        logger.info("GIM download complete.")

    def load(self) -> pd.DataFrame:
        """Load processed TEC map data."""
        path = self.processed_dir / "tec_maps.parquet"
        if not path.exists():
            raise FileNotFoundError(f"GIM data not found at {path}. Run download first.")
        df = pd.read_parquet(path)
        logger.info("Loaded GIM: %d rows", len(df))
        return df

    # ── internal ──────────────────────────────────────────────────────────────

    def _flush(self, dfs: List[pd.DataFrame], out_path: Path) -> None:
        """Append new DataFrames to the output Parquet file."""
        new_df = pd.concat(dfs, ignore_index=True)
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["epoch", "lat", "lon"], inplace=True)
        else:
            combined = new_df
        combined.sort_values(["epoch", "lat", "lon"]).to_parquet(out_path, index=False)
        logger.info("Flushed GIM data → %d rows total", len(combined))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def download_gim(cfg: Dict, use_synthetic: bool = True) -> None:
    """Entry point called by main.py --step download."""
    downloader = IGSGIMDownloader(cfg, use_synthetic=use_synthetic)
    downloader.run()
