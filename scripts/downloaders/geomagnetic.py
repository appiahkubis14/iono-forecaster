"""
geomagnetic.py — IonoForecaster
Download Kp and Dst geomagnetic activity indices.

- Kp  : global geomagnetic activity (0–9), 3-hourly, GFZ Potsdam
- Dst : disturbance storm time index (nT), hourly, WDC Kyoto

Both indices are key drivers of ionospheric scintillation.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from scripts.utils import CheckpointManager, get_logger, retry

logger = get_logger("geomagnetic")

# GFZ Potsdam Kp data (open access, no auth required)
GFZ_KP_URL = (
    "https://www.gfz-potsdam.de/fileadmin/gfz/sec132/"
    "Kp_ap_Ap_SN_F107_since_1932.txt"
)
# WDC Kyoto Dst (requires year-by-year fetching)
WDC_DST_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_final/index.html"


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gfz_kp(text: str, start: date, end: date) -> pd.DataFrame:
    """
    Parse GFZ Potsdam Kp/ap file.

    File format (fixed-width, after header):
    YYYY MM DD  Kp1 Kp2 Kp3 Kp4 Kp5 Kp6 Kp7 Kp8  ap1...ap8  Ap  SN  F107  D

    Kp values are given as integers × 10 (e.g. 27 = 2.7).
    Eight 3-hourly values per day at 0, 3, 6, 9, 12, 15, 18, 21 UTC.
    """
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            d = date(year, month, day)
            if not (start <= d <= end):
                continue
            for i, hour in enumerate([0, 3, 6, 9, 12, 15, 18, 21]):
                kp_raw = int(parts[3 + i])
                kp = kp_raw / 10.0
                records.append({
                    "timestamp": datetime(year, month, day, hour),
                    "kp": kp,
                })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_dst_year(year: int, session: requests.Session) -> pd.DataFrame:
    """
    Fetch Dst index for a specific year from WDC Kyoto.

    WDC Kyoto serves Dst as plain text files at predictable URLs.
    The final product is available for years up to ~2 years ago.
    """
    # Try final product first, then provisional
    for product in ["dst_final", "dst_provisional"]:
        url = f"https://wdc.kugi.kyoto-u.ac.jp/{product}/{year}/index.html"
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return _parse_dst_html(resp.text, year)
        except Exception:
            pass
    return pd.DataFrame()


def _parse_dst_html(html: str, year: int) -> pd.DataFrame:
    """
    Extract Dst hourly values from WDC Kyoto HTML table.

    The page contains a PRE block with fixed-width data:
    line format: Dst Jan  1  YYYY  HH HH HH ... (24 hourly values)
    """
    records = []
    in_pre = False
    months = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    for line in html.splitlines():
        if "<pre>" in line.lower():
            in_pre = True
            continue
        if "</pre>" in line.lower():
            in_pre = False
            continue
        if not in_pre:
            continue

        # Expected format: "Dst Jan  1 2020  -5   3  -2 ..."
        parts = line.split()
        if len(parts) < 28 or parts[0] != "Dst":
            continue
        try:
            month_str = parts[1]
            if month_str not in months:
                continue
            month = months[month_str]
            day = int(parts[2])
            yr = int(parts[3])
            dst_vals = [int(v) for v in parts[4:28]]
            for hour, dst in enumerate(dst_vals):
                if dst == 9999:  # fill value
                    dst = np.nan
                records.append({
                    "timestamp": datetime(yr, month, day, hour),
                    "dst": float(dst),
                })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_geomagnetic(
    start: date,
    end: date,
    resolution_minutes: int = 30,
) -> pd.DataFrame:
    """
    Generate synthetic Kp and Dst indices.

    Encodes:
    - Quiet-time Kp ~ 1–2, storm-time Kp spikes to 6–9
    - Dst: quiet ~0 to -20 nT, storm main phase down to -150 nT
    - 27-day solar rotation periodicity in activity
    """
    timestamps = pd.date_range(
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()) + timedelta(days=1),
        freq=f"{resolution_minutes}min",
        inclusive="left",
    )
    n = len(timestamps)
    rng = np.random.default_rng(9999)

    # Base Kp: quiet-time log-normal
    kp_base = rng.lognormal(mean=0.3, sigma=0.5, size=n)
    kp_base = np.clip(kp_base, 0, 9)

    # Dst: quiet time
    dst = rng.normal(-10, 5, n).astype(float)

    # Storm events (~3 per month)
    n_storms = max(1, int((end - start).days / 10))
    for _ in range(n_storms):
        idx = rng.integers(0, n - 100)
        # Storm sudden commencement
        storm_dur = rng.integers(12, 48)  # 6–24 hours
        recovery_dur = rng.integers(24, 96)  # 12–48 hours
        peak_kp = rng.uniform(5, 9)
        peak_dst = rng.uniform(-50, -200)

        # Main phase
        end_idx = min(idx + storm_dur, n)
        kp_base[idx:end_idx] = np.linspace(kp_base[idx], peak_kp, end_idx - idx)
        dst[idx:end_idx] = np.linspace(dst[idx], peak_dst, end_idx - idx)

        # Recovery
        rec_end = min(end_idx + recovery_dur, n)
        kp_base[end_idx:rec_end] = np.linspace(peak_kp, 2, rec_end - end_idx)
        dst[end_idx:rec_end] = np.linspace(peak_dst, -10, rec_end - end_idx)

    kp = np.clip(kp_base + rng.normal(0, 0.3, n), 0, 9)
    dst = dst + rng.normal(0, 3, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "kp": kp.astype(np.float32),
        "dst": dst.astype(np.float32),
        "source": "synthetic",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main downloader class
# ─────────────────────────────────────────────────────────────────────────────

class GeomagneticDownloader:
    """
    Download Kp (GFZ Potsdam) and Dst (WDC Kyoto) geomagnetic indices.

    Parameters
    ----------
    cfg : dict
    use_synthetic : bool
    """

    def __init__(self, cfg: Dict, use_synthetic: bool = False) -> None:
        self.cfg = cfg
        self.use_synthetic = use_synthetic
        self.processed_dir = Path(cfg["paths"]["processed_data"]) / "geomagnetic"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.start = date.fromisoformat(cfg["temporal"]["start_date"])
        self.end = date.fromisoformat(cfg["temporal"]["end_date"])
        self.resolution = cfg["temporal"]["resolution_minutes"]
        self._session = requests.Session()

    def run(self) -> None:
        """Execute geomagnetic index download pipeline."""
        logger.info("=== Geomagnetic Indices Download (Kp + Dst) ===")

        out_path = self.processed_dir / "geomagnetic.parquet"
        if out_path.exists():
            logger.info("Geomagnetic data already exists.")
            return

        if self.use_synthetic:
            df = _generate_synthetic_geomagnetic(self.start, self.end, self.resolution)
            df.to_parquet(out_path, index=False)
            logger.info("Synthetic geomagnetic data saved: %d rows", len(df))
            return

        df_kp = self._fetch_kp()
        df_dst = self._fetch_dst()

        if df_kp.empty and df_dst.empty:
            logger.warning("No real geomagnetic data; using synthetic.")
            df = _generate_synthetic_geomagnetic(self.start, self.end, self.resolution)
        else:
            df = self._merge_and_resample(df_kp, df_dst)

        df.to_parquet(out_path, index=False)
        logger.info("Geomagnetic data saved: %d rows", len(df))

    def load(self) -> pd.DataFrame:
        """Load processed geomagnetic index data."""
        path = self.processed_dir / "geomagnetic.parquet"
        if not path.exists():
            raise FileNotFoundError("Geomagnetic data not found. Run download first.")
        return pd.read_parquet(path)

    @retry(max_attempts=3, delay=10.0, exceptions=(requests.RequestException,))
    def _fetch_kp(self) -> pd.DataFrame:
        """Download and parse GFZ Kp file."""
        try:
            logger.info("Downloading GFZ Kp index…")
            resp = self._session.get(GFZ_KP_URL, timeout=120)
            resp.raise_for_status()
            return _parse_gfz_kp(resp.text, self.start, self.end)
        except Exception as exc:
            logger.error("Kp download failed: %s", exc)
            return pd.DataFrame()

    def _fetch_dst(self) -> pd.DataFrame:
        """Download Dst year by year from WDC Kyoto."""
        dfs = []
        for year in range(self.start.year, self.end.year + 1):
            logger.info("Fetching Dst for %d…", year)
            df_year = _fetch_dst_year(year, self._session)
            if not df_year.empty:
                dfs.append(df_year)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _merge_and_resample(
        self, df_kp: pd.DataFrame, df_dst: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge Kp and Dst onto common time grid at target resolution."""
        target_index = pd.date_range(
            start=datetime.combine(self.start, datetime.min.time()),
            end=datetime.combine(self.end, datetime.min.time()) + timedelta(days=1),
            freq=f"{self.resolution}min",
            inclusive="left",
        )
        result = pd.DataFrame({"timestamp": target_index})

        if not df_kp.empty:
            df_kp = df_kp.set_index("timestamp").sort_index()
            kp_resampled = df_kp["kp"].reindex(target_index, method="ffill", tolerance="3h")
            result["kp"] = kp_resampled.values

        if not df_dst.empty:
            df_dst = df_dst.set_index("timestamp").sort_index()
            dst_resampled = df_dst["dst"].reindex(target_index, method="ffill", tolerance="1h")
            result["dst"] = dst_resampled.values

        result["source"] = "real"
        return result


def download_geomagnetic(cfg: Dict, use_synthetic: bool = True) -> None:
    """Entry point called by main.py --step download."""
    GeomagneticDownloader(cfg, use_synthetic=use_synthetic).run()
