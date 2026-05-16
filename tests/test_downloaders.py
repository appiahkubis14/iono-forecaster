"""
tests/test_downloaders.py — IonoForecaster
Unit tests for synthetic data downloaders.

Author: Samuel Appiah Kubi
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
import tempfile


# ── Minimal config fixture ────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path):
    return {
        "temporal": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-07",
            "resolution_minutes": 30,
            "train_end": "2023-01-05",
            "val_end": "2023-01-06",
        },
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "processed_data": str(tmp_path / "processed"),
            "features_data": str(tmp_path / "features"),
            "graphs_data": str(tmp_path / "graphs"),
            "models_dir": str(tmp_path / "models"),
            "outputs_dir": str(tmp_path / "outputs"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "stations": [
            {"id": "ACCR", "lat": 5.56, "lon": -0.20, "country": "Ghana", "network": "SAGAING", "alt_m": 73},
            {"id": "LAGO", "lat": 6.52, "lon":  3.38, "country": "Nigeria", "network": "AFREF", "alt_m": 39},
        ],
    }


# ── GNSS ─────────────────────────────────────────────────────────────────────

def test_gnss_synthetic_shape(cfg):
    from scripts.downloaders.gnss_samba import GNSSDownloader
    dl = GNSSDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    assert len(df) > 0
    assert "s4" in df.columns
    assert "station_id" in df.columns
    assert df["s4"].between(0, 1).all()


def test_gnss_s4_non_negative(cfg):
    from scripts.downloaders.gnss_samba import GNSSDownloader
    dl = GNSSDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    assert (df["s4"] >= 0).all()


# ── Solar wind ────────────────────────────────────────────────────────────────

def test_solar_wind_synthetic_columns(cfg):
    from scripts.downloaders.solar_wind import SolarWindDownloader
    dl = SolarWindDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    for col in ["bz", "v_sw", "n_p", "t_p"]:
        assert col in df.columns, f"Missing column: {col}"


def test_solar_wind_v_sw_range(cfg):
    from scripts.downloaders.solar_wind import SolarWindDownloader
    dl = SolarWindDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    assert df["v_sw"].between(200, 1000).all()


# ── Geomagnetic ───────────────────────────────────────────────────────────────

def test_geomagnetic_synthetic(cfg):
    from scripts.downloaders.geomagnetic import GeomagneticDownloader
    dl = GeomagneticDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    assert "kp" in df.columns
    assert "dst" in df.columns
    assert df["kp"].between(0, 9).all()


# ── Swarm ─────────────────────────────────────────────────────────────────────

def test_swarm_synthetic(cfg):
    from scripts.downloaders.swarm import SwarmDownloader
    dl = SwarmDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    assert "ne" in df.columns
    assert (df["ne"] >= 0).all()


# ── Aurora ────────────────────────────────────────────────────────────────────

def test_aurora_synthetic(cfg):
    from scripts.downloaders.aurora import AuroraDownloader
    dl = AuroraDownloader(cfg, use_synthetic=True)
    dl.run()
    df = dl.load()
    assert "hp_north" in df.columns
    assert "hp_south" in df.columns
