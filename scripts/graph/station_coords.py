"""
graph/station_coords.py — IonoForecaster
Station metadata registry: coordinates, country, network affiliation.

15 GNSS stations across equatorial Africa (30°W–60°E, 20°S–20°N).
Sources: IGS, AFREF, SAGAING, and national geodetic agencies.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ── Station registry ──────────────────────────────────────────────────────────

STATION_REGISTRY: List[Dict] = [
    # West Africa
    {"id": "DAKR", "name": "Dakar",         "lat":  14.73, "lon": -17.47, "country": "Senegal",           "network": "AFREF",  "alt_m":  23.0},
    {"id": "ABMF", "name": "Abymes",        "lat":  16.26, "lon": -61.53, "country": "Guadeloupe",        "network": "IGS",    "alt_m":  -25.0},
    {"id": "ACCR", "name": "Accra",         "lat":   5.56, "lon":  -0.20, "country": "Ghana",             "network": "SAGAING","alt_m":   73.0},
    {"id": "LAGO", "name": "Lagos",         "lat":   6.52, "lon":   3.38, "country": "Nigeria",           "network": "AFREF",  "alt_m":   39.0},
    {"id": "ABUJ", "name": "Abuja",         "lat":   9.03, "lon":   7.49, "country": "Nigeria",           "network": "AFREF",  "alt_m":  302.0},
    {"id": "NKLG", "name": "Libreville",    "lat":   0.35, "lon":   9.67, "country": "Gabon",             "network": "IGS",    "alt_m":   31.0},

    # Central Africa
    {"id": "TETB", "name": "Tete",          "lat": -16.18, "lon":  33.59, "country": "Mozambique",        "network": "AFREF",  "alt_m":  149.0},
    {"id": "KOUC", "name": "Koumac",        "lat": -20.56, "lon": 164.28, "country": "New Caledonia",     "network": "IGS",    "alt_m":  105.0},
    {"id": "DARE", "name": "Dar es Salaam", "lat":  -6.88, "lon":  39.20, "country": "Tanzania",          "network": "AFREF",  "alt_m":   63.0},
    {"id": "LUAK", "name": "Lusaka",        "lat": -15.41, "lon":  28.31, "country": "Zambia",            "network": "AFREF",  "alt_m": 1275.0},

    # East Africa
    {"id": "MBAR", "name": "Mbarara",       "lat":  -0.60, "lon":  30.74, "country": "Uganda",            "network": "AFREF",  "alt_m": 1380.0},
    {"id": "NAIR", "name": "Nairobi",       "lat":  -1.22, "lon":  36.89, "country": "Kenya",             "network": "IGS",    "alt_m": 1675.0},
    {"id": "ADIS", "name": "Addis Ababa",   "lat":   9.04, "lon":  38.77, "country": "Ethiopia",          "network": "IGS",    "alt_m": 2442.0},
    {"id": "KHAR", "name": "Khartoum",      "lat":  15.67, "lon":  32.55, "country": "Sudan",             "network": "AFREF",  "alt_m":  381.0},
    {"id": "HARR", "name": "Harare",        "lat": -17.83, "lon":  31.02, "country": "Zimbabwe",          "network": "AFREF",  "alt_m": 1460.0},
]

# Quick lookup by station ID
STATION_MAP: Dict[str, Dict] = {s["id"]: s for s in STATION_REGISTRY}


def get_station(station_id: str) -> Optional[Dict]:
    """Return station metadata dict, or None if not found."""
    return STATION_MAP.get(station_id)


def get_all_stations() -> List[Dict]:
    """Return all station metadata records."""
    return STATION_REGISTRY


def get_station_coords(station_id: str) -> Optional[tuple]:
    """Return (lat, lon) for a station, or None."""
    meta = get_station(station_id)
    if meta:
        return meta["lat"], meta["lon"]
    return None


def filter_by_region(
    lat_min: float = -20, lat_max: float = 20,
    lon_min: float = -30, lon_max: float = 60,
) -> List[Dict]:
    """Return stations within a bounding box."""
    return [
        s for s in STATION_REGISTRY
        if lat_min <= s["lat"] <= lat_max and lon_min <= s["lon"] <= lon_max
    ]
