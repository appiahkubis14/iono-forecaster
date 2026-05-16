"""
events/detect.py — IonoForecaster
Scintillation event detection from S4 forecast time series.

Events are defined as consecutive periods where S4 exceeds a threshold
for at least min_duration_minutes.

Output: GeoJSON polygons + CSV event log.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("events.detect")

LEVEL_LABELS = {
    "watch": {"s4_min": 0.3, "colour": "yellow"},
    "warning": {"s4_min": 0.4, "colour": "orange"},
    "severe": {"s4_min": 0.7, "colour": "red"},
}


def detect_events(
    forecast_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    min_duration_minutes: int = 30,
    resolution_minutes: int = 30,
) -> pd.DataFrame:
    """
    Detect scintillation events from S4 forecast.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Must have: timestamp, station_id, s4_forecast
    thresholds : dict, optional
        {'watch': 0.3, 'warning': 0.4, 'severe': 0.7}
    min_duration_minutes : int
    resolution_minutes : int

    Returns
    -------
    pd.DataFrame
        Event log: event_id, station_id, onset, end, peak_s4, level, duration_min
    """
    if thresholds is None:
        thresholds = {k: v["s4_min"] for k, v in LEVEL_LABELS.items()}

    min_steps = max(1, min_duration_minutes // resolution_minutes)
    events = []
    event_id = 0

    for sid, grp in forecast_df.groupby("station_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        s4 = grp["s4_forecast"].values
        ts = pd.to_datetime(grp["timestamp"].values)

        # Detect for each level
        for level, thresh in sorted(thresholds.items(), key=lambda x: -x[1]):
            in_event = False
            event_start = None
            event_vals = []

            for i, (t, val) in enumerate(zip(ts, s4)):
                if val >= thresh:
                    if not in_event:
                        in_event = True
                        event_start = t
                        event_vals = [val]
                    else:
                        event_vals.append(val)
                else:
                    if in_event and len(event_vals) >= min_steps:
                        event_id += 1
                        events.append({
                            "event_id": event_id,
                            "station_id": sid,
                            "level": level,
                            "onset": event_start,
                            "end": ts[i - 1],
                            "peak_s4": float(np.max(event_vals)),
                            "mean_s4": float(np.mean(event_vals)),
                            "duration_min": len(event_vals) * resolution_minutes,
                            "colour": LEVEL_LABELS[level]["colour"],
                        })
                    in_event = False
                    event_vals = []

            # Close open event at series end
            if in_event and len(event_vals) >= min_steps:
                event_id += 1
                events.append({
                    "event_id": event_id,
                    "station_id": sid,
                    "level": level,
                    "onset": event_start,
                    "end": ts[-1],
                    "peak_s4": float(np.max(event_vals)),
                    "mean_s4": float(np.mean(event_vals)),
                    "duration_min": len(event_vals) * resolution_minutes,
                    "colour": LEVEL_LABELS[level]["colour"],
                })

    df = pd.DataFrame(events)
    logger.info("Detected %d scintillation events.", len(df))
    return df


def events_to_geojson(
    events_df: pd.DataFrame,
    stations_meta: List[Dict],
    out_path: Path,
) -> None:
    """
    Export scintillation events as GeoJSON FeatureCollection.

    Each event is represented as a Point feature at the station location,
    with event properties as attributes.

    Parameters
    ----------
    events_df : pd.DataFrame
    stations_meta : list of dict
        Must have: id, lat, lon.
    out_path : Path
    """
    station_map = {s["id"]: s for s in stations_meta}
    features = []

    for _, row in events_df.iterrows():
        sid = row["station_id"]
        meta = station_map.get(sid, {})
        lat = meta.get("lat", 0.0)
        lon = meta.get("lon", 0.0)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "event_id": int(row["event_id"]),
                "station_id": sid,
                "level": row["level"],
                "onset": str(row["onset"]),
                "end": str(row["end"]),
                "peak_s4": round(float(row["peak_s4"]), 4),
                "mean_s4": round(float(row["mean_s4"]), 4),
                "duration_min": int(row["duration_min"]),
                "colour": row["colour"],
                "country": meta.get("country", ""),
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "generated": datetime.utcnow().isoformat(),
        "features": features,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(geojson, fh, indent=2)
    logger.info("Events GeoJSON saved → %s (%d features)", out_path, len(features))
