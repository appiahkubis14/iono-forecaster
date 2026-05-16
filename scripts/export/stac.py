"""
export/stac.py — IonoForecaster
STAC 1.0 catalog generation for scintillation forecast outputs.

STAC = SpatioTemporal Asset Catalog — the ESA/Copernicus standard
for describing geospatial data assets.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils import get_logger

logger = get_logger("export.stac")


def build_stac_catalog(
    cfg: Dict,
    forecast_paths: List[Path],
    events_path: Optional[Path] = None,
) -> Dict:
    """
    Build a STAC 1.0 Catalog JSON describing IonoForecaster outputs.

    Parameters
    ----------
    cfg : dict
    forecast_paths : list of Path
    events_path : Path, optional

    Returns
    -------
    dict
        STAC catalog.
    """
    bounds = cfg["study_region"]["bounds"]
    bbox = [
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"],
    ]

    items = []
    for path in forecast_paths:
        item = _build_stac_item(path, bbox, cfg)
        if item:
            items.append(item)

    if events_path and events_path.exists():
        events_item = _build_stac_item(events_path, bbox, cfg, item_type="scintillation_events")
        if events_item:
            items.append(events_item)

    catalog = {
        "type": "Catalog",
        "id": "iono-forecaster-catalog",
        "stac_version": "1.0.0",
        "description": (
            "IonoForecaster: AI-Based Ionospheric Scintillation Forecast "
            "for Equatorial Africa — ESA Copernicus Master's Portfolio"
        ),
        "title": "IonoForecaster Outputs — Equatorial Africa",
        "links": [],
        "items": items,
        "providers": [
            {
                "name": "Paris Lodron University Salzburg",
                "roles": ["producer"],
                "url": "https://www.plus.ac.at/",
            },
            {
                "name": "ESA / Copernicus",
                "roles": ["licensor"],
                "url": "https://www.esa.int/",
            },
        ],
        "extent": {
            "spatial": {"bbox": [bbox]},
            "temporal": {
                "interval": [
                    [cfg["temporal"]["start_date"], cfg["temporal"]["end_date"]]
                ]
            },
        },
        "generated": datetime.utcnow().isoformat(),
    }

    return catalog


def _build_stac_item(
    path: Path,
    bbox: List[float],
    cfg: Dict,
    item_type: str = "s4_forecast",
) -> Optional[Dict]:
    """Build a STAC Item for a single output file."""
    if not path.exists():
        return None

    stat = path.stat()
    media_type = "text/csv" if path.suffix == ".csv" else "application/geo+json"

    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": path.stem,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]],
        },
        "bbox": bbox,
        "properties": {
            "datetime": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            "title": f"IonoForecaster {item_type}",
            "description": f"Ionospheric scintillation {item_type} over equatorial Africa",
            "product_type": item_type,
            "platform": "IonoForecaster ST-GNN",
            "processing:level": "L4",
            "sci:doi": "https://github.com/appiahkubis14/iono-forecaster",
        },
        "links": [],
        "assets": {
            "data": {
                "href": str(path.resolve()),
                "type": media_type,
                "title": path.name,
                "roles": ["data"],
            }
        },
    }


def save_stac_catalog(cfg: Dict, catalog: Dict) -> Path:
    """Write STAC catalog JSON to disk."""
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "stac"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "catalog.json"
    with open(path, "w") as fh:
        json.dump(catalog, fh, indent=2)
    logger.info("STAC catalog saved → %s", path)
    return path
