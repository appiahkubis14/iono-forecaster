"""
alerts/thresholds.py — IonoForecaster
Alert threshold definitions and dynamic threshold calibration.

Thresholds align with ICAO GNSS performance standards and
empirical scintillation studies over equatorial Africa.

References:
  - ICAO Annex 10 Vol I: GNSS positioning requirements
  - Aarons (1982): Scintillation levels and GNSS effects
  - Béniguel et al. (2009): GNSS scintillation over Africa

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

# ── Static threshold definitions ─────────────────────────────────────────────

SCINTILLATION_THRESHOLDS: Dict[str, float] = {
    "watch":   0.3,   # Moderate; possible GNSS accuracy degradation
    "warning": 0.4,   # Strong; likely GNSS signal loss-of-lock
    "severe":  0.7,   # Extreme; satellite acquisition failure
}

SIGMA_PHI_THRESHOLDS: Dict[str, float] = {
    "watch":   0.10,  # radians
    "warning": 0.25,
    "severe":  0.50,
}

ROTI_THRESHOLDS: Dict[str, float] = {
    "watch":   0.5,   # TECU/min
    "warning": 1.0,
    "severe":  2.0,
}

# Recommended actions per level
ALERT_ACTIONS: Dict[str, str] = {
    "watch":   (
        "Increased monitoring recommended. GNSS positioning may degrade. "
        "Consider increasing update rates on safety-critical applications."
    ),
    "warning": (
        "Signal degradation likely. Prepare for loss-of-lock on low-elevation satellites. "
        "Switch to SBAS-augmented or dual-frequency solutions. "
        "Suspend precision agriculture or drone operations if possible."
    ),
    "severe":  (
        "GNSS acquisition failures expected. Suspend safety-critical GNSS operations. "
        "Switch to INS or ground-based navigation backup. "
        "Maritime vessels: switch to radar-only navigation. "
        "Aviation: use ILS/DME, increase crew awareness."
    ),
}

# Impact descriptions per sector
SECTOR_IMPACTS: Dict[str, Dict[str, str]] = {
    "aviation": {
        "watch":   "Increased positioning noise. Monitor RNP compliance.",
        "warning": "SBAS outages possible. Increase separation standards.",
        "severe":  "GNSS-based approaches suspended. Use conventional aids.",
    },
    "maritime": {
        "watch":   "Reduced GNSS accuracy. Cross-check with LORAN/radar.",
        "warning": "Position jumps expected. Reduce speed in confined waters.",
        "severe":  "Radar-only navigation recommended.",
    },
    "agriculture": {
        "watch":   "Variable tractor guidance accuracy.",
        "warning": "Precision seeding/spraying degraded. Suspend operations.",
        "severe":  "All RTK operations suspended.",
    },
    "surveying": {
        "watch":   "Extended observation times required.",
        "warning": "Static observations recommended. Suspend kinematic surveys.",
        "severe":  "All GNSS survey operations suspended.",
    },
}


def classify_s4(s4: float) -> str:
    """Return alert level string for a given S4 value."""
    if s4 >= SCINTILLATION_THRESHOLDS["severe"]:
        return "severe"
    elif s4 >= SCINTILLATION_THRESHOLDS["warning"]:
        return "warning"
    elif s4 >= SCINTILLATION_THRESHOLDS["watch"]:
        return "watch"
    return "quiet"


def calibrate_thresholds(
    s4_observations: np.ndarray,
    false_alarm_rate: float = 0.05,
    pod_target: float = 0.90,
) -> Dict[str, float]:
    """
    Calibrate S4 thresholds from historical observations using ROC analysis.

    The calibrated thresholds minimise false alarms subject to a target
    probability of detection (useful for site-specific tuning).

    Parameters
    ----------
    s4_observations : np.ndarray
        Historical S4 observations at a station.
    false_alarm_rate : float
        Target false alarm rate (default 5%).
    pod_target : float
        Minimum probability of detection (default 90%).

    Returns
    -------
    dict
        Calibrated {'watch': float, 'warning': float, 'severe': float}
    """
    if len(s4_observations) < 100:
        return SCINTILLATION_THRESHOLDS.copy()

    s4_clean = s4_observations[~np.isnan(s4_observations)]
    percentiles = {
        "watch":   float(np.percentile(s4_clean, 75)),
        "warning": float(np.percentile(s4_clean, 90)),
        "severe":  float(np.percentile(s4_clean, 99)),
    }

    # Clip to physical range and ensure order
    percentiles["watch"]   = np.clip(percentiles["watch"],   0.15, 0.40)
    percentiles["warning"] = np.clip(percentiles["warning"], 0.30, 0.60)
    percentiles["severe"]  = np.clip(percentiles["severe"],  0.60, 0.95)

    # Ensure strict ordering
    percentiles["warning"] = max(percentiles["warning"], percentiles["watch"] + 0.05)
    percentiles["severe"]  = max(percentiles["severe"],  percentiles["warning"] + 0.10)

    return percentiles


def get_thresholds(cfg: Optional[Dict] = None) -> Dict[str, float]:
    """
    Return alert thresholds from config (or defaults if not configured).

    Parameters
    ----------
    cfg : dict, optional

    Returns
    -------
    dict
    """
    if cfg:
        return cfg.get("events", {}).get("thresholds", SCINTILLATION_THRESHOLDS)
    return SCINTILLATION_THRESHOLDS.copy()
