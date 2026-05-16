"""
export/dashboard.py — IonoForecaster
Interactive Folium + Plotly dashboard for scintillation forecast visualisation.

Features:
  - Map with station markers coloured by current S4 level
  - Folium FeatureGroup layers: watch / warning / severe
  - Time series chart per station (Plotly, embedded as HTML)
  - 6-hour forecast with uncertainty bands
  - Alert overlay

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.utils import get_logger

logger = get_logger("export.dashboard")

ALERT_COLOURS = {
    "quiet":   "#2ecc71",
    "watch":   "#f1c40f",
    "warning": "#e67e22",
    "severe":  "#e74c3c",
}


def _s4_to_level(s4: float) -> str:
    if s4 >= 0.7:
        return "severe"
    elif s4 >= 0.4:
        return "warning"
    elif s4 >= 0.3:
        return "watch"
    return "quiet"


def build_dashboard(
    cfg: Dict,
    forecast_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    stations_meta: Optional[List[Dict]] = None,
) -> str:
    """
    Build an interactive HTML dashboard.

    Parameters
    ----------
    cfg : dict
    forecast_df : pd.DataFrame
        Columns: timestamp, station_id, s4_forecast, s4_lower, s4_upper
    events_df : pd.DataFrame, optional
    stations_meta : list of dict

    Returns
    -------
    str
        Path to saved dashboard HTML file.
    """
    try:
        import folium
        from folium import plugins
    except ImportError:
        logger.error("folium not installed. Run: pip install folium")
        return ""

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        has_plotly = True
    except ImportError:
        has_plotly = False
        logger.warning("plotly not installed — charts will be omitted.")

    dash_cfg = cfg.get("dashboard", {})
    default_view = dash_cfg.get("default_view", {})
    center_lat = default_view.get("lat", 0)
    center_lon = default_view.get("lon", 15)
    zoom = default_view.get("zoom", 3)

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    # Feature groups per alert level
    fg_severe = folium.FeatureGroup(name="🔴 Severe (S4 > 0.7)", show=True)
    fg_warning = folium.FeatureGroup(name="🟠 Warning (S4 > 0.4)", show=True)
    fg_watch = folium.FeatureGroup(name="🟡 Watch (S4 > 0.3)", show=True)
    fg_quiet = folium.FeatureGroup(name="🟢 Quiet", show=True)
    groups = {
        "severe": fg_severe, "warning": fg_warning,
        "watch": fg_watch, "quiet": fg_quiet,
    }

    # Add station markers
    if stations_meta is None:
        from scripts.downloaders.gnss_samba import STATION_META
        stations_meta = STATION_META

    station_map = {s["id"]: s for s in stations_meta}

    # Get latest forecast per station
    if not forecast_df.empty:
        latest = forecast_df.sort_values("timestamp").groupby("station_id").last().reset_index()
    else:
        latest = pd.DataFrame(columns=["station_id", "s4_forecast"])

    for sid, meta in station_map.items():
        lat, lon = meta["lat"], meta["lon"]
        row = latest[latest["station_id"] == sid]
        s4_val = float(row["s4_forecast"].values[0]) if not row.empty else 0.0
        level = _s4_to_level(s4_val)
        colour = ALERT_COLOURS[level]

        # Build popup
        popup_html = _station_popup(sid, meta, s4_val, forecast_df, has_plotly)

        folium.CircleMarker(
            location=[lat, lon],
            radius=12 + s4_val * 20,  # size scales with S4
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            tooltip=f"{sid} ({meta.get('country','')}) | S4={s4_val:.3f} | {level.upper()}",
            popup=folium.Popup(popup_html, max_width=450),
        ).add_to(groups[level])

    for fg in groups.values():
        fg.add_to(m)

    # Event polygons
    if events_df is not None and not events_df.empty:
        fg_events = folium.FeatureGroup(name="🌩 Scintillation Events", show=True)
        for _, evt in events_df.iterrows():
            meta = station_map.get(evt["station_id"], {})
            folium.Circle(
                location=[meta.get("lat", 0), meta.get("lon", 0)],
                radius=200_000 * evt["peak_s4"],
                color=evt.get("colour", "red"),
                fill=False,
                weight=2,
                tooltip=(
                    f"Event {evt['event_id']} | {evt['level'].upper()} | "
                    f"Peak S4={evt['peak_s4']:.3f} | {evt['duration_min']} min"
                ),
            ).add_to(fg_events)
        fg_events.add_to(m)

    # Equatorial ionisation anomaly reference lines
    for lat_line in [-15, -10, 0, 10, 15]:
        folium.PolyLine(
            locations=[[lat_line, -30], [lat_line, 60]],
            color="steelblue",
            weight=0.5,
            opacity=0.4,
            tooltip=f"{lat_line}° latitude",
            dash_array="5 5",
        ).add_to(m)

    # Title + legend HTML
    legend_html = _build_legend_html(cfg)
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)

    # Save
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "iono_dashboard.html")
    m.save(out_path)
    logger.info("Dashboard saved → %s", out_path)
    return out_path


def _station_popup(
    sid: str, meta: Dict, s4_val: float,
    forecast_df: pd.DataFrame, has_plotly: bool,
) -> str:
    """Build HTML popup for a station marker."""
    level = _s4_to_level(s4_val)
    colour = ALERT_COLOURS[level]

    html = f"""
    <div style="font-family: Arial; width: 400px;">
        <h4 style="color:{colour}; margin:0;">{sid} — {meta.get('country','')}</h4>
        <p style="margin:2px 0;">Network: <b>{meta.get('network','')}</b> |
           Lat: {meta.get('lat',0):.2f}° | Lon: {meta.get('lon',0):.2f}°</p>
        <p>Current S4: <b style="color:{colour}">{s4_val:.3f}</b>
           ({level.upper()})</p>
    """

    # Mini time series
    if not forecast_df.empty:
        station_fc = forecast_df[forecast_df["station_id"] == sid].sort_values("timestamp")
        if not station_fc.empty and has_plotly:
            try:
                import plotly.graph_objects as go
                import plotly.io as pio

                ts = pd.to_datetime(station_fc["timestamp"])
                s4_fc = station_fc["s4_forecast"].values
                s4_lo = station_fc.get("s4_lower", pd.Series(np.zeros(len(s4_fc)))).values
                s4_hi = station_fc.get("s4_upper", pd.Series(np.zeros(len(s4_fc)))).values

                fig = go.Figure()
                fig.add_scatter(
                    x=ts, y=s4_fc,
                    name="S4 forecast", line=dict(color="#2980b9"),
                )
                fig.add_scatter(
                    x=list(ts) + list(ts[::-1]),
                    y=list(s4_hi) + list(s4_lo[::-1]),
                    fill="toself", fillcolor="rgba(41,128,185,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% CI",
                )
                for thresh, col, lbl in [
                    (0.3, "gold", "Watch"),
                    (0.4, "orange", "Warning"),
                    (0.7, "red", "Severe"),
                ]:
                    fig.add_hline(
                        y=thresh, line_dash="dash", line_color=col,
                        annotation_text=lbl, annotation_position="right",
                    )
                fig.update_layout(
                    height=200, margin=dict(l=30, r=10, t=10, b=30),
                    showlegend=False,
                    yaxis=dict(range=[0, 1], title="S4"),
                    xaxis_title="UTC",
                    plot_bgcolor="#f9f9f9",
                )
                chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
                html += chart_html
            except Exception as exc:
                html += f"<p><i>Chart unavailable: {exc}</i></p>"

    html += "</div>"
    return html


def _build_legend_html(cfg: Dict) -> str:
    return """
    <div style="position:fixed; bottom:30px; left:10px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                border:2px solid #ccc; font-family:Arial; font-size:13px;">
        <b>IonoForecaster</b><br>
        <b>S4 Scintillation Level</b><br>
        <span style="color:#2ecc71;">●</span> Quiet (S4 < 0.3)<br>
        <span style="color:#f1c40f;">●</span> Watch (0.3–0.4)<br>
        <span style="color:#e67e22;">●</span> Warning (0.4–0.7)<br>
        <span style="color:#e74c3c;">●</span> Severe (> 0.7)<br>
        <hr style="margin:6px 0;">
        <small>Copernicus Master's in Digital Earth<br>
        ESA SSA Portfolio | S. Appiah Kubi</small>
    </div>
    """
