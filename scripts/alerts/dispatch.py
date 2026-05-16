"""
alerts/dispatch.py — IonoForecaster
Alert dispatch system for ionospheric scintillation events.

Formats: JSON (machine-readable), CSV (human-readable log),
         optional Telegram bot and SMTP email.

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import csv
import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.utils import get_logger

logger = get_logger("alerts.dispatch")


ALERT_COLOURS = {
    "watch":   "🟡",
    "warning": "🟠",
    "severe":  "🔴",
}

ALERT_ACTIONS = {
    "watch":   "Monitor GNSS performance closely.",
    "warning": "Prepare for signal degradation. Check backup positioning.",
    "severe":  "Suspend critical GNSS-dependent operations. Switch to INS/backup.",
}


class AlertDispatcher:
    """
    Generates and dispatches scintillation alerts.

    Parameters
    ----------
    cfg : dict
    out_dir : Path
    """

    def __init__(self, cfg: Dict, out_dir: Optional[Path] = None) -> None:
        self.cfg = cfg
        alert_cfg = cfg.get("alerts", {})
        self.enabled = alert_cfg.get("enabled", True)
        self.formats = alert_cfg.get("formats", ["json", "csv"])
        self.tg_token = alert_cfg.get("telegram_bot_token") or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.tg_chat = alert_cfg.get("telegram_chat_id") or os.environ.get("TELEGRAM_CHAT_ID")
        self.smtp_server = alert_cfg.get("email_smtp") or os.environ.get("SMTP_SERVER")
        self.email_from = alert_cfg.get("email_from") or os.environ.get("EMAIL_FROM")
        self.email_to = alert_cfg.get("email_to") or os.environ.get("EMAIL_TO")

        self.out_dir = out_dir or Path(cfg["paths"]["outputs_dir"]) / "alerts"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def dispatch(self, events_df: pd.DataFrame) -> None:
        """
        Dispatch alerts for all detected events.

        Parameters
        ----------
        events_df : pd.DataFrame
            Output of events/detect.py detect_events().
        """
        if not self.enabled or events_df.empty:
            logger.info("No alerts to dispatch.")
            return

        logger.info("Dispatching %d event alerts.", len(events_df))

        # Filter to active (current / future) events only
        now = datetime.utcnow()
        active = events_df[
            pd.to_datetime(events_df["end"]) >= pd.Timestamp(now)
        ]
        if active.empty:
            logger.info("No active/future events to alert.")
            return

        alert_payload = self._build_payload(active)

        if "json" in self.formats:
            self._write_json(alert_payload)
        if "csv" in self.formats:
            self._append_csv(active)
        if self.tg_token and self.tg_chat:
            self._send_telegram(alert_payload)
        if self.smtp_server and self.email_from and self.email_to:
            self._send_email(alert_payload)

    # ── alert builders ────────────────────────────────────────────────────────

    def _build_payload(self, events_df: pd.DataFrame) -> Dict:
        """Build structured alert payload."""
        alerts = []
        for _, row in events_df.iterrows():
            level = row["level"]
            alerts.append({
                "alert_id": f"IONO-{int(row['event_id']):05d}",
                "generated_utc": datetime.utcnow().isoformat(),
                "level": level,
                "colour": row["colour"],
                "station_id": row["station_id"],
                "onset_utc": str(row["onset"]),
                "end_utc": str(row["end"]),
                "peak_s4": round(float(row["peak_s4"]), 3),
                "duration_min": int(row["duration_min"]),
                "recommended_action": ALERT_ACTIONS.get(level, "Monitor."),
                "icon": ALERT_COLOURS.get(level, "⚪"),
            })

        # Summarise worst level
        level_order = {"severe": 3, "warning": 2, "watch": 1}
        max_level = max(
            events_df["level"],
            key=lambda lvl: level_order.get(lvl, 0),
            default="watch",
        )

        return {
            "product": "IonoForecaster — ESA Scintillation Alert",
            "version": "1.0.0",
            "generated_utc": datetime.utcnow().isoformat(),
            "region": "Equatorial Africa (30°W–60°E, 20°S–20°N)",
            "summary_level": max_level,
            "n_active_events": len(alerts),
            "alerts": alerts,
        }

    def _write_json(self, payload: Dict) -> None:
        """Write JSON alert to disk."""
        path = self.out_dir / "alert_latest.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("JSON alert written → %s", path)

    def _append_csv(self, events_df: pd.DataFrame) -> None:
        """Append events to rolling CSV alert history."""
        path = self.out_dir / "alert_history.csv"
        events_df["generated_utc"] = datetime.utcnow().isoformat()
        write_header = not path.exists()
        events_df.to_csv(path, mode="a", header=write_header, index=False)
        logger.info("CSV alert history updated → %s", path)

    def _send_telegram(self, payload: Dict) -> None:
        """Send Telegram bot message."""
        try:
            import requests

            text = self._format_telegram_message(payload)
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            resp = requests.post(url, data={
                "chat_id": self.tg_chat,
                "text": text,
                "parse_mode": "Markdown",
            }, timeout=10)
            resp.raise_for_status()
            logger.info("Telegram alert sent.")
        except Exception as exc:
            logger.error("Telegram dispatch failed: %s", exc)

    def _send_email(self, payload: Dict) -> None:
        """Send alert via SMTP email."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"[IonoForecaster] {payload['summary_level'].upper()} — "
                f"Scintillation Alert: {payload['n_active_events']} event(s)"
            )
            msg["From"] = self.email_from
            msg["To"] = self.email_to

            text = self._format_email_text(payload)
            msg.attach(MIMEText(text, "plain"))

            with smtplib.SMTP(self.smtp_server, 587) as server:
                server.starttls()
                server.send_message(msg)
            logger.info("Email alert sent to %s.", self.email_to)
        except Exception as exc:
            logger.error("Email dispatch failed: %s", exc)

    # ── message formatters ────────────────────────────────────────────────────

    @staticmethod
    def _format_telegram_message(payload: Dict) -> str:
        icon = ALERT_COLOURS.get(payload["summary_level"], "⚪")
        lines = [
            f"*{icon} IonoForecaster Alert*",
            f"Region: {payload['region']}",
            f"Level: *{payload['summary_level'].upper()}*",
            f"Active events: {payload['n_active_events']}",
            "",
        ]
        for a in payload["alerts"][:5]:  # truncate long lists
            lines.append(
                f"{a['icon']} *{a['level'].upper()}* | {a['station_id']} | "
                f"S4={a['peak_s4']:.2f} | {a['duration_min']} min"
            )
        lines.append(f"\n_Generated: {payload['generated_utc']} UTC_")
        return "\n".join(lines)

    @staticmethod
    def _format_email_text(payload: Dict) -> str:
        lines = [
            "IonoForecaster — ESA Ionospheric Scintillation Alert",
            "=" * 60,
            f"Generated: {payload['generated_utc']} UTC",
            f"Region: {payload['region']}",
            f"Summary level: {payload['summary_level'].upper()}",
            f"Active events: {payload['n_active_events']}",
            "",
            "Events:",
        ]
        for a in payload["alerts"]:
            lines.append(
                f"  [{a['level'].upper()}] Station {a['station_id']} | "
                f"Onset: {a['onset_utc']} | Peak S4: {a['peak_s4']:.3f} | "
                f"Duration: {a['duration_min']} min"
            )
            lines.append(f"    Action: {a['recommended_action']}")
        return "\n".join(lines)


def dispatch_alerts(
    cfg: Dict,
    events_df: pd.DataFrame,
) -> None:
    """Entry point called by main.py --step alerts."""
    dispatcher = AlertDispatcher(cfg)
    dispatcher.dispatch(events_df)
