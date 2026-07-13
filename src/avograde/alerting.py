"""Alert / retrain trigger — acts on drift. Detection lives in DriftMonitor;
this decides what to DO when it fires: log an alert and flag a retrain.
Retraining is expensive and human-gated, so we flag, not auto-run.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from avograde.monitoring.drift import DriftMonitor

ALERT_LOG = Path("drift_alerts.jsonl")
RETRAIN_FLAG = Path("RETRAIN_NEEDED")


def evaluate(monitor: DriftMonitor, current: dict) -> bool:
    """Check a batch of incoming features; act if drift alerts. Returns True if alerted."""
    results = monitor.check(current)
    alerted = DriftMonitor.any_alert(results)

    record = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "alert": alerted,
        "features": {r.feature: {"psi": round(r.psi, 3), "status": r.status}
                     for r in results},
    }
    with ALERT_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")

    if alerted:
        drifted = [r.feature for r in results if r.status == "alert"]
        RETRAIN_FLAG.write_text(
            f"retrain recommended {record['time']}\n"
            f"drifted features: {', '.join(drifted)}\n"
        )
        print(f"ALERT: drift on {drifted} — logged and flagged for retrain")
    else:
        print("no drift — logged, no action")
    return alerted
