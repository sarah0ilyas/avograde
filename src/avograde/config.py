"""Central configuration — no magic numbers scattered across modules."""
from __future__ import annotations

# Ripeness taxonomy. The 5th class is a defect/reject flag.
STAGES = ["underripe", "breaking", "ripe", "overripe"]
CLASS_NAMES = [*STAGES, "reject"]

# Drift thresholds (Population Stability Index).
PSI_WATCH = 0.1
PSI_ALERT = 0.2

# Serving.
LATENCY_BUDGET_MS = 100.0
CACHE_TTL_S = 7 * 24 * 3600      # inputs refresh ~weekly
MODEL_VERSION = "avograder-v1"

# Features watched for drift.
DRIFT_FEATURES = ["brightness", "green_ratio", "dark_frac"]
