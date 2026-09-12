"""
Tier 3 SOC API — Member 3's backend.

Run:
    pip install fastapi uvicorn requests numpy --break-system-packages
    uvicorn backend:app --port 5001 --reload

Endpoints:
    POST /review_alert             (Member 2 -> me)
    POST /review_alerts_batch      (Member 2 batch -> me)
    GET  /pending_alerts           (frontend polls this; Ambiguous only)
    GET  /all_events               (everything currently tracked — live feed)
    POST /resolve_alert/{flow_id}  (human verdict -> forwarded to Member 2)
    GET  /stats                    (counters for the dashboard)
    GET  /feature_analysis         (proxy to ML Backend for SOC reference panel)

Fixes applied:
    - Auto-grouping on ingestion (Euclidean similarity < 0.15, 200-pkt window)
    - Auto-cascade resolution (SOC verdict propagates to grouped children)
    - Bounded state storage (eviction of old resolved history)
    - Feature analysis proxy endpoint
"""
import math
import time
from collections import deque
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from status_logic import get_alert_status

MEMBER2_URL = "http://localhost:5000/update_weights"
ML_BACKEND_URL = "http://localhost:5000"
MEMBER2_TIMEOUT_SECONDS = 8

# Auto-grouping config
SIMILARITY_THRESHOLD = 0.15       # Euclidean distance in 10-dim scaled space
AUTO_GROUP_WINDOW = 200           # Max grouped children per parent
MAX_RESOLVED_HISTORY = 5000       # Bounded history to prevent OOM
ACCEPTING_PACKETS = True          # Global flag for flushing/closing

app = FastAPI(title="TTA-NIDS Tier 3 SOC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewAlertPayload(BaseModel):
    flow_id: str
    model_confidence: float = Field(ge=0.0, le=1.0)
    predicted_class: str
    features: List[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 10:
            raise ValueError("features must contain exactly 10 values")
        for x in v:
            if not (0.0 <= x <= 1.0):
                raise ValueError("each feature must be in [0.0, 1.0]")
        return v


class ResolvePayload(BaseModel):
    human_verified_label: int

    @field_validator("human_verified_label")
    @classmethod
    def validate_label(cls, v):
        if v not in (0, 1):
            raise ValueError("human_verified_label must be exactly 0 or 1")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# In-memory stores with bounded eviction
# ──────────────────────────────────────────────────────────────────────────────

_alerts = {}                              # flow_id -> record (live/pending)
_resolved_history = deque(maxlen=MAX_RESOLVED_HISTORY)  # Bounded! LRU eviction

# Auto-grouping: parent_flow_id -> list of grouped child records
_grouped_children = {}                    # parent_flow_id -> [child_record, ...]
_group_counters = {}                      # parent_flow_id -> count (up to AUTO_GROUP_WINDOW)

# Stats counters
_auto_grouped_total = 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_record(payload: ReviewAlertPayload) -> dict:
    status_label, color, severity = get_alert_status(payload.model_confidence)
    return {
        "flow_id": payload.flow_id,
        "model_confidence": payload.model_confidence,
        "predicted_class": payload.predicted_class,  # supplementary only
        "features": payload.features,
        "status_label": status_label,
        "color": color,
        "severity": severity,
        "received_at": time.time(),
        "source": "review_alert",
    }


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two feature vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _find_similar_pending(features: List[float]) -> Optional[str]:
    """
    Check if any already-pending ambiguous alert has a similar feature vector.
    Returns the parent flow_id if found (and the group window isn't exhausted),
    else None.
    """
    for flow_id, alert in _alerts.items():
        if alert["status_label"] != "Ambiguous":
            continue

        # Check if this parent's group window is still open
        current_count = _group_counters.get(flow_id, 0)
        if current_count >= AUTO_GROUP_WINDOW:
            continue

        dist = _euclidean_distance(features, alert["features"])
        if dist < SIMILARITY_THRESHOLD:
            return flow_id

    return None


def _ingest_single_alert(payload: ReviewAlertPayload) -> dict:
    """
    Ingest a single alert with auto-grouping logic.
    Returns a result dict with grouping info.
    """
    global _auto_grouped_total

    if not ACCEPTING_PACKETS:
        return {
            "ok": False,
            "flow_id": payload.flow_id,
            "status": "rejected_queue_flushed"
        }

    record = _build_record(payload)

    # Only attempt grouping for ambiguous alerts
    if record["status_label"] == "Ambiguous":
        parent_id = _find_similar_pending(payload.features)
        if parent_id:
            # AUTO-GROUP: don't add to pending queue, attach to parent
            record["auto_grouped_under"] = parent_id
            record["auto_grouped_at"] = time.time()

            if parent_id not in _grouped_children:
                _grouped_children[parent_id] = []
            _grouped_children[parent_id].append(record)
            _group_counters[parent_id] = _group_counters.get(parent_id, 0) + 1
            _auto_grouped_total += 1

            return {
                "ok": True,
                "flow_id": payload.flow_id,
                "status": "auto_grouped",
                "grouped_under": parent_id,
                "group_size": _group_counters[parent_id],
            }

    # Normal ingestion
    _alerts[payload.flow_id] = record
    # Initialize group counter for new ambiguous alerts
    if record["status_label"] == "Ambiguous":
        _group_counters[payload.flow_id] = 0

    return {
        "ok": True,
        "flow_id": payload.flow_id,
        "status": record["status_label"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/review_alert")
def review_alert(payload: ReviewAlertPayload):
    return _ingest_single_alert(payload)


@app.post("/review_alerts_batch")
def review_alerts_batch(payloads: List[ReviewAlertPayload]):
    results = []
    grouped_count = 0
    for payload in payloads:
        result = _ingest_single_alert(payload)
        if result.get("status") == "auto_grouped":
            grouped_count += 1
        results.append(result)
    return {
        "ok": True,
        "count": len(results),
        "auto_grouped": grouped_count,
        "alerts": results
    }


@app.get("/pending_alerts")
def pending_alerts():
    """Only Ambiguous flows are actionable review items for the human."""
    pending = []
    for a in _alerts.values():
        if a["status_label"] == "Ambiguous":
            # Enrich with group count so dashboard can show it
            enriched = dict(a)
            enriched["grouped_children_count"] = _group_counters.get(a["flow_id"], 0)
            pending.append(enriched)
    return pending


@app.get("/all_events")
def all_events():
    """Everything currently tracked (Benign/Attack/Ambiguous) for the live feed."""
    return list(_alerts.values()) + list(_resolved_history)


@app.post("/resolve_alert/{flow_id}")
def resolve_alert(flow_id: str, payload: ResolvePayload):
    alert = _alerts.get(flow_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="flow_id not found")

    # 1. Forward the primary alert to Member 2 for weight update
    forward_body = {
        "flow_id": flow_id,
        "human_verified_label": payload.human_verified_label,
        "features": alert["features"],
    }

    try:
        resp = requests.post(MEMBER2_URL, json=forward_body, timeout=MEMBER2_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {
            "ok": False,
            "error": f"Could not reach Member 2's /update_weights: {e}",
            "flow_id": flow_id,
        }

    # 2. Resolve the primary alert
    resolved = _alerts.pop(flow_id)
    resolved["human_verified_label"] = payload.human_verified_label
    resolved["resolved_at"] = time.time()
    _resolved_history.append(resolved)

    # 3. CASCADE: Auto-resolve all grouped children with the same verdict
    children = _grouped_children.pop(flow_id, [])
    _group_counters.pop(flow_id, None)

    auto_resolved_count = 0
    auto_weight_update_failures = 0

    for child in children:
        child["human_verified_label"] = payload.human_verified_label
        child["resolved_at"] = time.time()
        child["auto_resolved_by"] = flow_id
        _resolved_history.append(child)
        auto_resolved_count += 1

    # Forward all children's features to Member 2 in a single batch (no heavy retraining)
    if children:
        batch_body = [
            {
                "flow_id": child["flow_id"],
                "human_verified_label": payload.human_verified_label,
                "features": child["features"],
            }
            for child in children
        ]
        try:
            batch_url = MEMBER2_URL.replace("/update_weights", "/insert_buffer_batch")
            batch_resp = requests.post(batch_url, json=batch_body, timeout=MEMBER2_TIMEOUT_SECONDS)
            batch_resp.raise_for_status()
        except requests.RequestException:
            auto_weight_update_failures += len(children)

    return {
        "ok": True,
        "flow_id": flow_id,
        "member2_response": resp.json() if resp.content else None,
        "auto_resolved_count": auto_resolved_count,
        "auto_weight_update_failures": auto_weight_update_failures,
    }


@app.get("/stats")
def stats():
    live = list(_alerts.values())
    total_grouped = sum(len(v) for v in _grouped_children.values())
    return {
        "total_seen": len(live) + len(_resolved_history),
        "currently_pending": len([a for a in live if a["status_label"] == "Ambiguous"]),
        "resolved": len(_resolved_history),
        "benign_logged": len([a for a in live if a["status_label"] == "Benign"]),
        "attack_autonomous": len([a for a in live if a["status_label"] == "Attack"]),
        "auto_grouped_waiting": total_grouped,
        "auto_grouped_total": _auto_grouped_total,
    }


@app.get("/feature_analysis")
def feature_analysis():
    """Proxy to ML Backend's /feature_analysis endpoint for the SOC dashboard."""
    try:
        resp = requests.get(f"{ML_BACKEND_URL}/feature_analysis", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": f"Could not reach ML Backend: {e}"}


import os
import threading

@app.post("/flush")
def flush_system():
    global ACCEPTING_PACKETS, _alerts, _grouped_children, _group_counters
    ACCEPTING_PACKETS = False
    _alerts.clear()
    _grouped_children.clear()
    _group_counters.clear()
    
    def shutdown():
        time.sleep(2)
        os._exit(0)
        
    threading.Thread(target=shutdown, daemon=True).start()
    return {"ok": True, "msg": "System flushed, queue cleared, not accepting packets. Shutting down."}
