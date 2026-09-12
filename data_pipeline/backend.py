"""
Tier 3 SOC API — Member 3's backend.

Run:
    pip install fastapi uvicorn requests --break-system-packages
    uvicorn backend:app --port 5001 --reload

Endpoints:
    POST /review_alert             (Member 2 -> me)
    GET  /pending_alerts           (frontend polls this; Ambiguous only)
    GET  /all_events               (everything currently tracked — live feed)
    POST /resolve_alert/{flow_id}  (human verdict -> forwarded to Member 2)
    GET  /stats                    (counters for the dashboard)
"""
import time
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from status_logic import get_alert_status

MEMBER2_URL = "http://localhost:5000/update_weights"
MEMBER2_TIMEOUT_SECONDS = 3

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


# In-memory stores — fine for a 36-hour hackathon, not durable storage.
_alerts = {}            # flow_id -> record, everything still "live"
_resolved_history = []  # resolved records, for stats


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


@app.post("/review_alert")
def review_alert(payload: ReviewAlertPayload):
    # Duplicate flow_id -> update the existing record, don't duplicate it.
    _alerts[payload.flow_id] = _build_record(payload)
    return {
        "ok": True,
        "flow_id": payload.flow_id,
        "status": _alerts[payload.flow_id]["status_label"],
    }


@app.get("/pending_alerts")
def pending_alerts():
    """Only Ambiguous flows are actionable review items for the human."""
    return [a for a in _alerts.values() if a["status_label"] == "Ambiguous"]


@app.get("/all_events")
def all_events():
    """Everything currently tracked (Benign/Attack/Ambiguous) for the live feed."""
    return list(_alerts.values()) + list(_resolved_history)


@app.post("/resolve_alert/{flow_id}")
def resolve_alert(flow_id: str, payload: ResolvePayload):
    alert = _alerts.get(flow_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="flow_id not found")

    forward_body = {
        "flow_id": flow_id,
        "human_verified_label": payload.human_verified_label,
        "features": alert["features"],
    }

    try:
        resp = requests.post(MEMBER2_URL, json=forward_body, timeout=MEMBER2_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as e:
        # Member 2 unreachable/erroring: KEEP the alert, report failure clearly.
        return {
            "ok": False,
            "error": f"Could not reach Member 2's /update_weights: {e}",
            "flow_id": flow_id,
        }

    # Only remove from the live queue after Member 2 confirms success.
    resolved = _alerts.pop(flow_id)
    resolved["human_verified_label"] = payload.human_verified_label
    resolved["resolved_at"] = time.time()
    _resolved_history.append(resolved)

    return {
        "ok": True,
        "flow_id": flow_id,
        "member2_response": resp.json() if resp.content else None,
    }


@app.get("/stats")
def stats():
    live = list(_alerts.values())
    return {
        "total_seen": len(live) + len(_resolved_history),
        "currently_pending": len([a for a in live if a["status_label"] == "Ambiguous"]),
        "resolved": len(_resolved_history),
        "benign_logged": len([a for a in live if a["status_label"] == "Benign"]),
        "attack_autonomous": len([a for a in live if a["status_label"] == "Attack"]),
    }
