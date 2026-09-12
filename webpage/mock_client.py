"""
Mock traffic generator for testing Member 3's backend without waiting on
Member 1 or Member 2's real services.

Usage:
    python mock_client.py
"""
import random
import time
import uuid

import requests

SOC_URL = "http://localhost:5001"

# Deliberately span all three buckets, including the exact boundary values
# (0.10 and 0.90 are both Ambiguous by the spec), to visually confirm
# get_alert_status() draws the line in the right place.
TEST_CONFIDENCES = [0.02, 0.09, 0.10, 0.35, 0.65, 0.90, 0.91, 0.99]


def random_features():
    return [round(random.random(), 3) for _ in range(10)]


def send_alert(confidence: float) -> str:
    flow_id = str(uuid.uuid4())
    payload = {
        "flow_id": flow_id,
        "model_confidence": confidence,
        "predicted_class": "attack" if confidence > 0.5 else "benign",
        "features": random_features(),
    }
    resp = requests.post(f"{SOC_URL}/review_alert", json=payload, timeout=3)
    print(f"P={confidence:<5} -> {resp.json()}")
    return flow_id


if __name__ == "__main__":
    print("Sending test flows across Benign / Ambiguous / Attack buckets...\n")
    flow_ids = [send_alert(p) for p in TEST_CONFIDENCES]
    time.sleep(0.5)

    print("\npending_alerts (should contain ONLY the Ambiguous ones: 0.10, 0.35, 0.65, 0.90, 0.91? no -> 0.91 is Attack):")
    print(requests.get(f"{SOC_URL}/pending_alerts", timeout=3).json())

    print("\nstats:")
    print(requests.get(f"{SOC_URL}/stats", timeout=3).json())

    # Resolve one ambiguous flow to exercise the feedback loop.
    # index 3 -> confidence 0.35 (Ambiguous)
    ambiguous_flow = flow_ids[3]
    print(f"\nResolving {ambiguous_flow} (P=0.35) as Malicious...")
    resolve_resp = requests.post(
        f"{SOC_URL}/resolve_alert/{ambiguous_flow}",
        json={"human_verified_label": 1},
        timeout=5,
    )
    print(resolve_resp.json())

    print("\nstats after resolve:")
    print(requests.get(f"{SOC_URL}/stats", timeout=3).json())
