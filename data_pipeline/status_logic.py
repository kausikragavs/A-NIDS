"""
Single source of truth for translating a probability P (likelihood a flow
is malicious) into a status label, color, and severity.

Import this from the backend, the mock test client, and the frontend
dashboard. Do NOT re-implement this comparison anywhere else — every
alert card, live-feed row, badge, and counter must derive its color from
this one function so the thresholds can never drift out of sync.

Boundary rule (intentional, do not change without updating all callers):
    P < 0.1         -> Benign     (green,  low)
    0.1 <= P <= 0.9 -> Ambiguous  (yellow, medium)  <- only this bucket
                                                        is eligible for
                                                        the pending queue
    P > 0.9         -> Attack     (red,    high)
"""
from typing import Tuple

BENIGN_THRESHOLD = 0.1
ATTACK_THRESHOLD = 0.9


def get_alert_status(p: float) -> Tuple[str, str, str]:
    """
    Args:
        p: probability in [0.0, 1.0] that the flow is malicious.

    Returns:
        (status_label, color, severity)
    """
    if p < BENIGN_THRESHOLD:
        return "Benign", "green", "low"
    if p > ATTACK_THRESHOLD:
        return "Attack", "red", "high"
    return "Ambiguous", "yellow", "medium"


if __name__ == "__main__":
    # Quick manual sanity check of the exact boundary values.
    for p in [0.0, 0.05, 0.099999, 0.1, 0.5, 0.9, 0.900001, 0.95, 1.0]:
        print(f"P={p:<10} -> {get_alert_status(p)}")
