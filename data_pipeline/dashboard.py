"""
Tier 3 Active Learning SOC Dashboard — Streamlit frontend (Step 2).

Consumes the FastAPI backend running on http://localhost:5001.
Run:
    streamlit run dashboard.py
"""

import random
import time
from datetime import datetime, timedelta

import requests
import streamlit as st

from status_logic import get_alert_status

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:5001"
POLL_INTERVAL_SEC = 2  # auto-refresh every 2 seconds

# The 10 contract features expected by the backend.
FEATURE_NAMES = [
    "sttl", "sbytes", "dbytes", "sload", "dur",
    "rate", "tcprtt", "synack", "ackdat", "ct_dst_src_ltm",
]

# ---------------------------------------------------------------------------
# GLOBAL STATE INITIALIZATION
# ---------------------------------------------------------------------------
# Initialized here so fragments don't throw NameError if sidebar UI is skipped
BASELINE_VECTOR = [0.01, 0.05, 0.02, 0.90, 0.03, 0.01, 0.05, 0.01, 0.02, 0.10]
DEVIATION_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# PAGE SETUP & SIDEBAR
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TTA-NIDS SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.sidebar.title("🛡️ TTA-NIDS Tier 3")
st.sidebar.subheader("🎬 Demo Controls")
if st.sidebar.button("🚨 Inject Zero-Day Attack", use_container_width=True):
    try:
        # Inject an ambiguous high-severity alert so it hits the pending queue
        requests.post(
            f"{BACKEND_URL}/review_alert",
            json={
                "flow_id": f"zero-day-{int(time.time())}",
                "model_confidence": 0.85,
                "predicted_class": "Attack",
                "features": [0.99] * 10
            },
            timeout=2
        )
        st.sidebar.success("Zero-day injected!")
        time.sleep(0.5)
        st.rerun()
    except requests.RequestException:
        st.sidebar.error("Backend unreachable!")

st.sidebar.subheader("⚙️ Configuration")
with st.sidebar.expander("Baseline Vector (UNSW-NB15)"):
    st.write("Configure benign feature baselines:")
    # Realistic placeholder values for benign traffic
    default_b = [0.01, 0.05, 0.02, 0.90, 0.03, 0.01, 0.05, 0.01, 0.02, 0.10]
    BASELINE_VECTOR = []
    for i, fname in enumerate(FEATURE_NAMES):
        val = st.number_input(f"{fname}", value=default_b[i], min_value=0.0, max_value=1.0, step=0.05)
        BASELINE_VECTOR.append(val)
    DEVIATION_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# POLLING HELPERS
# ---------------------------------------------------------------------------

def _safe_get(endpoint: str):
    """GET a backend endpoint; return (data, True) on success or (None, False)."""
    try:
        resp = requests.get(f"{BACKEND_URL}{endpoint}", timeout=2)
        resp.raise_for_status()
        return resp.json(), True
    except requests.RequestException:
        return None, False

def _resolve_alert(flow_id: str, label: int):
    """POST to /resolve_alert/{flow_id} to record human decision."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/resolve_alert/{flow_id}",
            json={"human_verified_label": label},
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", False):
            return False, data.get("error", "Unknown backend error")
        return True, ""
    except requests.RequestException as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# INITIAL SESSION STATE DEFAULTS (first run only)
# ---------------------------------------------------------------------------
for key, default in [
    ("pending_alerts", []),
    ("all_events", []),
    ("stats", {
        "total_seen": 0,
        "currently_pending": 0,
        "resolved": 0,
        "benign_logged": 0,
        "attack_autonomous": 0,
    }),
    ("backend_ok", False),
    ("mock_events", None),  # generated once, then kept in session state
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# POLLING FRAGMENT — runs every POLL_INTERVAL_SEC without blocking the UI
# ---------------------------------------------------------------------------
@st.fragment(run_every=timedelta(seconds=POLL_INTERVAL_SEC))
def _poll_fragment():
    """
    Fetch /pending_alerts, /all_events, and /stats from the backend.
    Results are stored in st.session_state so the main UI can read them.
    This fragment re-runs independently on its timer without triggering
    a full page rerun, keeping the UI responsive.
    """
    pending, pending_ok = _safe_get("/pending_alerts")
    all_ev, all_ev_ok = _safe_get("/all_events")
    stats, stats_ok = _safe_get("/stats")

    backend_ok = pending_ok and all_ev_ok and stats_ok

    # Only overwrite session state when we actually got fresh data.
    if pending_ok:
        st.session_state["pending_alerts"] = pending
    if all_ev_ok:
        st.session_state["all_events"] = all_ev
    if stats_ok:
        st.session_state["stats"] = stats

    st.session_state["backend_ok"] = backend_ok

    # --- Connection health indicator (re-renders inside this fragment) ---
    if backend_ok:
        st.success("✅ Backend connected — polling http://localhost:5001")
    else:
        st.error(
            "❌ Backend unreachable — is `uvicorn backend:app --port 5001` running?"
        )


# Render the polling fragment (its output appears here in the page flow).
_poll_fragment()

# ============================================================================
# SECTION 6: Summary counters & legend
# ============================================================================
st.subheader("📊 System Stats")
stats = st.session_state["stats"]
cols = st.columns(5)
cols[0].metric("Total Seen", stats.get("total_seen", 0))
cols[1].metric("Pending", stats.get("currently_pending", 0))
cols[2].metric("Resolved", stats.get("resolved", 0))
cols[3].metric("Benign Logged", stats.get("benign_logged", 0))
cols[4].metric("Attack Auto", stats.get("attack_autonomous", 0))

st.markdown("**Legend:** 🟢 Benign (P<0.1) | 🟡 Ambiguous (0.1<=P<=0.9) | 🔴 Attack (P>0.9)")
st.divider()


# ============================================================================
# --- MOCK EVENT GENERATOR (remove when real pipeline is live) ---------------
# These are LOCAL-only demo events to populate the live feed with green/red
# entries while Member 2 only sends Ambiguous ones.  They are never sent to
# the backend — purely cosmetic.
# ============================================================================

def _generate_mock_events(n_benign: int = 3, n_attack: int = 2) -> list:
    """Create a fixed set of mock Benign and Attack events for demo purposes."""
    mocks = []
    now = time.time()
    for i in range(n_benign):
        p = round(random.uniform(0.01, 0.08), 4)  # well under BENIGN_THRESHOLD
        status_label, color, severity = get_alert_status(p)
        mocks.append({
            "flow_id": f"mock-benign-{i+1:03d}",
            "model_confidence": p,
            "predicted_class": "Benign",
            "features": [round(random.uniform(0.0, 0.15), 3) for _ in range(10)],
            "status_label": status_label,
            "color": color,
            "severity": severity,
            "received_at": now - random.randint(10, 300),
            "source": "mock/demo",
        })
    for i in range(n_attack):
        p = round(random.uniform(0.92, 0.99), 4)  # well above ATTACK_THRESHOLD
        status_label, color, severity = get_alert_status(p)
        mocks.append({
            "flow_id": f"mock-attack-{i+1:03d}",
            "model_confidence": p,
            "predicted_class": "Attack",
            "features": [round(random.uniform(0.5, 1.0), 3) for _ in range(10)],
            "status_label": status_label,
            "color": color,
            "severity": severity,
            "received_at": now - random.randint(10, 300),
            "source": "mock/demo",
        })
    return mocks


# Generate once and cache in session state so they don't change every rerun.
if st.session_state["mock_events"] is None:
    st.session_state["mock_events"] = _generate_mock_events()

# ============================================================================
# --- END MOCK EVENT GENERATOR -----------------------------------------------
# ============================================================================


# ---------------------------------------------------------------------------
# SECTION 3, 4, 5: Pending Alerts & Resolution
# ---------------------------------------------------------------------------
st.subheader("⚠️ Action Required: Pending Alerts")

pending_alerts = st.session_state.get("pending_alerts", [])

if not pending_alerts:
    st.success("No pending alerts! The queue is clear.")
else:
    st.warning(f"{len(pending_alerts)} ambiguous flows require human verification.")
    
    # Render visually distinct card for each alert
    for alert in pending_alerts:
        flow_id = alert["flow_id"]
        status_label, color, _ = get_alert_status(alert["model_confidence"])
        
        with st.expander(f"🟡 Alert: {flow_id} | Confidence: {alert['model_confidence']:.4f}", expanded=False):
            st.write(f"**Flow ID:** `{flow_id}`")
            st.write(f"**Model Confidence:** `{alert['model_confidence']:.4f}` (Predicted: `{alert['predicted_class']}`)")
            
            # Feature Deviation Analysis
            st.markdown("##### Feature Deviation Analysis (Baseline Reference, not SHAP)")
            features = alert.get("features", [])
            
            if len(features) == len(FEATURE_NAMES):
                for i, fname in enumerate(FEATURE_NAMES):
                    val = float(features[i])
                    baseline = BASELINE_VECTOR[i]
                    diff = abs(val - baseline)
                    
                    if diff > DEVIATION_THRESHOLD:
                        st.markdown(f"**`{fname}`: {val:.3f} (Deviates by {diff:.3f} 🚨)**")
                    else:
                        st.markdown(f"`{fname}`: {val:.3f}")
            else:
                st.write("Features:", features)
            
            # Resolution Buttons
            st.markdown("##### Human Resolution")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Label as Benign", key=f"btn_benign_{flow_id}", use_container_width=True):
                    success, err = _resolve_alert(flow_id, 0)
                    if success:
                        st.session_state["pending_alerts"] = [a for a in st.session_state["pending_alerts"] if a["flow_id"] != flow_id]
                        st.success("Model updated")
                        st.rerun()
                    else:
                        st.error(f"Member 2 offline or unreachable - alert preserved in queue. (Detail: {err})")
                        
            with col2:
                if st.button("Label as Malicious", key=f"btn_attack_{flow_id}", use_container_width=True):
                    success, err = _resolve_alert(flow_id, 1)
                    if success:
                        st.session_state["pending_alerts"] = [a for a in st.session_state["pending_alerts"] if a["flow_id"] != flow_id]
                        st.success("Model updated")
                        st.rerun()
                    else:
                        st.error(f"Member 2 offline or unreachable - alert preserved in queue. (Detail: {err})")

st.divider()

# ---------------------------------------------------------------------------
# SECTION 2: Live Feed Panel
# ---------------------------------------------------------------------------
st.subheader("📡 Live Event Feed")

# Combine real backend events with local mock events.
_real_events = st.session_state["all_events"]
_mock_events = st.session_state["mock_events"]
_combined = _real_events + _mock_events

# Sort by timestamp descending (most recent first).
_combined.sort(key=lambda e: e.get("received_at", 0), reverse=True)

COLOR_DOT = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

if _combined:
    for ev in _combined:
        # Derive color from model_confidence via the shared status function.
        status_label, color, _ = get_alert_status(ev["model_confidence"])
        dot = COLOR_DOT.get(color, "⚪")
        short_id = ev["flow_id"][:12] + ("…" if len(ev["flow_id"]) > 12 else "")
        ts = datetime.fromtimestamp(ev.get("received_at", 0)).strftime("%H:%M:%S")
        source_tag = "  ⸺ _mock/demo_" if ev.get("source") == "mock/demo" else ""
        
        audit_tag = ""
        if "resolved_at" in ev:
            resolved_ts = datetime.fromtimestamp(ev["resolved_at"]).strftime("%H:%M:%S")
            verdict = "Malicious" if ev.get("human_verified_label") == 1 else "Benign"
            audit_tag = f"  |  `[Resolved: {verdict}]` at `{resolved_ts}`"

        st.markdown(
            f"{dot} **{short_id}**  |  `{ts}`  |  "
            f"{status_label}  |  class: {str(ev.get('predicted_class', '?')).title()}"
            f"{source_tag}{audit_tag}"
        )
else:
    st.info("No events yet — waiting for data from the backend or mock generator.")
