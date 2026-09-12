"""
Tier 3 Active Learning SOC Dashboard — Streamlit frontend.

Consumes the FastAPI backend running on http://localhost:5001.
Run:
    streamlit run dashboard.py

Enhancements:
    - Feature Range Analysis Panel (SOC reference, NOT model prediction)
    - Auto-resolve feedback (shows count of cascaded resolutions)
    - Mock event generator removed (real pipeline data only)
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

from status_logic import get_alert_status

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:5001"
ML_BACKEND_URL = "http://localhost:5000"
POLL_INTERVAL_SEC = 2  # auto-refresh every 2 seconds

# The 10 contract features expected by the backend.
FEATURE_NAMES = [
    "sttl", "sbytes", "dbytes", "sload", "dur",
    "rate", "tcprtt", "synack", "ackdat", "ct_dst_src_ltm",
]

# ---------------------------------------------------------------------------
# GLOBAL STATE INITIALIZATION
# ---------------------------------------------------------------------------
BASELINE_VECTOR = [0.01, 0.05, 0.02, 0.90, 0.03, 0.01, 0.05, 0.01, 0.02, 0.10]
DEVIATION_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# PAGE SETUP & SIDEBAR
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TTA-NIDS SOC Dashboard",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
)

st.sidebar.title("\U0001f6e1\ufe0f TTA-NIDS Tier 3")
st.sidebar.subheader("\U0001f3ac Demo Controls")
if st.sidebar.button("\U0001f6a8 Inject Zero-Day Attack", use_container_width=True):
    try:
        base_time = int(time.time())
        payloads = [
            {
                "flow_id": f"zero-day-{base_time}-{i}",
                "model_confidence": 0.85,
                "predicted_class": "Attack",
                "features": [0.99] * 10
            }
            for i in range(100)
        ]
        requests.post(
            f"{BACKEND_URL}/review_alerts_batch",
            json=payloads,
            timeout=5
        )
        st.sidebar.success("100 Zero-day packets injected!")
        time.sleep(0.5)
        st.rerun()
    except requests.RequestException:
        st.sidebar.error("Backend unreachable!")

if st.sidebar.button("\U0001f525 Zero-Day Attack (Diff)", use_container_width=True):
    try:
        base_time = int(time.time())
        payloads = []
        # 5 groups of 20 identical packets
        for group in range(5):
            feature_val = 0.90 + (group * 0.02) # slightly different feature per group
            for i in range(20):
                payloads.append({
                    "flow_id": f"zero-day-diff-{base_time}-g{group}-{i}",
                    "model_confidence": 0.82 + (group * 0.01),
                    "predicted_class": "Attack",
                    "features": [feature_val] * 10
                })
        requests.post(
            f"{BACKEND_URL}/review_alerts_batch",
            json=payloads,
            timeout=5
        )
        st.sidebar.success("100 packets (5 groups) injected!")
        time.sleep(0.5)
        st.rerun()
    except requests.RequestException:
        st.sidebar.error("Backend unreachable!")

st.sidebar.divider()
if st.sidebar.button("\U0001f6bd Flush System", type="primary", use_container_width=True):
    try:
        resp = requests.post(f"{BACKEND_URL}/flush", timeout=5)
        if resp.status_code == 200:
            st.session_state["pending_alerts"] = []
            st.sidebar.success("System flushed & server closing!")
        else:
            st.sidebar.error("Flush failed.")
        time.sleep(1)
        st.rerun()
    except requests.RequestException:
        st.sidebar.error("Backend unreachable!")

st.sidebar.subheader("\u2699\ufe0f Configuration")
with st.sidebar.expander("Baseline Vector (UNSW-NB15)"):
    st.write("Configure benign feature baselines:")
    default_b = [0.01, 0.05, 0.02, 0.90, 0.03, 0.01, 0.05, 0.01, 0.02, 0.10]
    BASELINE_VECTOR = []
    for i, fname in enumerate(FEATURE_NAMES):
        val = st.number_input(f"{fname}", value=default_b[i], min_value=0.0, max_value=1.0, step=0.05)
        BASELINE_VECTOR.append(val)
    DEVIATION_THRESHOLD = 0.3

st.sidebar.divider()
st.sidebar.subheader("\u23f1\ufe0f Real-Time Simulation")
sim_running = st.sidebar.toggle("Start Simulation")
if sim_running != st.session_state.get("simulation_running", False):
    st.session_state["simulation_running"] = sim_running


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
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", False):
            return False, data.get("error", "Unknown backend error"), data
        return True, "", data
    except requests.RequestException as e:
        return False, str(e), {}


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
        "auto_grouped_waiting": 0,
        "auto_grouped_total": 0,
    }),
    ("backend_ok", False),
    ("feature_analysis", None),
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
    """
    pending, pending_ok = _safe_get("/pending_alerts")
    all_ev, all_ev_ok = _safe_get("/all_events")
    stats, stats_ok = _safe_get("/stats")

    backend_ok = pending_ok and all_ev_ok and stats_ok

    if pending_ok:
        st.session_state["pending_alerts"] = pending
    if all_ev_ok:
        st.session_state["all_events"] = all_ev
    if stats_ok:
        st.session_state["stats"] = stats

    st.session_state["backend_ok"] = backend_ok

    if backend_ok:
        st.success("\u2705 Backend connected \u2014 polling http://localhost:5001")
    else:
        st.error(
            "\u274c Backend unreachable \u2014 is `uvicorn backend:app --port 5001` running?"
        )


# Render the polling fragment.
_poll_fragment()


# ---------------------------------------------------------------------------
# SIMULATION FRAGMENT — runs every 20s if enabled
# ---------------------------------------------------------------------------
import random

@st.fragment(run_every=timedelta(seconds=20))
def _simulation_fragment():
    if not st.session_state.get("simulation_running", False):
        return

    base_time = int(time.time())
    payloads = []
    
    # 96% Benign
    for i in range(96):
        payloads.append({
            "flow_id": f"sim-benign-{base_time}-{i}",
            "model_confidence": 0.05,
            "predicted_class": "Benign",
            "features": [random.uniform(0.0, 0.05) for _ in range(10)]
        })
        
    # 2% Attack
    for i in range(2):
        payloads.append({
            "flow_id": f"sim-attack-{base_time}-{i}",
            "model_confidence": 0.95,
            "predicted_class": "Attack",
            "features": [random.uniform(0.9, 1.0) for _ in range(10)]
        })
        
    # 2% Ambiguous
    for i in range(2):
        payloads.append({
            "flow_id": f"sim-ambig-{base_time}-{i}",
            "model_confidence": 0.5,
            "predicted_class": "Ambiguous",
            "features": [random.uniform(0.4, 0.6) for _ in range(10)]
        })
        
    random.shuffle(payloads)
    
    try:
        # Send directly to Tier 3 so the dashboard sees the Benign and Attack packets in its stats immediately
        requests.post(f"{BACKEND_URL}/review_alerts_batch", json=payloads, timeout=5)
    except requests.RequestException:
        pass

# Render the simulation fragment
_simulation_fragment()


# ============================================================================
# SECTION: Summary counters & legend
# ============================================================================
st.subheader("\U0001f4ca System Stats")
stats = st.session_state["stats"]
cols = st.columns(7)
cols[0].metric("Total Seen", stats.get("total_seen", 0))
cols[1].metric("Pending", stats.get("currently_pending", 0))
cols[2].metric("Resolved", stats.get("resolved", 0))
cols[3].metric("Benign Logged", stats.get("benign_logged", 0))
cols[4].metric("Attack Auto", stats.get("attack_autonomous", 0))
cols[5].metric("Auto-Grouped", stats.get("auto_grouped_waiting", 0))
cols[6].metric("Total Grouped", stats.get("auto_grouped_total", 0))

st.markdown("**Legend:** \U0001f7e2 Benign (P<0.1) | \U0001f7e1 Ambiguous (0.1<=P<=0.9) | \U0001f534 Attack (P>0.9)")
st.divider()


# ============================================================================
# SECTION: Feature Range Analysis Panel — SOC Reference Guide
# ============================================================================
st.subheader("\U0001f4d6 Feature Range Analysis \u2014 SOC Reference Guide")
st.caption(
    "Historical reference from the UNSW-NB15 training dataset. "
    "This is NOT a model prediction \u2014 it shows what benign vs. attack traffic "
    "typically looked like during training. The SOC analyst still makes the final call."
)

# Fetch feature analysis (cache in session state to avoid repeated calls)
if st.session_state.get("feature_analysis") is None:
    fa_data, fa_ok = _safe_get("/feature_analysis")
    if fa_ok and "error" not in (fa_data or {}):
        st.session_state["feature_analysis"] = fa_data

fa = st.session_state.get("feature_analysis")

if fa and "features" in fa:
    feature_list = fa["features"]

    # ---- Grouped horizontal bar chart ----
    chart_data = []
    for f in feature_list:
        chart_data.append({
            "Feature": f["feature"],
            "Category": "Benign",
            "Mean Value": f["benign"]["mean"],
        })
        chart_data.append({
            "Feature": f["feature"],
            "Category": "Attack",
            "Mean Value": f["attack"]["mean"],
        })

    chart_df = pd.DataFrame(chart_data)

    # Use st.bar_chart via a pivot for simplicity
    pivot = chart_df.pivot(index="Feature", columns="Category", values="Mean Value")
    pivot = pivot.reindex(FEATURE_NAMES)  # Preserve feature order
    st.bar_chart(pivot, color=["#2ecc71", "#e74c3c"], horizontal=True)

    # ---- Reference table ----
    table_rows = []
    for f in feature_list:
        table_rows.append({
            "Feature": f["feature"],
            "Benign Range (Q25-Q75)": f"{f['benign']['q25']:.3f} \u2013 {f['benign']['q75']:.3f}",
            "Benign Mean": f"{f['benign']['mean']:.4f}",
            "Attack Range (Q25-Q75)": f"{f['attack']['q25']:.3f} \u2013 {f['attack']['q75']:.3f}",
            "Attack Mean": f"{f['attack']['mean']:.4f}",
            "SOC Guidance": f["guidance"],
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Source: {fa.get('source', 'unknown')} \u2014 {fa.get('total_benign', '?')} benign, {fa.get('total_attack', '?')} attack flows.")
else:
    st.info(
        "Feature analysis not yet available. "
        "Ensure the ML Backend (port 5000) is running with `/feature_analysis` endpoint."
    )

if st.button("\U0001f504 Refresh Feature Analysis"):
    st.session_state["feature_analysis"] = None
    st.rerun()

st.divider()


# ---------------------------------------------------------------------------
# SECTION: Pending Alerts & Resolution
# ---------------------------------------------------------------------------
st.subheader("\u26a0\ufe0f Action Required: Pending Alerts")

pending_alerts = st.session_state.get("pending_alerts", [])

if not pending_alerts:
    st.success("No pending alerts! The queue is clear.")
else:
    st.warning(f"{len(pending_alerts)} ambiguous flows require human verification.")

    # Render visually distinct card for each alert
    for alert in pending_alerts:
        flow_id = alert["flow_id"]
        grouped_count = alert.get("grouped_children_count", 0)
        status_label, color, _ = get_alert_status(alert["model_confidence"])

        group_badge = f" | \U0001f517 {grouped_count} similar auto-grouped" if grouped_count > 0 else ""

        with st.expander(
            f"\U0001f7e1 Alert: {flow_id} | Confidence: {alert['model_confidence']:.4f}{group_badge}",
            expanded=False
        ):
            st.write(f"**Flow ID:** `{flow_id}`")
            st.write(f"**Model Confidence:** `{alert['model_confidence']:.4f}` (Predicted: `{alert['predicted_class']}`)")
            if grouped_count > 0:
                st.info(f"\U0001f517 **{grouped_count} similar packets** are auto-grouped under this alert. "
                        f"When you resolve this alert, all {grouped_count} grouped packets will be "
                        f"auto-resolved with the same verdict and their weights forwarded to the model.")

            # Feature Deviation Analysis
            st.markdown("##### Feature Deviation Analysis (Baseline Reference, not SHAP)")
            features = alert.get("features", [])

            if len(features) == len(FEATURE_NAMES):
                for i, fname in enumerate(FEATURE_NAMES):
                    val = float(features[i])
                    baseline = BASELINE_VECTOR[i]
                    diff = abs(val - baseline)

                    if diff > DEVIATION_THRESHOLD:
                        st.markdown(f"**`{fname}`: {val:.3f} (Deviates by {diff:.3f} \U0001f6a8)**")
                    else:
                        st.markdown(f"`{fname}`: {val:.3f}")
            else:
                st.write("Features:", features)

            # Resolution Buttons
            st.markdown("##### Human Resolution")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Label as Benign", key=f"btn_benign_{flow_id}", use_container_width=True):
                    success, err, response_data = _resolve_alert(flow_id, 0)
                    if success:
                        st.session_state["pending_alerts"] = [
                            a for a in st.session_state["pending_alerts"]
                            if a["flow_id"] != flow_id
                        ]
                        auto_count = response_data.get("auto_resolved_count", 0)
                        if auto_count > 0:
                            st.success(f"\u2705 Model updated \u2014 {auto_count} similar flows auto-resolved as Benign")
                        else:
                            st.success("\u2705 Model updated")
                        st.rerun()
                    else:
                        st.error(f"Member 2 offline or unreachable - alert preserved in queue. (Detail: {err})")

            with col2:
                if st.button("Label as Malicious", key=f"btn_attack_{flow_id}", use_container_width=True):
                    success, err, response_data = _resolve_alert(flow_id, 1)
                    if success:
                        st.session_state["pending_alerts"] = [
                            a for a in st.session_state["pending_alerts"]
                            if a["flow_id"] != flow_id
                        ]
                        auto_count = response_data.get("auto_resolved_count", 0)
                        if auto_count > 0:
                            st.success(f"\u2705 Model updated \u2014 {auto_count} similar flows auto-resolved as Malicious")
                        else:
                            st.success("\u2705 Model updated")
                        st.rerun()
                    else:
                        st.error(f"Member 2 offline or unreachable - alert preserved in queue. (Detail: {err})")

st.divider()

# ---------------------------------------------------------------------------
# SECTION: Live Event Feed
# ---------------------------------------------------------------------------
st.subheader("\U0001f4e1 Live Event Feed")

_real_events = st.session_state["all_events"]

# Sort by timestamp descending (most recent first).
_real_events.sort(key=lambda e: e.get("received_at", 0), reverse=True)

COLOR_DOT = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}

if _real_events:
    # Show at most 200 events to keep the UI responsive
    display_events = _real_events[:200]
    if len(_real_events) > 200:
        st.caption(f"Showing most recent 200 of {len(_real_events)} events.")

    for ev in display_events:
        status_label, color, _ = get_alert_status(ev["model_confidence"])
        dot = COLOR_DOT.get(color, "\u26aa")
        short_id = ev["flow_id"][:12] + ("\u2026" if len(ev["flow_id"]) > 12 else "")
        ts = datetime.fromtimestamp(ev.get("received_at", 0)).strftime("%H:%M:%S")

        audit_tag = ""
        if "resolved_at" in ev:
            resolved_ts = datetime.fromtimestamp(ev["resolved_at"]).strftime("%H:%M:%S")
            verdict = "Malicious" if ev.get("human_verified_label") == 1 else "Benign"
            auto_tag = " (auto)" if ev.get("auto_resolved_by") else ""
            audit_tag = f"  |  `[Resolved{auto_tag}: {verdict}]` at `{resolved_ts}`"

        st.markdown(
            f"{dot} **{short_id}**  |  `{ts}`  |  "
            f"{status_label}  |  class: {str(ev.get('predicted_class', '?')).title()}"
            f"{audit_tag}"
        )
else:
    st.info("No events yet \u2014 waiting for data from the pipeline. "
            "Run the stream injector to generate traffic.")
