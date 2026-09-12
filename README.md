# TTA-NIDS: Tier-3 Threat Analysis & Network Intrusion Detection System

TTA-NIDS is an advanced, human-in-the-loop Active Learning cybersecurity pipeline designed to autonomously handle the vast majority of network traffic while gracefully delegating zero-day or ambiguous threats to a human Security Operations Center (SOC) analyst. The system features a continuous learning loop protected by Validation Gating and Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

## 🚀 Quickstart: Running the Project

The entire system consists of three interconnected services. We have provided a PowerShell script to spin them all up simultaneously in separate terminal windows.

1. Open a PowerShell terminal.
2. Navigate to the project root directory:
   ```powershell
   cd d:\vit\SEM3\AIDS
   ```
3. Execute the startup script:
   ```powershell
   .\start_all.ps1
   ```

This script will automatically launch:
- **Tier 2 ML Backend** (FastAPI running on `http://localhost:5000`)
- **Tier 3 SOC API** (FastAPI running on `http://localhost:5001`)
- **SOC Dashboard** (Streamlit running on `http://localhost:8501`)

The script will automatically open the dashboard in your default web browser.

---

## 🛡️ The SOC Dashboard (Tier 3)

The Streamlit dashboard (`http://localhost:8501`) acts as the command center for the SOC analyst. It is packed with features designed to reduce alert fatigue and accelerate human decision-making.

### Key Features & Panels

#### 1. System Stats
A real-time overview of the network's health. It tracks:
- **Total Seen:** All packets processed by the pipeline.
- **Benign Logged / Attack Auto:** Packets that the ML Backend was highly confident about (>90% or <10%). These were handled entirely autonomously without bothering the SOC.
- **Pending:** Ambiguous packets (10% - 90% confidence) currently awaiting your manual review.
- **Auto-Grouped / Total Grouped:** The number of similar threats that have been clustered together to save you time.

#### 2. Feature Range Analysis — SOC Reference Guide
A historical reference table built directly from the training data. For any given feature (e.g., `sbytes` or `sttl`), it shows you what the standard "Benign" range looks like compared to the "Attack" range.
- *Why it's useful:* When reviewing an ambiguous packet, you can compare its features against this panel to make an informed, data-driven decision.

#### 3. Action Required: Pending Alerts
The core active-learning queue. When the ML model isn't sure about a packet, it quarantines it here.
- **Feature Deviation Analysis:** Highlights exactly which features of the packet deviate significantly from the known benign baselines.
- **Auto-Grouping & Cascade Resolution:** If 100 identical zero-day packets hit the network, the dashboard won't show you 100 alerts. It shows **1 parent alert** with a badge indicating `🔗 99 similar auto-grouped`. When you click "Label as Malicious", that single click cascades to all 100 packets, instantly resolving them and retraining the ML model on the entire batch.

#### 4. Live Event Feed
A continuous, scrolling log of all traffic evaluated by the system, color-coded by severity (🟢 Benign, 🟡 Ambiguous, 🔴 Attack) with audit timestamps showing exactly when an analyst intervened.

---

## 🎬 Demo Controls (Sidebar)

To help you test and demonstrate the pipeline's capabilities without needing external traffic generators, the dashboard includes a suite of built-in demo controls on the left sidebar:

* **🚨 Inject Zero-Day Attack:** Injects a massive burst of 100 identical zero-day malicious packets. This perfectly demonstrates the **Auto-Grouping** feature (you'll see 1 pending alert representing all 100 packets).
* **🔥 Zero-Day Attack (Diff):** Injects 100 zero-day packets split across 5 slightly different structural groups. Demonstrates the system's ability to cluster distinct zero-day campaigns simultaneously.
* **⏱️ Real-Time Simulation:** A toggle that continuously injects a realistic traffic stream (96% benign, 2% attack, 2% ambiguous) every 20 seconds. This runs entirely in the background without freezing the UI, allowing you to watch the autonomous stats climb while you manually resolve the trickle of ambiguous alerts.
* **🚽 Flush System:** A hard reset button. It clears the pending queue, flushes the auto-grouped children, immediately stops accepting new packets, and safely shuts down the Tier 3 server. *(Note: You will need to manually restart `uvicorn backend:app --port 5001` if you use this!)*

---

## 🧠 Under the Hood: The ML Backend

While you interact with the Dashboard, the Tier 2 ML Backend (Port 5000) does the heavy lifting:
- **Double-Buffer Inference:** The system separates the "inference model" (answering live traffic) from the "training model". The system never blocks live traffic while learning.
- **Elastic Weight Consolidation (EWC):** When you label an ambiguous packet, the model immediately retrains on it. EWC ensures the model remembers past traffic patterns (preventing "catastrophic forgetting" of old threats while learning the new one).
- **Validation Gating:** Before any new weights are pushed to production, the backend tests them against a hold-out validation set. If accuracy drops below 70%, the update is automatically rejected and rolled back, preventing model poisoning attacks.
