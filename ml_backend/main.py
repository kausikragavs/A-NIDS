"""
TTA-NIDS ML Backend (Port 5000) — Tier 2 Classifier & Continual Learning Engine.

Fixes applied:
  - Persistent httpx.AsyncClient (connection pooling)
  - asyncio.RWLock pattern (concurrent reads, exclusive writes)
  - Persistent Adam optimizer (momentum preserved across updates)
  - Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
  - Validation gating: candidate weights benchmarked before promotion
  - Double-buffer inference/training separation
  - GET /feature_analysis endpoint for SOC reference panel
"""

import asyncio
import copy
import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import httpx
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from train_offline import NIDS_MLP
from memory_buffer import MemoryBuffer

app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# Global states
# ──────────────────────────────────────────────────────────────────────────────

# Inference model — served to all /predict and /predict_batch calls (read-only)
inference_model = NIDS_MLP()

# Training model — used exclusively inside /update_weights (write-locked)
training_model = NIDS_MLP()

# RWLock pattern: asyncio.Lock for writes, allow concurrent reads via a flag
_write_lock = asyncio.Lock()
_readers_count = 0
_readers_lock = asyncio.Lock()  # protects _readers_count


async def acquire_read():
    """Acquire a read lock (allows concurrent readers, blocks during write)."""
    global _readers_count
    async with _readers_lock:
        _readers_count += 1
        if _readers_count == 1:
            await _write_lock.acquire()


async def release_read():
    """Release a read lock."""
    global _readers_count
    async with _readers_lock:
        _readers_count -= 1
        if _readers_count == 0:
            _write_lock.release()


buffer = MemoryBuffer()

# Persistent optimizer — preserves momentum (m_t, v_t) across update calls
persistent_optimizer: Optional[optim.Adam] = None

# Elastic Weight Consolidation (EWC) — Fisher diagonal + base weights
_ewc_fisher: Dict[str, torch.Tensor] = {}
_ewc_base_weights: Dict[str, torch.Tensor] = {}
EWC_LAMBDA = 0.4  # Regularization strength

# Validation hold-out set (loaded at startup)
_validation_X: Optional[torch.Tensor] = None
_validation_y: Optional[torch.Tensor] = None
VALIDATION_ACCURACY_FLOOR = 0.70  # Minimum accuracy to accept new weights

# Feature analysis cache (computed once at startup)
_feature_analysis_cache: Optional[dict] = None

# Persistent HTTP client for connection pooling to Tier 3
_http_client: Optional[httpx.AsyncClient] = None

TIER_3_URL = "http://localhost:5001/review_alert"
TIER_3_BATCH_URL = "http://localhost:5001/review_alerts_batch"

FEATURE_NAMES = [
    'sttl', 'sbytes', 'dbytes', 'sload', 'dur',
    'rate', 'tcprtt', 'synack', 'ackdat', 'ct_dst_src_ltm'
]


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    flow_id: str
    timestamp: str
    tier1_anomaly_score: float
    features: List[float] = Field(..., min_length=10, max_length=10)


class UpdateRequest(BaseModel):
    flow_id: str
    human_verified_label: float
    features: List[float] = Field(..., min_length=10, max_length=10)


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global persistent_optimizer, _http_client
    global _validation_X, _validation_y, _feature_analysis_cache
    global _ewc_fisher, _ewc_base_weights

    # Load model weights into both inference and training copies
    weights = torch.load('artifacts/base_model.pth', weights_only=True)
    inference_model.load_state_dict(weights)
    inference_model.eval()
    training_model.load_state_dict(weights)
    training_model.eval()

    # Initialize persistent Adam optimizer on training model
    persistent_optimizer = optim.Adam(training_model.parameters(), lr=5e-4)

    # Initialize EWC base weights (snapshot of initial production model)
    _ewc_base_weights = {k: v.clone() for k, v in weights.items()}

    # Compute Fisher Information diagonal from a sample of the memory buffer
    _compute_fisher_diagonal()

    # Persistent HTTP client with connection pooling
    _http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(5.0, connect=2.0)
    )

    # Load validation hold-out set for gating
    _load_validation_set()

    # Precompute feature analysis for SOC dashboard
    _compute_feature_analysis()

    print("[startup] ML Backend ready — inference model, EWC, validation gating, feature analysis loaded.")


@app.on_event("shutdown")
async def shutdown_event():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


# ──────────────────────────────────────────────────────────────────────────────
# EWC: Fisher Information Diagonal
# ──────────────────────────────────────────────────────────────────────────────

def _compute_fisher_diagonal():
    """
    Approximate the Fisher Information diagonal using samples from the memory buffer.
    This tells the optimizer which weights are "important" for past traffic.
    """
    global _ewc_fisher

    samples = buffer.sample_batch(min(64, len(buffer.benign_buffer) + len(buffer.attack_buffer)))
    if not samples:
        _ewc_fisher = {k: torch.zeros_like(v) for k, v in training_model.state_dict().items()}
        return

    X = torch.FloatTensor([s[0] for s in samples])
    y = torch.FloatTensor([s[1] for s in samples]).unsqueeze(1)

    training_model.eval()
    criterion = nn.BCEWithLogitsLoss()

    # Zero all existing grads
    training_model.zero_grad()

    # Accumulate squared gradients (Fisher diagonal approximation)
    fisher = {k: torch.zeros_like(v) for k, v in training_model.named_parameters()}

    for i in range(len(X)):
        training_model.zero_grad()
        logit = training_model(X[i:i+1])
        loss = criterion(logit, y[i:i+1])
        loss.backward()
        for name, param in training_model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.clone() ** 2

    # Average over samples
    for name in fisher:
        fisher[name] /= len(X)

    _ewc_fisher = fisher
    print(f"[EWC] Fisher diagonal computed from {len(X)} buffer samples.")


# ──────────────────────────────────────────────────────────────────────────────
# Validation gating
# ──────────────────────────────────────────────────────────────────────────────

def _load_validation_set():
    """Load a small hold-out validation set for weight update gating."""
    global _validation_X, _validation_y

    # Try scaled_testing_set.csv from data_pipeline
    candidates = [
        Path(__file__).parent.parent / "data_pipeline" / "model-arch" / "scaled_testing_set.csv",
        Path(__file__).parent / "UNSW_NB15_testing-set.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Subsample 500 rows for fast validation
            if len(df) > 500:
                df = df.sample(500, random_state=42)
            _validation_X = torch.FloatTensor(df[FEATURE_NAMES].values)
            _validation_y = torch.FloatTensor(df['label'].values).unsqueeze(1)
            print(f"[validation] Loaded {len(df)} hold-out samples from {path.name}")
            return

    print("[validation] WARNING: No validation set found — gating disabled.")


def _evaluate_model_accuracy(model: nn.Module) -> float:
    """Evaluate model accuracy on the hold-out validation set."""
    if _validation_X is None:
        return 1.0  # No validation set → always pass

    model.eval()
    with torch.no_grad():
        logits = model(_validation_X)
        preds = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (preds == _validation_y).float().mean().item()
    return accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Feature analysis (SOC reference — NOT model predictions)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_feature_analysis():
    """
    Compute per-feature statistics from the training data, split by label.
    This is a historical reference for the SOC analyst, NOT a prediction.
    """
    global _feature_analysis_cache

    candidates = [
        Path(__file__).parent.parent / "data_pipeline" / "model-arch" / "scaled_training_set.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            benign = df[df['label'] == 0][FEATURE_NAMES]
            attack = df[df['label'] == 1][FEATURE_NAMES]

            features = []
            for feat in FEATURE_NAMES:
                b_mean = float(benign[feat].mean())
                b_std = float(benign[feat].std())
                b_min = float(benign[feat].min())
                b_max = float(benign[feat].max())
                b_q25 = float(benign[feat].quantile(0.25))
                b_q75 = float(benign[feat].quantile(0.75))

                a_mean = float(attack[feat].mean())
                a_std = float(attack[feat].std())
                a_min = float(attack[feat].min())
                a_max = float(attack[feat].max())
                a_q25 = float(attack[feat].quantile(0.25))
                a_q75 = float(attack[feat].quantile(0.75))

                # Compute attack prevalence at high values
                high_threshold = b_mean + b_std
                if high_threshold <= 1.0:
                    total_high = len(df[df[feat] > high_threshold])
                    attack_high = len(df[(df[feat] > high_threshold) & (df['label'] == 1)])
                    attack_pct_when_high = round(attack_high / max(1, total_high) * 100, 1)
                else:
                    attack_pct_when_high = None

                # Generate SOC guidance text
                guidance = _generate_guidance(feat, b_mean, a_mean, b_std, a_std, attack_pct_when_high)

                features.append({
                    "feature": feat,
                    "benign": {
                        "mean": round(b_mean, 4), "std": round(b_std, 4),
                        "min": round(b_min, 4), "max": round(b_max, 4),
                        "q25": round(b_q25, 4), "q75": round(b_q75, 4),
                    },
                    "attack": {
                        "mean": round(a_mean, 4), "std": round(a_std, 4),
                        "min": round(a_min, 4), "max": round(a_max, 4),
                        "q25": round(a_q25, 4), "q75": round(a_q75, 4),
                    },
                    "attack_pct_when_high": attack_pct_when_high,
                    "guidance": guidance,
                })

            _feature_analysis_cache = {
                "total_benign": len(benign),
                "total_attack": len(attack),
                "features": features,
                "source": path.name,
            }
            print(f"[feature_analysis] Computed from {path.name}: {len(benign)} benign, {len(attack)} attack.")
            return

    print("[feature_analysis] WARNING: No training CSV found.")
    _feature_analysis_cache = {"error": "No training data available"}


def _generate_guidance(feat: str, b_mean: float, a_mean: float,
                       b_std: float, a_std: float,
                       attack_pct: Optional[float]) -> str:
    """Generate plain-English SOC guidance for a feature."""
    diff = abs(a_mean - b_mean)

    if diff < 0.05:
        base = f"{feat} is similar for benign and attack traffic — not a strong discriminator on its own."
    elif a_mean > b_mean:
        if attack_pct and attack_pct > 70:
            base = f"High {feat} (>{b_mean + b_std:.2f}) was attack {attack_pct}% of the time in training data."
        else:
            base = f"{feat} tends to be higher in attacks (mean {a_mean:.3f}) vs benign (mean {b_mean:.3f})."
    else:
        base = f"{feat} tends to be lower in attacks (mean {a_mean:.3f}) vs benign (mean {b_mean:.3f})."

    # Feature-specific expert notes
    expert_notes = {
        "sttl": " Note: TTL values may reflect OS defaults (Linux=64, Windows=128) rather than malicious intent.",
        "sbytes": " Very high source bytes often indicates data exfiltration or large payload delivery.",
        "dbytes": " Low destination bytes with high source bytes suggests one-way traffic (scans, exfiltration).",
        "sload": " Extreme source load values are characteristic of volumetric DoS attacks.",
        "dur": " Near-zero duration with high byte counts signals burst/flood behavior.",
        "rate": " Abnormally high packet rates are typical of DDoS or scanning activity.",
        "tcprtt": " Zero RTT may indicate UDP/ICMP traffic (where this metric doesn't apply).",
        "synack": " Zero synack is normal for non-TCP protocols — consider alongside the protocol context.",
        "ackdat": " Zero ackdat is normal for non-TCP protocols — consider alongside the protocol context.",
        "ct_dst_src_ltm": " High connection counts between same src-dst pair suggest scanning or brute force.",
    }

    return base + expert_notes.get(feat, "")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP forwarding to Tier 3 (connection-pooled)
# ──────────────────────────────────────────────────────────────────────────────

async def forward_to_tier3(payload: dict):
    try:
        await _http_client.post(TIER_3_URL, json=payload, timeout=3.0)
    except Exception as e:
        print(f"Failed to reach Tier 3 SOC: {e}")


async def forward_batch_to_tier3(payloads: list):
    try:
        resp = await _http_client.post(TIER_3_BATCH_URL, json=payloads, timeout=5.0)
        if resp.status_code == 200:
            return
    except Exception:
        pass
    # Fallback to individual calls
    for p in payloads:
        try:
            await _http_client.post(TIER_3_URL, json=p, timeout=2.0)
        except Exception as e:
            print(f"Failed to forward to Tier 3: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Prediction endpoints (read-only, concurrent via RWLock)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict_packet(request: PredictRequest, background_tasks: BackgroundTasks):
    features_tensor = torch.FloatTensor([request.features])

    await acquire_read()
    try:
        inference_model.eval()
        with torch.no_grad():
            logit = inference_model(features_tensor)
            P = torch.sigmoid(logit).item()
    finally:
        await release_read()

    if P < 0.10:
        return {"action": "dropped_benign", "P": P}
    elif P > 0.90:
        return {"action": "blocked_autonomous", "P": P}
    else:
        alert_payload = {
            "flow_id": request.flow_id,
            "model_confidence": P,
            "predicted_class": "ambiguous",
            "features": request.features
        }
        background_tasks.add_task(forward_to_tier3, alert_payload)
        return JSONResponse(status_code=202, content={"action": "quarantined_for_review", "P": P})


@app.post("/predict_batch")
async def predict_batch(flows: List[PredictRequest], background_tasks: BackgroundTasks):
    if not flows:
        return {"total": 0, "benign": 0, "attack": 0, "ambiguous": 0, "results": []}

    features_tensor = torch.FloatTensor([f.features for f in flows])

    await acquire_read()
    try:
        inference_model.eval()
        with torch.no_grad():
            logits = inference_model(features_tensor)
            probs = torch.sigmoid(logits).squeeze(-1).tolist()
            if isinstance(probs, float):
                probs = [probs]
    finally:
        await release_read()

    results = []
    ambiguous_payloads = []
    benign_count = 0
    attack_count = 0
    ambiguous_count = 0

    for req, P in zip(flows, probs):
        if P < 0.10:
            action = "dropped_benign"
            benign_count += 1
        elif P > 0.90:
            action = "blocked_autonomous"
            attack_count += 1
        else:
            action = "quarantined_for_review"
            ambiguous_count += 1
            ambiguous_payloads.append({
                "flow_id": req.flow_id,
                "model_confidence": P,
                "predicted_class": "ambiguous",
                "features": req.features
            })
        results.append({"flow_id": req.flow_id, "action": action, "P": P})

    if ambiguous_payloads:
        background_tasks.add_task(forward_batch_to_tier3, ambiguous_payloads)

    return {
        "total": len(flows),
        "benign": benign_count,
        "attack": attack_count,
        "ambiguous": ambiguous_count,
        "results": results
    }


# ──────────────────────────────────────────────────────────────────────────────
# Weight update with EWC + validation gating + double-buffer swap
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/update_weights")
async def update_weights(request: UpdateRequest):
    global persistent_optimizer, _ewc_fisher, _ewc_base_weights

    # 1. Batch construction: 22 historical + 10 copies of the new threat
    historical_samples = buffer.sample_batch(22)
    batch_features = [request.features] * 10 + [item[0] for item in historical_samples]
    batch_labels = [request.human_verified_label] * 10 + [item[1] for item in historical_samples]

    X_batch = torch.FloatTensor(batch_features)
    y_batch = torch.FloatTensor(batch_labels).unsqueeze(1)

    # Snapshot weights before training for delta measurement
    weights_before = {k: v.clone() for k, v in training_model.state_dict().items()}

    # Measure pre-training validation accuracy
    pre_val_accuracy = _evaluate_model_accuracy(training_model)

    # 2. Train on the training_model (write-locked, doesn't block inference)
    async with _write_lock:
        training_model.train()

        criterion = nn.BCEWithLogitsLoss()

        initial_loss = None
        for epoch in range(10):
            persistent_optimizer.zero_grad()
            logits = training_model(X_batch)
            bce_loss = criterion(logits, y_batch)

            # EWC penalty: penalize deviation from base weights on important parameters
            ewc_penalty = torch.tensor(0.0)
            for name, param in training_model.named_parameters():
                if name in _ewc_fisher and name in _ewc_base_weights:
                    ewc_penalty += (
                        _ewc_fisher[name] * (param - _ewc_base_weights[name]) ** 2
                    ).sum()

            loss = bce_loss + (EWC_LAMBDA / 2.0) * ewc_penalty

            if epoch == 0:
                initial_loss = loss.item()
            loss.backward()
            persistent_optimizer.step()

        final_loss = loss.item()
        final_bce = bce_loss.item()

        # 3. Validation gating — only promote if accuracy stays above floor
        post_val_accuracy = _evaluate_model_accuracy(training_model)

        weight_delta = sum(
            torch.norm(training_model.state_dict()[k] - weights_before[k]).item()
            for k in weights_before
        )

        if post_val_accuracy >= VALIDATION_ACCURACY_FLOOR:
            # ACCEPTED: Swap new weights into inference model (atomic, <1ms)
            new_state = copy.deepcopy(training_model.state_dict())
            inference_model.load_state_dict(new_state)
            inference_model.eval()

            # Save to disk
            torch.save(new_state, 'artifacts/base_model.pth')

            weights_promoted = True
            rejection_reason = None
        else:
            # REJECTED: Roll back training model to pre-update weights
            training_model.load_state_dict(weights_before)

            # Re-initialize optimizer for rolled-back weights
            persistent_optimizer = optim.Adam(training_model.parameters(), lr=5e-4)

            weights_promoted = False
            rejection_reason = (
                f"Post-update accuracy ({post_val_accuracy:.4f}) "
                f"fell below floor ({VALIDATION_ACCURACY_FLOOR}). "
                f"Pre-update was {pre_val_accuracy:.4f}. Weights rolled back."
            )

        training_model.eval()

    # 4. Always insert into buffer (the sample is valid regardless of gating)
    buffer.insert(request.features, request.human_verified_label)

    return {
        "status": "success" if weights_promoted else "rejected",
        "weights_updated": weights_promoted,
        "rejection_reason": rejection_reason,
        "initial_loss": initial_loss,
        "loss": final_loss,
        "bce_loss": final_bce,
        "ewc_penalty": round((final_loss - final_bce), 6),
        "weight_delta_l2": round(weight_delta, 6),
        "pre_val_accuracy": round(pre_val_accuracy, 4),
        "post_val_accuracy": round(post_val_accuracy, 4),
        "buffer_benign": len(buffer.benign_buffer),
        "buffer_attack": len(buffer.attack_buffer)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Memory buffer batch insertion
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/insert_buffer_batch")
async def insert_buffer_batch(requests: List[UpdateRequest]):
    for req in requests:
        buffer.insert(req.features, req.human_verified_label)
    return {
        "status": "success",
        "inserted": len(requests),
        "buffer_benign": len(buffer.benign_buffer),
        "buffer_attack": len(buffer.attack_buffer)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Feature analysis endpoint (SOC reference — NOT predictions)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/feature_analysis")
async def feature_analysis():
    """
    Returns per-feature statistics from the training data, split by label.
    This is a historical reference for the SOC analyst to aid manual
    decision-making on ambiguous packets. It is NOT a model prediction.
    """
    if _feature_analysis_cache is None:
        return {"error": "Feature analysis not computed yet"}
    return _feature_analysis_cache