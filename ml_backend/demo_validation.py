import pandas as pd
import joblib
import torch
import requests
import uuid
import sys
from train_offline import NIDS_MLP

FEATURES = ['sttl', 'sbytes', 'dbytes', 'sload', 'dur', 'rate', 'tcprtt', 'synack', 'ackdat', 'ct_dst_src_ltm']

print("--- 1. DYNAMIC AMBIGUOUS PACKET EXTRACTION ---")
try:
    df = pd.read_csv('UNSW_NB15_testing-set.csv')
    scaler = joblib.load('artifacts/scaler.pkl')
    
    # Load the current state of the model
    model = NIDS_MLP()
    model.load_state_dict(torch.load('artifacts/base_model.pth', weights_only=True))
    model.eval()
except Exception as e:
    print(f"Error loading artifacts: {e}. Did you run train_offline.py?")
    sys.exit(1)

# Filter for real attacks and score them offline
attacks = df[df['label'] == 1]
X_attacks = scaler.transform(attacks[FEATURES])

with torch.no_grad():
    P_vals = torch.sigmoid(model(torch.FloatTensor(X_attacks))).squeeze().numpy()

# Find currently ambiguous attacks (0.10 <= P <= 0.90)
ambiguous_mask = (P_vals >= 0.10) & (P_vals <= 0.90)
ambiguous_indices = ambiguous_mask.nonzero()[0]

if len(ambiguous_indices) == 0:
    print("Error: No ambiguous attacks found in the test set. Model is too confident.")
    sys.exit(1)

# Grab the first valid candidate
target_idx = ambiguous_indices[0]
target_features_scaled = X_attacks[target_idx].tolist()
initial_offline_P = P_vals[target_idx]

print(f"Found {len(ambiguous_indices)} currently ambiguous attacks.")
print(f"Selected packet offline P-score: {initial_offline_P:.4f}\n")

print("--- 2. API END-TO-END VALIDATION ---")
flow_id = str(uuid.uuid4())
payload = {
    "flow_id": flow_id,
    "timestamp": "2026-09-12T17:15:00",
    "tier1_anomaly_score": 0.8,
    "features": target_features_scaled
}

try:
    print("Baseline Prediction (Hitting API)...")
    base_resp = requests.post('http://localhost:5000/predict', json=payload).json()
    print(f"API Response: {base_resp}")
    
    if base_resp.get("action") != "quarantined_for_review":
        print("\n[!] WARNING: API did not quarantine the packet. This should not happen if extraction worked.")

    print("\nSOC Label Update (Human Analyst confirms Attack = 1.0)...")
    update_payload = payload.copy()
    update_payload["human_verified_label"] = 1.0
    update_resp = requests.post('http://localhost:5000/update_weights', json=update_payload).json()
    print(f"Update Status: Loss = {update_resp['loss']:.4f}")

    print("\nRe-testing the exact same packet...")
    new_pred = requests.post('http://localhost:5000/predict', json=payload).json()
    print(f"New Prediction: {new_pred}")
    
    if new_pred.get("P", 0) > 0.90:
        print("\n[SUCCESS] Packet successfully pushed into 'blocked_autonomous' zone!")
    else:
        print("\n[WARNING] Packet moved, but did not cross the 0.90 threshold.")

except requests.exceptions.ConnectionError:
    print("\n[!] ERROR: Connection Refused. You must start the FastAPI server first!")
    print("Run this in a separate terminal: uvicorn main:app --port 5000")