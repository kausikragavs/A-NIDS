import requests
import uuid
import pandas as pd
import joblib
from datetime import datetime

print("Loading real packets from test set...")
FEATURES = ['sttl', 'sbytes', 'dbytes', 'sload', 'dur', 'rate', 'tcprtt', 'synack', 'ackdat', 'ct_dst_src_ltm']

# Load the test data and your exact offline scaler
df = pd.read_csv('UNSW_NB15_testing-set.csv')
scaler = joblib.load('artifacts/scaler.pkl')

# Grab the first real benign row and first real attack row
benign_row = df[df['label'] == 0].iloc[0][FEATURES].values.reshape(1, -1)
attack_row = df[df['label'] == 1].iloc[0][FEATURES].values.reshape(1, -1)

# Scale them identically to the training data
benign_scaled = scaler.transform(benign_row)[0].tolist()
attack_scaled = scaler.transform(attack_row)[0].tolist()

real_packets = [
    {"desc": "Real Benign Flow", "features": benign_scaled},
    {"desc": "Real Attack Flow", "features": attack_scaled}
]

for packet in real_packets:
    payload = {
        "flow_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "tier1_anomaly_score": 0.5,
        "features": packet["features"]
    }
    
    print(f"\nSending {packet['desc']}...")
    response = requests.post("http://localhost:5000/predict", json=payload)
    print(f"Response ({response.status_code}): {response.json()}")