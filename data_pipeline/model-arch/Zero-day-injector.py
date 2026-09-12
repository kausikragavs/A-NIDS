import asyncio
import aiohttp
import json
import uuid
import sys
from datetime import datetime, timezone

# 1. The Synthetic Zero-Day Attack (Raw Values)
# Tweak these extreme values to simulate the exact concept drift you want to demonstrate
RAW_ATTACK = {
    "sttl": 255.0,                  # Max Time-to-Live
    "sbytes": 50000000.0,           # 50MB payload (exceeds training max of 14.3MB)
    "dbytes": 0.0,                  # No response bytes
    "sload": 9000000000.0,          # Massive load (exceeds training max)
    "dur": 0.0001,                  # Microsecond duration
    "rate": 2000000.0,              # Double the max training rate
    "tcprtt": 0.0,
    "synack": 0.0,
    "ackdat": 0.0,
    "ct_dst_src_ltm": 63.0          # Max connection count
}

FEATURE_ORDER = [
    'sttl', 'sbytes', 'dbytes', 'sload', 'dur', 
    'rate', 'tcprtt', 'synack', 'ackdat', 'ct_dst_src_ltm'
]

from pathlib import Path

def load_and_scale():
    """Reads Member 2's JSON scaler and applies MinMax scaling with strict [0.0, 1.0] clamping."""
    script_dir = Path(__file__).parent
    json_path = script_dir / "scaler_ranges.json"
    if not json_path.exists():
        json_path = script_dir.parent / "model-arch" / "scaler_ranges.json"

    try:
        with open(json_path, "r") as f:
            ranges = json.load(f)
    except FileNotFoundError:
        print("[FATAL] scaler_ranges.json not found.")
        sys.exit(1)

    scaled_features = []
    for key in FEATURE_ORDER:
        raw_val = RAW_ATTACK[key]
        r_min = ranges[key]["min"]
        r_max = ranges[key]["max"]
        
        # MinMax calculation
        if r_max == r_min:
            scaled_val = 0.0
        else:
            scaled_val = (raw_val - r_min) / (r_max - r_min)
            
        # Hard clamp to [0.0, 1.0] to prevent downstream hallucination
        scaled_val = max(0.0, min(1.0, scaled_val))
        scaled_features.append(scaled_val)
        
    return scaled_features

async def inject_zero_day():
    print("=== TTA-NIDS Zero-Day Injector (Phase 5) ===")
    
    scaled_vector = load_and_scale()
    
    # 2. Build the exact contract payload
    payload = {
        "flow_id": f"ZERO-DAY-{str(uuid.uuid4())[:27]}", # Custom prefix to stand out in logs
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier1_anomaly_score": 9.9999, # Forced extremely high score to guarantee attention
        "features": scaled_vector
    }
    
    # Update this to Member 2's real URL once they provide it
    target_url = "http://localhost:5000/predict"
    
    print(f"[*] Payload prepared. Targeting: {target_url}")
    print(f"[*] Synthetic Features (Scaled): {[round(x, 4) for x in scaled_vector]}")
    print("[*] Firing POST request...\n")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(target_url, json=payload, timeout=3.0) as response:
                if response.status == 200:
                    print(f"[\033[91mBOOM\033[0m] Zero-Day successfully injected! Target returned 200 OK.")
                else:
                    print(f"[FAIL] Injection rejected. Status: {response.status}")
        except Exception as e:
            print(f"[ERROR] Could not connect to target: {e}")

if __name__ == "__main__":
    asyncio.run(inject_zero_day())