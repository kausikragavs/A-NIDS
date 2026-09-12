#!/usr/bin/env python3
"""
stream_injector_100.py
High-throughput packet injector for TTA-NIDS (Anomaly Network Intrusion Detection System).
Streams 100 packets at a time to the ML Backend (localhost:5000), supporting both
real pre-scaled UNSW-NB15 test sets and synthetic zero-day attack drifts.

Usage:
    python stream_injector_100.py --batch-size 100 --batches 5 --interval 1.0
    python stream_injector_100.py --batch-size 100 --continuous
    python stream_injector_100.py --mode mixed --batch-size 100
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import httpx
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

FEATURE_ORDER = [
    'sttl', 'sbytes', 'dbytes', 'sload', 'dur', 
    'rate', 'tcprtt', 'synack', 'ackdat', 'ct_dst_src_ltm'
]

# Baseline synthetic zero-day pattern (out-of-distribution drift)
SYNTHETIC_ZERO_DAY_BASE = {
    "sttl": 0.99,
    "sbytes": 0.95,
    "dbytes": 0.01,
    "sload": 0.98,
    "dur": 0.0001,
    "rate": 0.99,
    "tcprtt": 0.0,
    "synack": 0.0,
    "ackdat": 0.0,
    "ct_dst_src_ltm": 0.95
}

class StreamInjector:
    def __init__(
        self,
        batch_size: int = 100,
        target_url: str = "http://localhost:5000/predict_batch",
        fallback_url: str = "http://localhost:5000/predict",
        mode: str = "dataset",
        data_path: str = None
    ):
        self.batch_size = batch_size
        self.target_url = target_url
        self.fallback_url = fallback_url
        self.mode = mode
        self.client = None
        self.dataset_features = None
        self.dataset_labels = None
        self.current_idx = 0
        
        # Cumulative stats
        self.total_sent = 0
        self.total_benign = 0
        self.total_attack = 0
        self.total_ambiguous = 0
        
        self._load_data(data_path)

    def _load_data(self, data_path: str = None):
        """Loads dataset if in dataset or mixed mode."""
        if self.mode in ("dataset", "mixed"):
            script_dir = Path(__file__).parent
            candidate_paths = [
                data_path,
                script_dir / "scaled_testing_set.csv",
                script_dir / "scaled_training_set.csv",
                script_dir.parent / "scaled_testing_set.csv",
            ]
            
            chosen_path = None
            for p in candidate_paths:
                if p and os.path.exists(p):
                    chosen_path = p
                    break
                    
            if chosen_path:
                print(f"[*] Loading network traffic dataset from: {chosen_path}")
                df = pd.read_csv(chosen_path)
                self.dataset_features = df[FEATURE_ORDER].to_numpy(dtype=np.float32)
                if 'label' in df.columns:
                    self.dataset_labels = df['label'].to_numpy(dtype=np.float32)
                print(f"[+] Loaded {len(self.dataset_features):,} packet vectors ready for streaming.")
            else:
                print("[!] Warning: Scaled CSV not found. Falling back to synthetic stream.")
                self.mode = "synthetic_zeroday"

    def _generate_synthetic_zeroday(self) -> List[float]:
        """Generates jittered synthetic zero-day packet features."""
        vector = []
        for feat in FEATURE_ORDER:
            base = SYNTHETIC_ZERO_DAY_BASE[feat]
            jitter = np.random.uniform(-0.05, 0.05)
            val = float(np.clip(base + jitter, 0.0, 1.0))
            vector.append(val)
        return vector

    def get_next_batch(self) -> List[Dict[str, Any]]:
        """Constructs a batch of exactly batch_size (default 100) packet payloads."""
        batch = []
        now_iso = datetime.now(timezone.utc).isoformat()
        
        for _ in range(self.batch_size):
            flow_uuid = str(uuid.uuid4())[:8]
            
            if self.mode == "dataset" and self.dataset_features is not None:
                idx = self.current_idx % len(self.dataset_features)
                self.current_idx += 1
                features = [float(x) for x in self.dataset_features[idx]]
                flow_id = f"FLOW-{flow_uuid}"
                score = 0.5
            elif self.mode == "synthetic_zeroday":
                features = self._generate_synthetic_zeroday()
                flow_id = f"ZERO-DAY-{flow_uuid}"
                score = 0.95
            elif self.mode == "mixed":
                # 90% dataset, 10% synthetic zero-day
                if np.random.rand() < 0.10:
                    features = self._generate_synthetic_zeroday()
                    flow_id = f"ZERO-DAY-{flow_uuid}"
                    score = 0.95
                elif self.dataset_features is not None:
                    idx = self.current_idx % len(self.dataset_features)
                    self.current_idx += 1
                    features = [float(x) for x in self.dataset_features[idx]]
                    flow_id = f"FLOW-{flow_uuid}"
                    score = 0.5
                else:
                    features = self._generate_synthetic_zeroday()
                    flow_id = f"FLOW-{flow_uuid}"
                    score = 0.5
            else:
                features = self._generate_synthetic_zeroday()
                flow_id = f"SYNTH-{flow_uuid}"
                score = 0.5
                
            batch.append({
                "flow_id": flow_id,
                "timestamp": now_iso,
                "tier1_anomaly_score": score,
                "features": features
            })
            
        return batch

    async def _send_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sends batch of packets to the backend via POST /predict_batch or fallback."""
        start_time = time.perf_counter()
        
        # Primary: Batch endpoint
        try:
            resp = await self.client.post(self.target_url, json=batch, timeout=10.0)
            if resp.status_code == 200:
                elapsed = time.perf_counter() - start_time
                data = resp.json()
                return {
                    "ok": True,
                    "elapsed": elapsed,
                    "benign": data.get("benign", 0),
                    "attack": data.get("attack", 0),
                    "ambiguous": data.get("ambiguous", 0),
                    "total": len(batch)
                }
        except Exception:
            pass

        # Fallback: Concurrent individual /predict requests
        try:
            tasks = [
                self.client.post(self.fallback_url, json=pkt, timeout=5.0)
                for pkt in batch
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - start_time
            
            benign = 0
            attack = 0
            ambiguous = 0
            
            for r in responses:
                if isinstance(r, httpx.Response):
                    if r.status_code == 202:
                        ambiguous += 1
                    elif r.status_code == 200:
                        act = r.json().get("action", "")
                        if "benign" in act:
                            benign += 1
                        else:
                            attack += 1
                            
            return {
                "ok": True,
                "elapsed": elapsed,
                "benign": benign,
                "attack": attack,
                "ambiguous": ambiguous,
                "total": len(batch)
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "total": len(batch)}

    async def run(self, num_batches: int = 5, interval: float = 1.0, continuous: bool = False):
        """Streams batches of 100 packets at the configured interval."""
        print(f"\n=======================================================")
        print(f"🚀 TTA-NIDS 100-Packet Stream Injector Active")
        print(f"=======================================================")
        print(f" • Mode:             {self.mode.upper()}")
        print(f" • Batch Size:       {self.batch_size} packets per stream")
        print(f" • Target Endpoint:  {self.target_url}")
        print(f" • Stream Interval:  {interval:.2f}s")
        print(f" • Batch Limit:      {'Continuous' if continuous else num_batches}")
        print(f"=======================================================\n")
        
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=120, max_keepalive_connections=50)) as client:
            self.client = client
            batch_count = 0
            
            while continuous or batch_count < num_batches:
                batch_count += 1
                batch = self.get_next_batch()
                
                result = await self._send_batch(batch)
                
                if result.get("ok"):
                    elapsed = result["elapsed"]
                    throughput = len(batch) / elapsed if elapsed > 0 else 0
                    b = result["benign"]
                    a = result["attack"]
                    amb = result["ambiguous"]
                    
                    self.total_sent += len(batch)
                    self.total_benign += b
                    self.total_attack += a
                    self.total_ambiguous += amb
                    
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Stream #{batch_count:03d}: {len(batch)} pkts in {elapsed*1000:.1f}ms ({throughput:6.0f} pkt/s) | "
                        f"🟢 Benign: {b:3d} | 🔴 Blocked: {a:3d} | 🟡 Quarantined (SOC): {amb:3d}"
                    )
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stream #{batch_count:03d} FAILED: {result.get('error')}")

                if (continuous or batch_count < num_batches) and interval > 0:
                    await asyncio.sleep(interval)

        print(f"\n-------------------------------------------------------")
        print(f"📊 STREAMING COMPLETE — Cumulative Summary:")
        print(f" • Total Packets Streamed: {self.total_sent}")
        print(f" • 🟢 Autonomous Benign:  {self.total_benign} ({self.total_benign/max(1, self.total_sent)*100:.1f}%)")
        print(f" • 🔴 Autonomous Block:   {self.total_attack} ({self.total_attack/max(1, self.total_sent)*100:.1f}%)")
        print(f" • 🟡 Tier 3 SOC Reviews: {self.total_ambiguous} ({self.total_ambiguous/max(1, self.total_sent)*100:.1f}%)")
        print(f"-------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="TTA-NIDS 100-Packet Stream Injector")
    parser.add_argument("--batch-size", type=int, default=100, help="Packets streamed per burst (default: 100)")
    parser.add_argument("--batches", type=int, default=5, help="Number of 100-packet streams to send (default: 5)")
    parser.add_argument("--interval", type=float, default=1.0, help="Delay between bursts in seconds (default: 1.0)")
    parser.add_argument("--continuous", action="store_true", help="Stream indefinitely until interrupted")
    parser.add_argument("--mode", choices=["dataset", "synthetic_zeroday", "mixed"], default="dataset",
                        help="Streaming traffic mode (default: dataset)")
    parser.add_argument("--url", default="http://localhost:5000/predict_batch", help="Target API endpoint")

    args = parser.parse_args()

    injector = StreamInjector(
        batch_size=args.batch_size,
        target_url=args.url,
        mode=args.mode
    )

    try:
        asyncio.run(injector.run(
            num_batches=args.batches,
            interval=args.interval,
            continuous=args.continuous
        ))
    except KeyboardInterrupt:
        print("\n[!] Stream stopped by user.")


if __name__ == "__main__":
    main()
