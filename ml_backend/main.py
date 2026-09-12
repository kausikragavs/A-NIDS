import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import httpx

from train_offline import NIDS_MLP
from memory_buffer import MemoryBuffer

app = FastAPI()

# Global states
model = NIDS_MLP()
model_lock = asyncio.Lock()
buffer = MemoryBuffer()

class PredictRequest(BaseModel):
    flow_id: str
    timestamp: str
    tier1_anomaly_score: float
    features: List[float] = Field(..., min_length=10, max_length=10)

class UpdateRequest(BaseModel):
    flow_id: str
    human_verified_label: float
    features: List[float] = Field(..., min_length=10, max_length=10)

TIER_3_URL = "http://localhost:5001/review_alert"

@app.on_event("startup")
async def startup_event():
    model.load_state_dict(torch.load('artifacts/base_model.pth', weights_only=True))
    model.eval()

async def forward_to_tier3(payload: dict):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(TIER_3_URL, json=payload)
        except Exception as e:
            print(f"Failed to reach Tier 3 SOC: {e}")

@app.post("/predict")
async def predict_packet(request: PredictRequest, background_tasks: BackgroundTasks):
    features_tensor = torch.FloatTensor([request.features])
    
    async with model_lock:
        model.eval()
        with torch.no_grad():
            logit = model(features_tensor)
            P = torch.sigmoid(logit).item()
            
    if P < 0.10:
        return {"action": "dropped_benign", "P": P}
    elif P > 0.90:
        return {"action": "blocked_autonomous", "P": P}
    else:
        confidence_display = abs(P - 0.5) * 2 
        alert_payload = {
            "flow_id": request.flow_id,
            "model_confidence": confidence_display,
            "predicted_class": "ambiguous",
            "features": request.features
        }
        background_tasks.add_task(forward_to_tier3, alert_payload)
        return JSONResponse(status_code=202, content={"action": "quarantined_for_review", "P": P})

@app.post("/update_weights")
async def update_weights(request: UpdateRequest):
    # Phase 4 - Continual Learning Loop (Aggressive One-Shot Demo Tuning)
    
    # 1. Batch construction: 22 historical + 10 copies of the new threat
    historical_samples = buffer.sample_batch(22)
    batch_features = [request.features] * 10 + [item[0] for item in historical_samples]
    batch_labels = [request.human_verified_label] * 10 + [item[1] for item in historical_samples]
    
    X_batch = torch.FloatTensor(batch_features)
    y_batch = torch.FloatTensor(batch_labels).unsqueeze(1)
    
    async with model_lock:
        model.train()
        
        # Fresh Adam optimizer with a stronger learning rate for the 10-epoch push
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
        final_loss = loss.item()
        torch.save(model.state_dict(), 'artifacts/base_model.pth')
        model.eval()
        
    buffer.insert(request.features, request.human_verified_label)
    
    return {
        "status": "success", 
        "loss": final_loss, 
        "buffer_benign": len(buffer.benign_buffer),
        "buffer_attack": len(buffer.attack_buffer)
    }