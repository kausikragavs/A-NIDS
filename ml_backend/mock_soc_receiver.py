from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Trivial second FastAPI app on port 5001[cite: 2]
app = FastAPI()

class ReviewAlert(BaseModel):
    flow_id: str
    model_confidence: float
    predicted_class: str
    features: List[float]

@app.post("/review_alert")
async def receive_alert(alert: ReviewAlert):
    print(f"\n[TIER 3 MOCK] Received Alert for Flow ID: {alert.flow_id}")
    print(f"Human-facing Confidence: {alert.model_confidence:.4f}")
    print(f"Features: {alert.features}\n")
    return {"status": "received"}

# Run with: uvicorn mock_soc_receiver:app --port 5001