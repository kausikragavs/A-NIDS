"""
Throwaway stand-in for Member 2's /update_weights endpoint, so you can
test the full resolve_alert -> Member 2 round trip before their real
service exists.

Run this in its own terminal BEFORE testing resolve_alert:
    uvicorn mock_member2:app --port 5000
"""
from fastapi import FastAPI

app = FastAPI(title="Mock Member 2 (/update_weights stub)")


@app.post("/update_weights")
def update_weights(payload: dict):
    print("Mock Member 2 received update_weights call:", payload)
    return {"ok": True, "message": "weights updated (mock)"}
