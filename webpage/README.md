# Tier 3 SOC API — Step 1 (Backend)

Tested end-to-end already. Confirmed:
- P=0.10 and P=0.90 both classify as **Ambiguous** (exact boundary spec).
- `/pending_alerts` returns only Ambiguous flows.
- `/resolve_alert/{flow_id}` forwards `{flow_id, human_verified_label, features}`
  to Member 2's `/update_weights`, and only removes the alert from the
  queue after Member 2 confirms success.
- Duplicate `flow_id` submissions update the existing record in place.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

## Run (3 terminals)

**Terminal 1 — mock Member 2** (stand-in until the real ML service exists):
```bash
uvicorn mock_member2:app --port 5000
```

**Terminal 2 — your backend:**
```bash
uvicorn backend:app --port 5001 --reload
```

**Terminal 3 — send test traffic:**
```bash
python mock_client.py
```

You should see 8 flows sent (spanning Benign/Ambiguous/Attack), `pending_alerts`
showing only the 4 Ambiguous ones, and one of them successfully resolved through
to the mock Member 2.

Once your backend is running, open `http://localhost:5001/docs` for an
interactive Swagger UI — useful for manually testing endpoints without
writing curl commands.

## Files

| File | Purpose |
|---|---|
| `status_logic.py` | The ONE place the P-thresholds live. Import this everywhere — backend, frontend, future modules. Never re-implement the comparison inline. |
| `backend.py` | Your Module A — the SOC API. |
| `mock_member2.py` | Throwaway stub for Member 2's `/update_weights`, so you're never blocked waiting on your teammate. Delete once their real service is ready. |
| `mock_client.py` | Sends test traffic across all three buckets, including exact boundary values (0.10, 0.90), and exercises the resolve flow. |

## Swap in the real Member 2

Nothing to change on your end — `backend.py` already points at
`http://localhost:5000/update_weights`. Just stop `mock_member2.py` and
start their real service on port 5000.

## Next: Step 2 (Streamlit frontend)

Say the word and I'll build the dashboard on top of this — polling
`/pending_alerts`, live feed from `/all_events`, the deviation-highlighting
panel, and the two label buttons wired to `/resolve_alert/{flow_id}`, all
using `get_alert_status()` from `status_logic.py` for every bit of color.
