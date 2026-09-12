# Data Pipeline Integration Architecture for ML Backend

This document outlines the architecture and integration instructions for **Member 2 (ML Backend)** to consume and process data coming from the **Data Pipelining (Member 1)** layer.

## Overview
The Data Pipeline layer processes incoming network flows, applies necessary preprocessing (such as Min-Max scaling), and constructs a standardized payload that is pushed to the ML Backend via HTTP POST requests for prediction/inference.

Your ML backend needs to expose an endpoint to accept this payload, run the necessary inference (using the subsequent models), and handle the response.

## 1. Network Interface

The ML Backend should expose an HTTP server. Currently, the pipeline is configured to point to:
- **Protocol:** HTTP POST
- **URL (Default):** `http://localhost:5000/predict`

> **Note:** If the backend port or endpoint path changes, please notify Member 1 to update the data pipeline's injector configuration.

## 2. Payload Contract

The pipeline sends a JSON payload representing a single flow or an injected zero-day attack. The structure is fixed to ensure consistency.

### JSON Payload Structure

```json
{
  "flow_id": "string",
  "timestamp": "ISO 8601 string",
  "tier1_anomaly_score": "float",
  "features": ["float"] // Array of 10 scaled numeric features
}
```

### Fields Description

- `flow_id`: A unique identifier for the flow. (e.g., `ZERO-DAY-123456...`). You can use this for logging and tracking.
- `timestamp`: UTC timestamp of when the flow/attack was generated (e.g., `2026-09-12T22:00:00.000000+00:00`).
- `tier1_anomaly_score`: A float representing the anomaly score from the initial filtering tier.
- `features`: A JSON array of exactly **10 numeric features**.

## 3. Feature Ordering and Scaling

The data pipeline has already normalized the raw values before they hit your backend. 
**You do not need to apply another scaler to these values in the ML backend.**

### The 10 Features (In Order)
The `features` array contains the following attributes in this exact sequence:
1. `sttl`
2. `sbytes`
3. `dbytes`
4. `sload`
5. `dur`
6. `rate`
7. `tcprtt`
8. `synack`
9. `ackdat`
10. `ct_dst_src_ltm`

### Scaling Specifications
- **Method:** Min-Max Scaling
- **Bounds:** Strictly clamped between `[0.0, 1.0]`.
- **Reference:** The scaling ranges are managed in the `scaler_ranges.json` file in the pipeline's architecture directory.

## 4. Integration Steps for Member 2

To successfully integrate this pipeline into the ML Backend, follow these steps:

1. **Set Up the Endpoint:** Create a route (e.g., in Flask or FastAPI) that listens for POST requests at `/predict`.
2. **Parse the Payload:** Extract the JSON payload from the request body.
3. **Validate:** 
   - Ensure the `features` array has exactly 10 elements.
   - All elements should be floats.
4. **Model Inference:** Pass the `features` array into your PyTorch/TensorFlow model (e.g., `tier1_autoencoder.pth` or subsequent classifier). 
5. **Response:** Return a 200 OK HTTP response with your prediction results. For example:
   ```json
   {
       "flow_id": "ZERO-DAY-...",
       "status": "success",
       "prediction": 1,
       "confidence": 0.98
   }
   ```

## 5. Testing the Integration

To verify the integration is working:
1. Start your ML Backend server on `localhost:5000`.
2. Have Member 1 run the `Zero-day-injector.py` script.
3. Check your backend logs to verify that the payload was received correctly, parsed, and that your model generated a prediction without crashing.

> [!WARNING]
> The Zero-Day Injector simulates extreme "out-of-bounds" network conditions. While the pipeline clamps these values between `0.0` and `1.0`, your models should be robust enough to handle the edge cases of precisely `0.0` or `1.0` across multiple features simultaneously.
