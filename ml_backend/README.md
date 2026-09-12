# TTA-NIDS Tier 2: Continual Deep Learner

This repository contains the standalone Tier 2 system for the Tri-Tiered Adaptive NIDS architecture. It strictly implements the data contracts and Continual Learning loop outlined in the hackathon master prompt.

## Prerequisites
```bash
python -m pip install torch pandas numpy scikit-learn fastapi uvicorn httpx "pydantic>=2.0" joblib