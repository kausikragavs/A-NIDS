# This script launches all 3 services for the TTA-NIDS project

$ml_backend_dir = "d:\vit\SEM3\AIDS\ml_backend"
$webpage_dir = "d:\vit\SEM3\AIDS\webpage"

Write-Host "Starting ML Backend (Port 5000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $ml_backend_dir; uvicorn main:app --port 5000"

Write-Host "Starting Tier 3 SOC API (Port 5001)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $webpage_dir; uvicorn backend:app --port 5001"

Write-Host "Starting Streamlit Dashboard..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $webpage_dir; streamlit run dashboard.py"

Write-Host "All services started in separate windows."
