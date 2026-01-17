# Bridge Laws Chatbot Startup Script

Write-Host "Bridge Laws Chatbot" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Check for .env file
if (-not (Test-Path ".env")) {
    Write-Host "Warning: .env file not found!" -ForegroundColor Yellow
    Write-Host "Please create a .env file with your ANTHROPIC_API_KEY" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Example:" -ForegroundColor Gray
    Write-Host "ANTHROPIC_API_KEY=sk-ant-..." -ForegroundColor Gray
    Write-Host ""
}

# Check if uvicorn is installed
$uvicornCheck = pip show uvicorn 2>$null
if (-not $uvicornCheck) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Starting server at http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

# Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
