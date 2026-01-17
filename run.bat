@echo off
echo Starting Bridge Laws Chatbot...
echo.
echo Make sure you have:
echo 1. Created a .env file with your ANTHROPIC_API_KEY
echo 2. Installed dependencies: pip install -r requirements.txt
echo.
echo The chatbot will be available at http://localhost:8000
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
