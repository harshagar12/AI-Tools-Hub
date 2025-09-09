@echo off
echo Starting Full Stack Application...

REM Start backend in a new command prompt window (separate process)
start "Backend Server" cmd /c ".\venv\Scripts\activate && cd /d venv && uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in another new command prompt window (separate process)
start "Frontend Server" cmd /c "cd /d frontend && npm run dev"

REM Wait a moment for frontend to start
timeout /t 5 /nobreak >nul

REM Open the application in default browser
start "" "http://localhost:3000/"

echo Both servers are running in separate windows...
echo Backend: http://localhost:8001 (check Backend Server window)
echo Frontend: http://localhost:3000 (check Frontend Server window)
echo Close the respective windows to stop each server.
pauses