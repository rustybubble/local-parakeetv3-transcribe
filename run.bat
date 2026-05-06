@echo off
REM Launcher for the local whisper.cpp transcription web app.
REM
REM First run: starts the server, downloads the model (~150 MB, one-time),
REM opens the browser. Subsequent runs while the server window is still open
REM just open the browser (no model reload). Close this window or press
REM Ctrl+C to stop the server.
REM
REM If you haven't run setup.bat yet, this will tell you to do that first.

cd /d "%~dp0"

REM ── Make sure setup has been done ───────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo .venv\ not found -- looks like you haven't run setup.bat yet.
    echo Double-click setup.bat once, then double-click run.bat.
    echo.
    pause
    exit /b 1
)

REM ── Reuse an already-running server so the model stays loaded ───────────
.venv\Scripts\python.exe -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/ready', timeout=2).getcode()==200 else 1)" 2>nul
if %errorlevel%==0 (
  echo Server is already running. Opening browser...
  start "" http://127.0.0.1:5000
  exit /b 0
)

echo Starting whisper.cpp transcription server...
echo The browser tab will open in a second; the model loads in the background.
echo (First run: also downloads the ~150 MB model into .\models -- one-time.)
echo.
echo Leave this window open while you use the app.
echo Re-running run.bat while it's open just reopens the browser (no model reload).
echo Close it (or press Ctrl+C) to stop the server.
echo ============================================================
echo.

REM -u = unbuffered stdout so progress prints appear in real time.
.venv\Scripts\python.exe -u app.py

echo.
echo ============================================================
echo Server stopped. Press any key to close this window.
pause >nul
