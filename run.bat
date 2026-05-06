@echo off
REM Launcher for the local faster-whisper transcription web app.
REM
REM First run: starts the server, loads the model (~10s), opens the browser.
REM Subsequent runs while the server window is still open: just opens the browser
REM (no model reload). Close this window or press Ctrl+C to stop the server.

cd /d "%~dp0"

REM ── Reuse an already-running server so the model stays loaded ───────────────
python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/ready', timeout=2).getcode()==200 else 1)" 2>nul
if %errorlevel%==0 (
  echo Server is already running. Opening browser...
  start "" http://127.0.0.1:5000
  exit /b 0
)

echo Starting Whisper transcription server...
echo The browser tab will open in a second; the model loads in the background (~10s).
echo.
echo Leave this window open while you use the app.
echo Re-running run.bat while it's open just reopens the browser (no model reload).
echo Close it (or press Ctrl+C) to stop the server.
echo ============================================================
echo.

REM -u = unbuffered stdout so progress prints appear in real time.
python -u app.py

echo.
echo ============================================================
echo Server stopped. Press any key to close this window.
pause >nul
