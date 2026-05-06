@echo off
REM One-time setup for the local whisper.cpp transcription app.
REM Creates a Python virtual environment in .venv\ and installs dependencies.
REM
REM After this finishes successfully, use run.bat for daily use.

cd /d "%~dp0"

echo ============================================================
echo   whisper.cpp transcription app -- first-time setup
echo ============================================================
echo.

REM ── Check Python ────────────────────────────────────────────────────────
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not on PATH.
    echo.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo and make sure to tick "Add Python to PATH" during install.
    echo Then close this window and double-click setup.bat again.
    echo.
    pause
    exit /b 1
)

REM ── Check ffmpeg (warn, don't block) ────────────────────────────────────
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo WARNING: ffmpeg is not on PATH.
    echo You'll need it to transcribe most files (mkv/mp4/mp3/etc.).
    echo Install it with:    winget install ffmpeg
    echo Then close and reopen the terminal so ffmpeg is on PATH.
    echo Continuing setup so the Python side is ready...
    echo.
)

REM ── Create venv ─────────────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv\ ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM ── Install dependencies ────────────────────────────────────────────────
echo.
echo Installing dependencies into .venv\ (this can take a minute) ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete.
echo   Double-click run.bat to start the app.
echo ============================================================
pause
