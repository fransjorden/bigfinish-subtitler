@echo off
echo Big Finish Caption Sync - Setup
echo ================================
echo.

REM Check Python version
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate and install
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Try to install faster-whisper (optional)
echo.
echo Installing faster-whisper for better performance...
pip install faster-whisper 2>NUL
if errorlevel 1 (
    echo Note: faster-whisper not available for your Python version.
    echo       Using standard whisper instead ^(slower but works^).
)

echo.
echo ================================
echo Setup complete!
echo.
echo To start the server:
echo   1. venv\Scripts\activate
echo   2. cd webapp
echo   3. python server.py
echo   4. Open http://localhost:8000
echo.
pause
