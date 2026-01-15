@echo off
REM Prepare Embedded Python for Big Finish Subtitler Installer
REM Run this script to download and set up the embedded Python

setlocal enabledelayedexpansion

set PYTHON_VERSION=3.11.9
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set GETPIP_URL=https://bootstrap.pypa.io/get-pip.py

echo.
echo ==========================================
echo  Big Finish Subtitler - Prepare Python
echo ==========================================
echo.

REM Create python directory
if not exist "python" mkdir python

REM Check if already downloaded
if exist "python\python.exe" (
    echo Python already downloaded. Checking pip...
    goto :check_pip
)

REM Download Python embeddable
echo Downloading Python %PYTHON_VERSION% embeddable...
curl -L -o python.zip "%PYTHON_URL%"
if errorlevel 1 (
    echo Error: Failed to download Python. Please download manually from:
    echo %PYTHON_URL%
    pause
    exit /b 1
)

REM Extract Python
echo Extracting Python...
powershell -command "Expand-Archive -Path 'python.zip' -DestinationPath 'python' -Force"
del python.zip

:check_pip
REM Check if pip is installed
python\python.exe -m pip --version >nul 2>&1
if not errorlevel 1 (
    echo Pip already installed.
    goto :enable_site
)

REM Download and install pip
echo Downloading pip...
curl -L -o python\get-pip.py "%GETPIP_URL%"
if errorlevel 1 (
    echo Error: Failed to download pip installer.
    pause
    exit /b 1
)

echo Installing pip...
python\python.exe python\get-pip.py
del python\get-pip.py

:enable_site
REM Enable site-packages by editing the ._pth file
echo Enabling site-packages...

REM Find the ._pth file
for %%f in (python\python*._pth) do (
    set PTH_FILE=%%f
)

if defined PTH_FILE (
    REM Check if import site is already uncommented
    findstr /C:"import site" "!PTH_FILE!" | findstr /V /C:"#import" >nul 2>&1
    if errorlevel 1 (
        REM Need to uncomment import site
        echo Updating !PTH_FILE!...
        powershell -command "(Get-Content '!PTH_FILE!') -replace '#import site', 'import site' | Set-Content '!PTH_FILE!'"
    ) else (
        echo site-packages already enabled.
    )
) else (
    echo Warning: Could not find ._pth file
)

echo.
echo ==========================================
echo  Python preparation complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Create an icon.ico file (or remove icon references from setup.iss)
echo 2. Open setup.iss in Inno Setup Compiler
echo 3. Click Compile to build the installer
echo.

pause
