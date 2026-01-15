# Building the Big Finish Subtitler Windows Installer

This guide explains how to create the Windows installer for Big Finish Subtitler.

## Prerequisites

1. **Inno Setup** - Download and install from https://jrsoftware.org/isinfo.php
2. **Python 3.11+** - For testing scripts
3. **Windows 10/11** - For building

## Directory Structure

```
installer/
├── BUILD.md           # This file
├── setup.iss          # Inno Setup script
├── detect_gpu.py      # GPU detection script
├── install_deps.py    # Dependency installer
├── launcher.pyw       # Application launcher
├── icon.ico           # Application icon (you need to create this)
├── python/            # Embedded Python (see below)
└── output/            # Generated installer will be here
```

## Step 1: Download Embedded Python

Download the Python embeddable package (Windows x64):

1. Go to https://www.python.org/downloads/windows/
2. Download "Windows embeddable package (64-bit)" for Python 3.11 or 3.12
3. Extract to `installer/python/`

After extraction, the folder should contain:
```
installer/python/
├── python.exe
├── pythonw.exe
├── python311.dll (or python312.dll)
├── python311.zip (or python312.zip)
└── ... other files
```

## Step 2: Enable pip in Embedded Python

The embeddable Python doesn't include pip by default. To add it:

```cmd
cd installer\python

# Download get-pip.py
curl -O https://bootstrap.pypa.io/get-pip.py

# Install pip
python.exe get-pip.py

# Edit python311._pth (or python312._pth) and uncomment the import site line:
# Change:
#   #import site
# To:
#   import site
```

## Step 3: Create Application Icon

Create an `icon.ico` file for the application. You can:
- Use an online converter to create .ico from a PNG
- Use a tool like IcoFX or GIMP
- Place the icon at `installer/icon.ico`

If you don't have an icon, create a placeholder or remove the icon references from `setup.iss`.

## Step 4: Create a LICENSE File

Make sure there's a `LICENSE` file in the project root, or remove the `LicenseFile` line from `setup.iss`.

## Step 5: Build the Installer

1. Open Inno Setup Compiler
2. Open `installer/setup.iss`
3. Click "Compile" (or press Ctrl+F9)

The installer will be created at `installer/output/BigFinishSubtitler-Setup-1.0.0.exe`

## What the Installer Does

1. **Copies files** - Application code, scripts, embedded Python
2. **Detects GPU** - Checks for NVIDIA GPU using nvidia-smi
3. **Shows GPU selection page** - Lets user choose GPU or CPU processing
4. **Runs dependency installer** - Downloads and installs:
   - Python packages (fastapi, uvicorn, faster-whisper, etc.)
   - FFmpeg (for audio processing)
   - Whisper model (small, ~500MB)
5. **Creates shortcuts** - Start menu and optionally desktop
6. **Launches app** - Optionally starts the app after install

## Installer Size

- **Base installer**: ~50-60 MB (embedded Python + application code)
- **Downloaded during install**: ~600-800 MB
  - Python packages: ~200 MB
  - FFmpeg: ~100 MB
  - Whisper model: ~500 MB

## Testing the Installer

Before distributing, test on a clean Windows installation or VM:

1. Install on a system without Python
2. Verify GPU detection works (test with and without NVIDIA GPU)
3. Check that the app launches correctly
4. Test processing an audio file

## Troubleshooting

### "Python not found" errors
Make sure the embedded Python is correctly placed in `installer/python/` and pip is enabled.

### GPU not detected
The installer uses `nvidia-smi` to detect NVIDIA GPUs. This requires NVIDIA drivers to be installed.

### Dependencies fail to install
Check internet connection. The installer needs to download packages from PyPI and GitHub.

### App won't start
Check that all paths in `launcher.pyw` are correct and that the server script exists.

## Advanced: Silent Install

For deployment, you can run the installer silently:

```cmd
BigFinishSubtitler-Setup-1.0.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

This will use CPU mode by default. To specify GPU:

```cmd
BigFinishSubtitler-Setup-1.0.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /GPU
```

(Note: The /GPU parameter would need to be added to the setup.iss script)
