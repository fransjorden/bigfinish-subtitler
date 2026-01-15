#!/usr/bin/env python3
"""
Dependency Installer for Big Finish Subtitler

Downloads and installs all required dependencies:
- Python packages (fastapi, uvicorn, faster-whisper, etc.)
- FFmpeg (for audio processing)

Run with: python install_deps.py [--gpu|--cpu] [--install-dir PATH]
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import json
from pathlib import Path
import ssl
import tempfile


# URLs for dependencies
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

# Python packages
BASE_PACKAGES = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "python-multipart>=0.0.6",
    "requests>=2.31.0",
]

# faster-whisper packages
WHISPER_CPU_PACKAGES = [
    "faster-whisper>=1.0.0",
]

WHISPER_GPU_PACKAGES = [
    "faster-whisper>=1.0.0",
    # CUDA packages will be handled by faster-whisper's dependencies
]


def log(message, progress=None):
    """Output progress message in a format the installer can parse."""
    output = {"message": message}
    if progress is not None:
        output["progress"] = progress
    print(json.dumps(output), flush=True)


def download_file(url, dest_path, description="file"):
    """Download a file with progress reporting."""
    log(f"Downloading {description}...")

    # Create SSL context that doesn't verify (for corporate proxies)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(url, context=ctx) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = int((downloaded / total_size) * 100)
                        if downloaded % (chunk_size * 100) == 0:  # Don't spam
                            log(f"Downloading {description}... {pct}%")

        log(f"Downloaded {description}")
        return True
    except Exception as e:
        log(f"Error downloading {description}: {e}")
        return False


def install_ffmpeg(install_dir):
    """Download and install FFmpeg."""
    log("Installing FFmpeg...", progress=10)

    ffmpeg_dir = Path(install_dir) / "ffmpeg"
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"

    # Check if already installed
    if ffmpeg_exe.exists():
        log("FFmpeg already installed")
        return True

    # Download FFmpeg
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "ffmpeg.zip"

        if not download_file(FFMPEG_URL, zip_path, "FFmpeg"):
            return False

        log("Extracting FFmpeg...", progress=20)

        # Extract
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)

            # Find the bin directory
            extracted_dirs = [d for d in Path(tmp_dir).iterdir() if d.is_dir() and "ffmpeg" in d.name.lower()]
            if extracted_dirs:
                bin_dir = extracted_dirs[0] / "bin"
                if bin_dir.exists():
                    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
                    for exe in ["ffmpeg.exe", "ffprobe.exe"]:
                        src = bin_dir / exe
                        if src.exists():
                            shutil.copy2(src, ffmpeg_dir / exe)

            log("FFmpeg installed successfully", progress=30)
            return True
        except Exception as e:
            log(f"Error extracting FFmpeg: {e}")
            return False


def install_python_packages(install_dir, use_gpu=False):
    """Install Python packages using pip."""
    log("Installing Python packages...", progress=40)

    python_exe = Path(install_dir) / "python" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    # Combine packages
    packages = BASE_PACKAGES.copy()
    if use_gpu:
        packages.extend(WHISPER_GPU_PACKAGES)
        log("Installing GPU-accelerated packages...")
    else:
        packages.extend(WHISPER_CPU_PACKAGES)
        log("Installing CPU packages...")

    # Install each package
    for i, package in enumerate(packages):
        progress = 40 + int((i / len(packages)) * 40)
        log(f"Installing {package.split('>=')[0]}...", progress=progress)

        try:
            result = subprocess.run(
                [str(python_exe), "-m", "pip", "install", "--quiet", package],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per package
            )
            if result.returncode != 0:
                log(f"Warning: Error installing {package}: {result.stderr}")
        except subprocess.TimeoutExpired:
            log(f"Warning: Timeout installing {package}")
        except Exception as e:
            log(f"Warning: Error installing {package}: {e}")

    log("Python packages installed", progress=80)
    return True


def download_whisper_model(install_dir, model="small"):
    """Pre-download Whisper model."""
    log(f"Downloading Whisper '{model}' model (this may take a few minutes)...", progress=85)

    python_exe = Path(install_dir) / "python" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    # Use faster-whisper to download the model
    download_script = f'''
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
try:
    from faster_whisper import WhisperModel
    print("Downloading model...")
    model = WhisperModel("{model}", device="cpu", compute_type="int8")
    print("Model downloaded successfully")
except Exception as e:
    print(f"Note: Model will be downloaded on first use: {{e}}")
'''

    try:
        result = subprocess.run(
            [str(python_exe), "-c", download_script],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if "successfully" in result.stdout:
            log("Whisper model downloaded", progress=95)
        else:
            log("Model will be downloaded on first use", progress=95)
    except Exception as e:
        log(f"Model will be downloaded on first use: {e}", progress=95)

    return True


def create_config(install_dir, use_gpu=False):
    """Create configuration file."""
    config = {
        "use_gpu": use_gpu,
        "whisper_model": "small",
        "device": "cuda" if use_gpu else "cpu",
        "compute_type": "float16" if use_gpu else "int8"
    }

    config_path = Path(install_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    log("Configuration saved")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Install Big Finish Subtitler dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU (CUDA) support")
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only support")
    parser.add_argument("--install-dir", type=str, default=".", help="Installation directory")
    parser.add_argument("--skip-model", action="store_true", help="Skip downloading Whisper model")

    args = parser.parse_args()

    use_gpu = args.gpu and not args.cpu
    install_dir = Path(args.install_dir).resolve()

    log(f"Installing to: {install_dir}", progress=0)
    log(f"Mode: {'GPU (CUDA)' if use_gpu else 'CPU'}", progress=2)

    # Step 1: FFmpeg
    if not install_ffmpeg(install_dir):
        log("Warning: FFmpeg installation failed. Some features may not work.")

    # Step 2: Python packages
    if not install_python_packages(install_dir, use_gpu):
        log("Error: Failed to install Python packages")
        return 1

    # Step 3: Whisper model (optional)
    if not args.skip_model:
        download_whisper_model(install_dir)

    # Step 4: Config
    create_config(install_dir, use_gpu)

    log("Installation complete!", progress=100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
