#!/usr/bin/env python3
"""
GPU Detection for Big Finish Subtitler Installer

Detects NVIDIA GPUs and CUDA availability for faster-whisper acceleration.
"""

import subprocess
import sys
import json


def detect_nvidia_gpu():
    """
    Detect if an NVIDIA GPU is present and get its info.
    Returns dict with gpu info or None if no NVIDIA GPU.
    """
    # Try nvidia-smi first (most reliable)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return {
                    "name": parts[0].strip(),
                    "memory_mb": int(float(parts[1].strip())),
                    "driver_version": parts[2].strip(),
                    "cuda_available": True
                }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Try WMI as fallback (Windows)
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            for line in lines[1:]:  # Skip header
                parts = line.split(",")
                if len(parts) >= 3:
                    name = parts[2] if len(parts) > 2 else parts[1]
                    if "NVIDIA" in name.upper():
                        try:
                            ram = int(parts[1]) // (1024 * 1024) if parts[1].isdigit() else 0
                        except:
                            ram = 0
                        return {
                            "name": name,
                            "memory_mb": ram,
                            "driver_version": "unknown",
                            "cuda_available": False  # Can't confirm CUDA without nvidia-smi
                        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return None


def check_cuda_available():
    """Check if CUDA is properly installed and accessible."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def get_recommendation(gpu_info):
    """Get installation recommendation based on GPU detection."""
    if gpu_info is None:
        return {
            "use_gpu": False,
            "reason": "No NVIDIA GPU detected. Will use CPU (slower but works on all systems).",
            "estimate": "Processing will take approximately 1-2x the audio length."
        }

    # Check memory - need at least 4GB for reasonable GPU performance
    mem_mb = gpu_info.get("memory_mb", 0)

    if mem_mb >= 4000:
        return {
            "use_gpu": True,
            "reason": f"Found {gpu_info['name']} with {mem_mb}MB VRAM. GPU acceleration recommended!",
            "estimate": "Processing will be 4-10x faster than CPU."
        }
    elif mem_mb >= 2000:
        return {
            "use_gpu": True,
            "reason": f"Found {gpu_info['name']} with {mem_mb}MB VRAM. GPU acceleration available.",
            "estimate": "Processing will be 2-4x faster than CPU. May need 'small' model for memory."
        }
    else:
        return {
            "use_gpu": False,
            "reason": f"Found {gpu_info['name']} but limited VRAM ({mem_mb}MB). CPU recommended.",
            "estimate": "GPU memory too low for reliable acceleration."
        }


def main():
    """Main detection routine - outputs JSON."""
    gpu_info = detect_nvidia_gpu()
    recommendation = get_recommendation(gpu_info)

    result = {
        "gpu_detected": gpu_info is not None,
        "gpu_info": gpu_info,
        "recommendation": recommendation
    }

    # Output as JSON for the installer to parse
    print(json.dumps(result, indent=2))

    # Return exit code: 0 = GPU available, 1 = CPU only
    return 0 if recommendation["use_gpu"] else 1


if __name__ == "__main__":
    sys.exit(main())
