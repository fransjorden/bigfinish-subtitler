#!/usr/bin/env pythonw
"""
Big Finish Subtitler Launcher

Starts the web server and opens the browser.
Uses .pyw extension to run without console window on Windows.
"""

import os
import sys
import subprocess
import webbrowser
import time
import socket
import json
import threading
from pathlib import Path
import ctypes

# Get the installation directory (where this script lives)
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    INSTALL_DIR = Path(sys.executable).parent
else:
    # Running as script
    INSTALL_DIR = Path(__file__).parent.parent

APP_DIR = INSTALL_DIR
WEBAPP_DIR = APP_DIR / "webapp"
PORT = 8000


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_free_port(start_port=8000):
    """Find a free port starting from start_port."""
    port = start_port
    while port < start_port + 100:
        if not is_port_in_use(port):
            return port
        port += 1
    return start_port  # Fallback


def get_python_exe():
    """Get the Python executable to use."""
    # Check for bundled Python
    bundled = INSTALL_DIR / "python" / "python.exe"
    if bundled.exists():
        return str(bundled)

    # Check for pythonw (no console)
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    if pythonw.exists():
        return str(pythonw)

    return sys.executable


def setup_environment():
    """Set up environment variables."""
    # Add FFmpeg to PATH
    ffmpeg_dir = INSTALL_DIR / "ffmpeg"
    if ffmpeg_dir.exists():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")

    # Add app directory to Python path
    os.environ["PYTHONPATH"] = str(APP_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Set working directory
    os.chdir(str(APP_DIR))


def show_error(message):
    """Show an error message box on Windows."""
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Big Finish Subtitler - Error", 0x10)
    except:
        print(f"Error: {message}", file=sys.stderr)


def show_info(message):
    """Show an info message box on Windows."""
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Big Finish Subtitler", 0x40)
    except:
        print(message)


def wait_for_server(port, timeout=30):
    """Wait for the server to start."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def start_server(port):
    """Start the FastAPI server."""
    python_exe = get_python_exe()
    server_script = WEBAPP_DIR / "server.py"

    if not server_script.exists():
        show_error(f"Server script not found: {server_script}")
        return None

    # Start server as subprocess
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup_info.wShowWindow = subprocess.SW_HIDE

    try:
        process = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "webapp.server:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(APP_DIR),
            startupinfo=startup_info,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return process
    except Exception as e:
        show_error(f"Failed to start server: {e}")
        return None


def main():
    """Main launcher function."""
    setup_environment()

    # Check if server is already running
    if is_port_in_use(PORT):
        # Just open browser to existing server
        webbrowser.open(f"http://localhost:{PORT}")
        return 0

    # Find a free port
    port = find_free_port(PORT)

    # Start the server
    server_process = start_server(port)
    if not server_process:
        return 1

    # Wait for server to start
    if not wait_for_server(port):
        show_error("Server failed to start. Please check the installation.")
        server_process.terminate()
        return 1

    # Open browser
    time.sleep(0.5)  # Small delay for server to be fully ready
    webbrowser.open(f"http://localhost:{port}")

    # Keep running until server exits
    try:
        server_process.wait()
    except KeyboardInterrupt:
        server_process.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
