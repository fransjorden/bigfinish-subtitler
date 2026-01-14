#!/bin/bash
echo "Big Finish Caption Sync - Setup"
echo "================================"
echo

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Try to install faster-whisper (optional)
echo
echo "Installing faster-whisper for better performance..."
pip install faster-whisper 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Note: faster-whisper not available for your Python version."
    echo "      Using standard whisper instead (slower but works)."
fi

echo
echo "================================"
echo "Setup complete!"
echo
echo "To start the server:"
echo "  1. source venv/bin/activate"
echo "  2. cd webapp"
echo "  3. python server.py"
echo "  4. Open http://localhost:8000"
echo
