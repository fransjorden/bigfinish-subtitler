# Big Finish Subtitler

An accessibility tool that generates synchronized captions for Big Finish Doctor Who audio dramas using AI transcription and official script matching.

## Features

- **AI Transcription** - Uses Whisper to transcribe audio with word-level timestamps
- **Auto Episode Detection** - Identifies which episode you're playing from the audio
- **Script Alignment** - Matches transcript to official script text for accurate captions
- **Speaker Labels** - Shows who's speaking (DOCTOR, CHARLEY, etc.)
- **WebVTT Export** - Standard caption format compatible with most players
- **Web Interface** - Simple drag-and-drop UI with built-in caption player
- **GPU Acceleration** - Optional CUDA support for faster processing

---

## Installation

### Option 1: Windows Installer (Recommended)

Download the latest installer - no technical knowledge required:

**[Download BigFinishSubtitler-Setup-1.0.0.exe](https://github.com/fransjorden/bigfinish-subtitler/releases/latest)**

The installer will:
- Auto-detect your GPU and recommend the best processing mode
- Download and install all dependencies automatically
- Create a Start Menu shortcut
- Launch with a single click

> **Requirements:** Windows 10/11 (64-bit), ~1GB disk space, internet connection for first-time setup

### Option 2: Run from Source (Developers)

For developers or users on Mac/Linux:

#### Prerequisites

- **Python 3.11+** (3.13 recommended)
- **FFmpeg** - For audio processing ([download](https://ffmpeg.org/download.html))
- **Git** - For cloning the repository

#### Setup

```bash
# Clone the repository
git clone https://github.com/fransjorden/bigfinish-subtitler.git
cd bigfinish-subtitler

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For faster transcription (recommended, 4x faster):
pip install faster-whisper
```

#### Run the Server

```bash
python webapp/server.py
```

Open http://localhost:8000 in your browser.

---

## How It Works

1. **Upload** - Drop your Big Finish MP3 into the web interface
2. **Transcribe** - Whisper AI generates timestamped transcript
3. **Identify** - System searches 287 stories to find a match
4. **Align** - Matches transcript to official script (preserving proper punctuation)
5. **Output** - WebVTT captions with speaker labels and accurate timing

## Using the Captions

After processing, you can:
- **Play in browser** - Built-in player shows captions as you listen
- **Download VTT** - Use with VLC, media servers, or other players
- **Download JSON** - For custom integrations

### VLC Usage

1. Open your MP3 in VLC
2. Go to Subtitle > Add Subtitle File
3. Select the downloaded .vtt file

---

## Script Database

| Metric | Count |
|--------|-------|
| Stories | 287 |
| Main Range | 1-275 |
| Special Releases | 12+ |

Covers the Main Range (releases 1-275), Tenth Doctor Adventures, Companion Chronicles, and more.

---

## Configuration

### Environment Variables (Optional)

```bash
# Use OpenAI Whisper API instead of local (faster, requires API key)
export OPENAI_API_KEY="sk-..."

# Custom paths (defaults work for most setups)
export SCRIPTS_DIR="./parsed_scripts"
```

### Whisper Models

By default, uses the `small` model. For different accuracy/speed tradeoffs, edit `pipeline.py`:

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| tiny | Fastest | Lower | ~1GB |
| base | Fast | Good | ~1GB |
| small | Medium | Better | ~2GB |
| medium | Slow | High | ~5GB |
| large | Slowest | Highest | ~10GB |

---

## Building the Installer

To build the Windows installer yourself:

1. Install [Inno Setup 6](https://jrsoftware.org/isinfo.php)
2. Run the preparation script:
   ```cmd
   cd installer
   prepare_python.bat
   ```
3. Open `installer/setup.iss` in Inno Setup Compiler
4. Click Compile

See [`installer/BUILD.md`](installer/BUILD.md) for detailed instructions.

---

## Troubleshooting

### "No module named 'whisper'"
```bash
pip install openai-whisper
# or for faster processing:
pip install faster-whisper
```

### "FFmpeg not found"
Install FFmpeg and ensure it's in your PATH:
- **Windows:** Download from ffmpeg.org, add bin folder to PATH
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### Low confidence match
If the episode isn't detected correctly:
- Ensure audio is from a supported release
- Check audio quality is reasonable
- First 3 minutes should contain recognizable dialogue

### GPU not detected (Windows Installer)
- Ensure NVIDIA drivers are installed
- The installer uses `nvidia-smi` to detect GPUs
- CPU mode works on all systems, just slower

---

## Privacy

- Your audio files are processed locally
- Nothing is uploaded to external servers (unless using OpenAI API)
- Uploaded files are deleted after processing

## License

MIT License - see [LICENSE](LICENSE) for details.

This tool is for personal accessibility use. It does not distribute copyrighted Big Finish content.

## Contributing

Issues and pull requests welcome! See the [GitHub issues](https://github.com/fransjorden/bigfinish-subtitler/issues) for planned improvements.
