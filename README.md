# Big Finish Subtitler

An accessibility tool that generates synchronized captions for Big Finish Doctor Who audio dramas using AI transcription and official script matching.

## Features

- **AI Transcription** - Uses Whisper to transcribe audio with word-level timestamps
- **Auto Episode Detection** - Identifies which episode you're playing from the audio
- **Script Alignment** - Matches transcript to official script text for accurate captions
- **Speaker Labels** - Shows who's speaking (DOCTOR, CHARLEY, etc.)
- **WebVTT Export** - Standard caption format compatible with most players
- **Web Interface** - Simple drag-and-drop UI with built-in caption player

## Quick Start

### Prerequisites

- **Python 3.11+** (3.13 recommended)
- **FFmpeg** - For audio processing ([download](https://ffmpeg.org/download.html))

### Installation

```bash
# Clone the repository
git clone https://github.com/fransjorden/bigfinish-subtitler.git
cd bigfinish-subtitler

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### For Faster Transcription (Recommended)

Install `faster-whisper` for 4x faster transcription with live progress:

```bash
pip install faster-whisper
```

> **Note:** Requires Python 3.11-3.13. If installation fails, standard Whisper will be used automatically.

### Run the Server

```bash
python webapp/server.py
```

Open http://localhost:8000 in your browser.

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

## Script Database

| Metric | Count |
|--------|-------|
| Stories | 287 |
| Main Range | 1-275 |
| Special Releases | 12+ |

Covers the Main Range (releases 1-275), Tenth Doctor Adventures, Companion Chronicles, and more.

## Configuration

### Environment Variables (Optional)

```bash
# Use OpenAI Whisper API instead of local (faster, requires API key)
export OPENAI_API_KEY="sk-..."
```

### Whisper Models

By default, uses the `base` model. For better accuracy (slower), edit `pipeline.py`:

| Model | Speed | Accuracy |
|-------|-------|----------|
| tiny | Fastest | Lower |
| base | Fast | Good |
| small | Medium | Better |
| medium | Slow | High |
| large | Slowest | Highest |

## Troubleshooting

### "No module named 'whisper'"
```bash
pip install openai-whisper
# or for faster processing:
pip install faster-whisper
```

### "FFmpeg not found"
Install FFmpeg and ensure it's in your PATH:
- Windows: Download from ffmpeg.org, add bin folder to PATH
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### Low confidence match
If the episode isn't detected correctly, check that:
- Audio is from the Main Range or supported spin-offs
- Audio quality is reasonable
- First 3 minutes contain recognizable dialogue

## Privacy

- Your audio files are processed locally
- Nothing is uploaded to external servers (unless using OpenAI API)
- Uploaded files are deleted after processing

## License

MIT License - see [LICENSE](LICENSE) for details.

This tool is for personal accessibility use. It does not distribute copyrighted Big Finish content.

## Contributing

Issues and pull requests welcome!
