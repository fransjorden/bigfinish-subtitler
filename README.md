# Big Finish Doctor Who Caption Sync

An accessibility tool for syncing captions to Big Finish Doctor Who audio dramas.

## Project Status: Phase 2 Complete ✅

### What's Built

**Phase 1: Script Database**
- Parsed 237 OCR-scanned scripts into structured JSON
- 310 individual episodes, 1,093 parts
- Search index for episode auto-detection

**Phase 2: Processing Pipeline**
- Whisper AI transcription integration
- Episode auto-detection from audio
- Transcript-to-script alignment algorithm
- WebVTT caption generation
- Web application with player

## Requirements

- Python 3.8+
- FFmpeg/FFprobe (for audio duration detection)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"  # For Whisper API
export SCRIPTS_DIR="./parsed_scripts"
export SEARCH_INDEX="./parsed_scripts/search_index.json"
```

### 3. Run the Server

```bash
cd webapp
python server.py
```

Then open http://localhost:8000 in your browser.

### Alternative: Local Whisper (No API Key)

If you don't have an OpenAI API key, install local Whisper:

```bash
pip install openai-whisper
```

The system will automatically use local Whisper if no API key is set.

## How It Works

1. **Upload MP3** → User uploads their Big Finish audio file
2. **Transcription** → Whisper AI generates timestamped transcript
3. **Episode Detection** → System searches script database to identify episode
4. **Alignment** → Matches transcript words to official script text
5. **Caption Output** → WebVTT file with correct text and accurate timing

## File Structure

```
bigfinish-captions/
├── parsed_scripts/           # Pre-parsed script database
│   ├── search_index.json     # Search index for matching
│   └── *.json                # Individual script files
├── webapp/
│   ├── server.py             # FastAPI backend
│   └── static/
│       └── index.html        # Frontend application
├── transcription_service.py  # Whisper integration
├── episode_matcher.py        # Episode auto-detection
├── alignment.py              # Transcript alignment
├── pipeline.py               # Main processing pipeline
├── script_parser.py          # Script parsing (for rebuilding DB)
├── build_search_index.py     # Index builder
└── requirements.txt          # Python dependencies
```

## Statistics

| Metric | Value |
|--------|-------|
| Total scripts | 237 original files |
| Parsed entries | 310 episodes |
| Total parts | 1,093 |
| Total dialogue | 6.4 million words |

## Development

### Running Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
cd webapp
python server.py
```

### Rebuilding the Search Index

If you update the parsed scripts, rebuild the search index:

```bash
python build_search_index.py
```

## License

This tool is for personal accessibility use only. It does not distribute any copyrighted audio content.
