#!/usr/bin/env python3
"""
Transcription Service for Big Finish Caption Sync

Uses OpenAI's Whisper API to transcribe audio with word-level timestamps.
"""

import os
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import subprocess


@dataclass
class WordTimestamp:
    """A single word with its timestamp"""
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing"""
    text: str
    start: float
    end: float
    words: list[WordTimestamp]


@dataclass 
class TranscriptionResult:
    """Complete transcription result"""
    segments: list[TranscriptSegment]
    full_text: str
    duration: float
    language: str


def transcribe_with_whisper_api(audio_path: str, api_key: str) -> TranscriptionResult:
    """
    Transcribe audio using OpenAI's Whisper API.
    
    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.)
        api_key: OpenAI API key
        
    Returns:
        TranscriptionResult with word-level timestamps
    """
    import requests
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    with open(audio_path, "rb") as audio_file:
        files = {
            "file": (Path(audio_path).name, audio_file, "audio/mpeg")
        }
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
            "language": "en"
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
    
    # Parse the response
    segments = []
    all_words = []
    
    for segment in result.get("segments", []):
        words = []
        for word_data in segment.get("words", []):
            word = WordTimestamp(
                word=word_data["word"].strip(),
                start=word_data["start"],
                end=word_data["end"]
            )
            words.append(word)
            all_words.append(word)
        
        segments.append(TranscriptSegment(
            text=segment["text"].strip(),
            start=segment["start"],
            end=segment["end"],
            words=words
        ))
    
    return TranscriptionResult(
        segments=segments,
        full_text=result.get("text", ""),
        duration=result.get("duration", 0),
        language=result.get("language", "en")
    )


def transcribe_with_local_whisper(audio_path: str, model: str = "base") -> TranscriptionResult:
    """
    Transcribe audio using local Whisper installation.
    Requires: pip install openai-whisper
    
    Args:
        audio_path: Path to the audio file
        model: Whisper model size (tiny, base, small, medium, large)
        
    Returns:
        TranscriptionResult with word-level timestamps
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Please install whisper: pip install openai-whisper")
    
    # Load model
    model_instance = whisper.load_model(model)
    
    # Transcribe with word timestamps
    result = model_instance.transcribe(
        audio_path,
        language="en",
        word_timestamps=True
    )
    
    segments = []
    
    for segment in result["segments"]:
        words = []
        for word_data in segment.get("words", []):
            words.append(WordTimestamp(
                word=word_data["word"].strip(),
                start=word_data["start"],
                end=word_data["end"]
            ))
        
        segments.append(TranscriptSegment(
            text=segment["text"].strip(),
            start=segment["start"],
            end=segment["end"],
            words=words
        ))
    
    # Calculate duration from last segment
    duration = segments[-1].end if segments else 0
    
    return TranscriptionResult(
        segments=segments,
        full_text=result["text"],
        duration=duration,
        language=result.get("language", "en")
    )


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def extract_sample_for_matching(transcript: TranscriptionResult, 
                                 duration_seconds: float = 180) -> str:
    """
    Extract the first N seconds of transcription for episode matching.
    
    Args:
        transcript: Full transcription result
        duration_seconds: How many seconds to extract (default 3 minutes)
        
    Returns:
        Text from the first duration_seconds of audio
    """
    words = []
    for segment in transcript.segments:
        for word in segment.words:
            if word.end <= duration_seconds:
                words.append(word.word)
            else:
                break
        if segment.end > duration_seconds:
            break
    
    return " ".join(words)


def save_transcript(transcript: TranscriptionResult, output_path: str):
    """Save transcription to JSON file"""
    data = {
        "full_text": transcript.full_text,
        "duration": transcript.duration,
        "language": transcript.language,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in seg.words
                ]
            }
            for seg in transcript.segments
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_transcript(input_path: str) -> TranscriptionResult:
    """Load transcription from JSON file"""
    with open(input_path) as f:
        data = json.load(f)
    
    segments = []
    for seg_data in data["segments"]:
        words = [
            WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
            for w in seg_data["words"]
        ]
        segments.append(TranscriptSegment(
            text=seg_data["text"],
            start=seg_data["start"],
            end=seg_data["end"],
            words=words
        ))
    
    return TranscriptionResult(
        segments=segments,
        full_text=data["full_text"],
        duration=data["duration"],
        language=data["language"]
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python transcription_service.py <audio_file> <output_json> [api_key]")
        print("       If api_key not provided, uses local Whisper")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Transcribing: {audio_file}")
    
    if api_key:
        print("Using OpenAI Whisper API...")
        transcript = transcribe_with_whisper_api(audio_file, api_key)
    else:
        print("Using local Whisper (this may take a while)...")
        transcript = transcribe_with_local_whisper(audio_file)
    
    save_transcript(transcript, output_file)
    print(f"Saved transcript to: {output_file}")
    print(f"Duration: {transcript.duration:.1f}s")
    print(f"Segments: {len(transcript.segments)}")
