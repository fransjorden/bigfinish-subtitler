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


def transcribe_with_local_whisper(
    audio_path: str,
    model: str = "base",
    progress_callback: Optional[callable] = None
) -> TranscriptionResult:
    """
    Transcribe audio using local Whisper installation.
    Tries faster-whisper first (with progress), falls back to openai-whisper.

    Args:
        audio_path: Path to the audio file
        model: Whisper model size (tiny, base, small, medium, large)
        progress_callback: Optional callback(progress: float, message: str) for updates

    Returns:
        TranscriptionResult with word-level timestamps
    """
    # Try faster-whisper first (has progress callbacks and is faster)
    try:
        return _transcribe_with_faster_whisper(audio_path, model, progress_callback)
    except ImportError:
        pass

    # Fall back to openai-whisper
    try:
        import whisper
    except ImportError:
        raise ImportError("Please install whisper: pip install openai-whisper (or faster-whisper for progress support)")

    if progress_callback:
        # Get duration for progress estimation
        duration = get_audio_duration(audio_path)
        if duration > 0:
            progress_callback(0, f"Transcribing {duration/60:.1f} minutes of audio...")

    # Load model
    model_instance = whisper.load_model(model)

    if progress_callback:
        progress_callback(5, "Model loaded, transcribing...")

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

    if progress_callback:
        progress_callback(100, f"Transcribed {len(segments)} segments")

    return TranscriptionResult(
        segments=segments,
        full_text=result["text"],
        duration=duration,
        language=result.get("language", "en")
    )


def _transcribe_with_faster_whisper(
    audio_path: str,
    model: str = "base",
    progress_callback: Optional[callable] = None
) -> TranscriptionResult:
    """
    Transcribe using faster-whisper (CTranslate2-based, much faster with progress).
    """
    from faster_whisper import WhisperModel

    if progress_callback:
        progress_callback(0, "Loading transcription model...")

    # Load model (use CPU, int8 for efficiency)
    model_instance = WhisperModel(model, device="cpu", compute_type="int8")

    # Get audio duration for progress tracking
    audio_duration = get_audio_duration(audio_path)

    if progress_callback:
        progress_callback(5, f"Transcribing {audio_duration/60:.1f} minutes of audio...")

    # Transcribe with word timestamps
    segments_generator, info = model_instance.transcribe(
        audio_path,
        language="en",
        word_timestamps=True
    )

    segments = []
    full_text_parts = []
    last_progress = 5

    for segment in segments_generator:
        words = []
        for word in segment.words or []:
            words.append(WordTimestamp(
                word=word.word.strip(),
                start=word.start,
                end=word.end
            ))

        segments.append(TranscriptSegment(
            text=segment.text.strip(),
            start=segment.start,
            end=segment.end,
            words=words
        ))
        full_text_parts.append(segment.text.strip())

        # Update progress based on position in audio
        if progress_callback and audio_duration > 0:
            # Scale progress from 5% to 95% based on audio position
            progress = 5 + (segment.end / audio_duration) * 90
            # Only update if progress changed significantly (avoid too many updates)
            if progress - last_progress >= 2:
                mins_done = segment.end / 60
                mins_total = audio_duration / 60
                progress_callback(
                    progress,
                    f"Transcribing... {mins_done:.1f}/{mins_total:.1f} min"
                )
                last_progress = progress

    duration = segments[-1].end if segments else info.duration

    if progress_callback:
        progress_callback(100, f"Transcribed {len(segments)} segments")

    return TranscriptionResult(
        segments=segments,
        full_text=" ".join(full_text_parts),
        duration=duration,
        language=info.language or "en"
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
