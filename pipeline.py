#!/usr/bin/env python3
"""
Big Finish Caption Sync - Main Processing Pipeline

This is the main entry point that:
1. Takes an uploaded MP3
2. Transcribes it with Whisper
3. Identifies the episode and part
4. Aligns transcript with official script
5. Generates synchronized captions
"""

import os
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from transcription_service import (
    transcribe_with_whisper_api,
    transcribe_with_local_whisper,
    extract_sample_for_matching,
    save_transcript,
    TranscriptionResult
)
from episode_matcher import match_episode, get_script_part_text, MatchResult
from alignment import (
    align_transcript_to_script,
    align_transcript_to_script_elements,
    export_to_webvtt,
    export_to_json,
    AlignmentResult
)


@dataclass
class ProcessingResult:
    """Result of the complete processing pipeline"""
    success: bool
    episode_title: str
    part_number: int
    release_number: int
    match_confidence: float
    captions_count: int
    webvtt_path: Optional[str]
    json_path: Optional[str]
    error: Optional[str] = None
    warnings: list = None


class CaptionSyncPipeline:
    """Main pipeline for processing audio and generating captions"""
    
    def __init__(self, 
                 scripts_dir: str,
                 search_index_path: str,
                 openai_api_key: Optional[str] = None,
                 use_local_whisper: bool = False,
                 whisper_model: str = "base"):
        """
        Initialize the pipeline.
        
        Args:
            scripts_dir: Path to directory with parsed script JSONs
            search_index_path: Path to search_index.json
            openai_api_key: OpenAI API key for Whisper API (optional)
            use_local_whisper: Use local Whisper instead of API
            whisper_model: Model size for local Whisper
        """
        self.scripts_dir = Path(scripts_dir)
        self.search_index_path = search_index_path
        self.openai_api_key = openai_api_key
        self.use_local_whisper = use_local_whisper
        self.whisper_model = whisper_model
        
        # Validate paths
        if not self.scripts_dir.exists():
            raise ValueError(f"Scripts directory not found: {scripts_dir}")
        if not Path(search_index_path).exists():
            raise ValueError(f"Search index not found: {search_index_path}")
    
    def process_audio(self,
                      audio_path: str,
                      output_dir: str,
                      manual_episode: Optional[str] = None,
                      manual_part: Optional[int] = None,
                      progress_callback: Optional[callable] = None) -> ProcessingResult:
        """
        Process an audio file and generate captions.

        Args:
            audio_path: Path to the MP3 file
            output_dir: Directory to save output files
            manual_episode: Override auto-detection with specific script_id
            manual_part: Override auto-detection with specific part number
            progress_callback: Optional callback(progress: int, message: str) for updates

        Returns:
            ProcessingResult with status and output paths
        """
        def update_progress(progress: int, message: str):
            print(message)
            if progress_callback:
                progress_callback(progress, message)

        warnings = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Transcribe audio
        update_progress(10, "Step 1: Transcribing audio...")

        # Create a callback that scales transcription progress (0-100) to pipeline progress (10-30)
        def transcription_progress(pct: float, msg: str):
            scaled_progress = 10 + (pct / 100) * 20  # 10% to 30%
            update_progress(int(scaled_progress), f"Step 1: {msg}")

        try:
            if self.use_local_whisper:
                transcript = transcribe_with_local_whisper(
                    audio_path,
                    self.whisper_model,
                    progress_callback=transcription_progress
                )
            elif self.openai_api_key:
                transcript = transcribe_with_whisper_api(audio_path, self.openai_api_key)
            else:
                return ProcessingResult(
                    success=False,
                    episode_title="",
                    part_number=0,
                    release_number=0,
                    match_confidence=0,
                    captions_count=0,
                    webvtt_path=None,
                    json_path=None,
                    error="No transcription method available. Provide API key or enable local Whisper."
                )
        except Exception as e:
            return ProcessingResult(
                success=False,
                episode_title="",
                part_number=0,
                release_number=0,
                match_confidence=0,
                captions_count=0,
                webvtt_path=None,
                json_path=None,
                error=f"Transcription failed: {str(e)}"
            )

        update_progress(30, f"Transcribed {transcript.duration:.1f}s of audio ({len(transcript.segments)} segments)")

        # Save transcript for debugging
        transcript_path = output_path / "transcript.json"
        save_transcript(transcript, str(transcript_path))

        # Step 2: Identify episode
        update_progress(40, "Step 2: Identifying episode...")
        
        if manual_episode and manual_part:
            # Use manual override
            match_result = MatchResult(
                script_id=manual_episode,
                title=manual_episode,  # Will be updated below
                part_number=manual_part,
                release_number=0,
                confidence=1.0
            )
            # Try to get actual title
            script_path = self.scripts_dir / f"{manual_episode}.json"
            if script_path.exists():
                with open(script_path, encoding='utf-8') as f:
                    script_data = json.load(f)
                    match_result.title = script_data.get('title', manual_episode)
                    match_result.release_number = script_data.get('release_number', 0)
        else:
            # Auto-detect episode
            sample_text = extract_sample_for_matching(transcript, duration_seconds=180)
            match_result = match_episode(
                sample_text,
                self.search_index_path,
                str(self.scripts_dir)
            )
        
        update_progress(50, f"Matched: {match_result.title} Part {match_result.part_number} ({match_result.confidence:.0%} confidence)")

        if match_result.confidence < 0.3:
            warnings.append(f"Low confidence match ({match_result.confidence:.1%}). Consider manual selection.")

        # Step 3: Get script elements (with speaker info)
        update_progress(60, "Step 3: Loading script...")
        script_path = self.scripts_dir / f"{match_result.script_id}.json"

        if not script_path.exists():
            return ProcessingResult(
                success=False,
                episode_title=match_result.title,
                part_number=match_result.part_number,
                release_number=match_result.release_number,
                match_confidence=match_result.confidence,
                captions_count=0,
                webvtt_path=None,
                json_path=None,
                error=f"Script file not found: {match_result.script_id}"
            )

        with open(script_path, encoding='utf-8') as f:
            script_data = json.load(f)

        # Find the right part and get elements
        script_elements = None
        for part in script_data.get('parts', []):
            if part['part_number'] == match_result.part_number:
                script_elements = part.get('elements', [])
                break

        if not script_elements:
            return ProcessingResult(
                success=False,
                episode_title=match_result.title,
                part_number=match_result.part_number,
                release_number=match_result.release_number,
                match_confidence=match_result.confidence,
                captions_count=0,
                webvtt_path=None,
                json_path=None,
                error=f"Could not load script elements for {match_result.script_id} part {match_result.part_number}"
            )

        dialogue_count = sum(1 for e in script_elements if e.get('type') == 'dialogue')
        update_progress(65, f"Loaded {dialogue_count} dialogue elements")

        # Step 4: Align transcript to script
        update_progress(70, "Step 4: Aligning transcript to script...")
        
        # Convert transcript to dict format for alignment
        transcript_dict = {
            'segments': [
                {
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'words': [
                        {'word': w.word, 'start': w.start, 'end': w.end}
                        for w in seg.words
                    ]
                }
                for seg in transcript.segments
            ]
        }
        
        alignment = align_transcript_to_script_elements(transcript_dict, script_elements)

        update_progress(85, f"Created {len(alignment.captions)} captions ({alignment.script_coverage:.0%} coverage)")

        if alignment.script_coverage < 0.3:
            warnings.append(f"Low script coverage ({alignment.script_coverage:.1%}). Alignment may be poor.")

        # Step 5: Export captions
        update_progress(90, "Step 5: Exporting captions...")
        
        # Generate output filename
        safe_title = "".join(c if c.isalnum() else "_" for c in match_result.title)
        base_name = f"{safe_title}_Part{match_result.part_number}"
        
        webvtt_path = output_path / f"{base_name}.vtt"
        json_path = output_path / f"{base_name}.json"
        
        export_to_webvtt(alignment.captions, str(webvtt_path))
        export_to_json(alignment, str(json_path))

        update_progress(95, f"Saved captions to {base_name}.vtt")
        
        return ProcessingResult(
            success=True,
            episode_title=match_result.title,
            part_number=match_result.part_number,
            release_number=match_result.release_number,
            match_confidence=match_result.confidence,
            captions_count=len(alignment.captions),
            webvtt_path=str(webvtt_path),
            json_path=str(json_path),
            warnings=warnings
        )


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Big Finish Caption Sync - Generate synchronized captions for audio dramas"
    )
    parser.add_argument("audio_file", help="Path to MP3 audio file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--scripts", "-s", required=True, help="Path to parsed scripts directory")
    parser.add_argument("--index", "-i", required=True, help="Path to search_index.json")
    parser.add_argument("--api-key", "-k", help="OpenAI API key for Whisper")
    parser.add_argument("--local-whisper", action="store_true", help="Use local Whisper instead of API")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--episode", "-e", help="Manual episode override (script_id)")
    parser.add_argument("--part", "-p", type=int, help="Manual part number override")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    pipeline = CaptionSyncPipeline(
        scripts_dir=args.scripts,
        search_index_path=args.index,
        openai_api_key=api_key,
        use_local_whisper=args.local_whisper,
        whisper_model=args.whisper_model
    )
    
    result = pipeline.process_audio(
        audio_path=args.audio_file,
        output_dir=args.output,
        manual_episode=args.episode,
        manual_part=args.part
    )
    
    print()
    print("=" * 50)
    if result.success:
        print("SUCCESS!")
        print(f"Episode: {result.episode_title} Part {result.part_number}")
        print(f"Captions: {result.captions_count}")
        print(f"WebVTT: {result.webvtt_path}")
        if result.warnings:
            print()
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
    else:
        print("FAILED")
        print(f"Error: {result.error}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
