#!/usr/bin/env python3
"""
Big Finish Caption Sync - Web API Server

FastAPI-based server for the caption sync web application.
"""

import os
import sys
import json
import tempfile
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import uuid

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our pipeline modules
from pipeline import CaptionSyncPipeline, ProcessingResult
from episode_matcher import match_episode
from script_crypto import load_script, load_search_index

app = FastAPI(
    title="Big Finish Caption Sync",
    description="Accessibility captions for Big Finish Doctor Who audio dramas",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - set these via environment variables
# Default paths are relative to project root (parent of webapp/)
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", str(PROJECT_ROOT / "parsed_scripts"))
SEARCH_INDEX = os.environ.get("SEARCH_INDEX", str(PROJECT_ROOT / "parsed_scripts" / "search_index.json"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))

# Ensure directories exist
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Job tracking
jobs = {}


def get_search_index() -> dict:
    """Load search index, supporting both obfuscated (.enc) and plain (.json) files."""
    index = load_search_index()
    if index:
        return index
    raise FileNotFoundError("Search index not found")


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    result: Optional[dict] = None


class EpisodeInfo(BaseModel):
    script_id: str
    title: str
    part_number: int
    release_number: int


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Big Finish Caption Sync"}


@app.get("/api/episodes")
async def list_episodes():
    """List all available episodes"""
    try:
        index = get_search_index()

        episodes = []
        seen = set()

        for script_id, script_info in index['scripts'].items():
            if script_id not in seen:
                seen.add(script_id)
                episodes.append({
                    'script_id': script_id,
                    'title': script_info['title'],
                    'release_number': script_info['release_number'],
                    'parts': script_info['num_parts']
                })

        # Sort by release number
        episodes.sort(key=lambda x: x['release_number'])

        return {"episodes": episodes, "total": len(episodes)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories")
async def list_stories():
    """List all supported story titles for display on homepage"""
    try:
        index = get_search_index()

        # Deduplicate by release number, picking the best title
        by_release = {}
        other_releases = []  # Non-main-range stories

        # Titles to skip (generic/wrong titles from OCR)
        bad_titles = {
            'doctor who', 'docior who', 'doctdr who',
            'episode one', 'episode two', 'episode three', 'episode four',
            'a four-part story', 'a two-part story',
            'dramatis personae', 'purity', 'the eternal',
            'bi bic', 'donc iloik', 'by steve lyons',
            'the doctor', 'the doctor peter davison',
            'and other stories', 'iterstitial'
        }

        # Manual title fixes for known product codes
        title_fixes = {
            '10da1x1': 'Technophobia (10th Doctor)',
            '10da1x2': 'Time Reaver (10th Doctor)',
            '10da1x3': 'Death and the Queen (10th Doctor)',
            'cc07x09': 'The Scorchies (Companion Chronicles)',
            'cc10x01': 'The Mouthless Dead (Companion Chronicles)',
            'cc10x02': 'The Story of Extinction (Companion Chronicles)',
            'cc10x03': 'The Integral (Companion Chronicles)',
            'cc10x04': 'The Edge (Companion Chronicles)',
            'fp1x1': 'The Eleven-Day Empire (Faction Paradox)',
            'fp1x2': 'The Shadow Play (Faction Paradox)',
            'fp1x3': 'Sabbath Dei (Faction Paradox)',
            'fp1x4': 'In the Year of the Cat (Faction Paradox)',
            'fp1x5': 'Movers (Faction Paradox)',
            'fp1x6': 'A Labyrinth of Histories (Faction Paradox)',
            'bs03x2': 'The Green Eyed Monsters (Bernice Summerfield)',
            'dw090a': 'Return of the Daleks',
            'dw142a': 'The Four Doctors',
            'dw155a': 'The Five Companions',
            'dw168a': 'Night of the Stormcrow',
            'dw181a': 'Trial of the Valeyard',
            'sp 1.05': 'The Light at the End (50th Anniversary)',
            'dalek empire 1&2': 'Dalek Empire (Series 1 & 2)',
        }

        for script_id, script_info in index['scripts'].items():
            release = script_info['release_number']
            title = script_info['title']
            title_lower = title.lower().strip()

            # Check for title fix
            if title_lower in title_fixes:
                title = title_fixes[title_lower]
                title_lower = title.lower()

            # Handle release 0 (special/non-main-range content)
            if release == 0:
                if title_lower not in bad_titles and title_lower not in [t.lower() for t in [o['title'] for o in other_releases]]:
                    other_releases.append({'title': title, 'release_number': 0})
                continue

            # Skip bad titles
            if title_lower in bad_titles:
                continue

            # If we already have this release, keep the better title
            if release in by_release:
                existing = by_release[release]
                # Prefer longer titles (more descriptive) but not ALL CAPS
                existing_score = len(existing) if not existing.isupper() else len(existing) * 0.5
                new_score = len(title) if not title.isupper() else len(title) * 0.5
                if new_score > existing_score:
                    by_release[release] = title
            else:
                by_release[release] = title

        stories = [
            {'title': title, 'release_number': rel}
            for rel, title in by_release.items()
        ]

        # Sort main range by release number
        stories.sort(key=lambda x: x['release_number'])

        # Add other releases at the end
        stories.extend(other_releases)

        return {"stories": stories, "total": len(stories)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload an MP3 file for processing.
    Returns a job ID for tracking progress.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an MP3, WAV, M4A, or OGG file."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_path = Path(UPLOAD_DIR) / f"{job_id}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Initialize job status
    jobs[job_id] = {
        'status': 'pending',
        'progress': 0,
        'message': 'Queued for processing',
        'file_path': str(upload_path),
        'created_at': datetime.now().isoformat(),
        'result': None
    }
    
    # Start processing in background
    background_tasks.add_task(process_audio_job, job_id)

    return {"job_id": job_id, "status": "pending"}


@app.post("/api/upload-multi")
async def upload_multiple_audio(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload multiple MP3 files (tracks) to be merged and processed.
    Returns a job ID for tracking progress.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate all files
    for file in files:
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Please upload MP3, WAV, M4A, or OGG files."
            )

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save all uploaded files
    saved_files = []
    try:
        for i, file in enumerate(files):
            # Use index prefix to preserve order
            file_path = job_dir / f"{i:03d}_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(file_path))
    except Exception as e:
        # Clean up on error
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")

    # Initialize job status
    jobs[job_id] = {
        'status': 'pending',
        'progress': 0,
        'message': 'Queued for processing',
        'file_paths': saved_files,
        'merged_path': None,
        'created_at': datetime.now().isoformat(),
        'result': None
    }

    # Start processing in background
    background_tasks.add_task(process_multi_audio_job, job_id)

    return {"job_id": job_id, "status": "pending", "files_count": len(saved_files)}


def merge_audio_files(file_paths: List[str], output_path: str) -> bool:
    """
    Merge multiple audio files using ffmpeg concat demuxer.
    Returns True if successful.
    """
    # Create a file list for ffmpeg
    list_path = output_path + ".txt"
    try:
        with open(list_path, "w", encoding='utf-8') as f:
            for path in file_paths:
                # Escape single quotes in path
                escaped_path = path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        # Run ffmpeg to concatenate
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", output_path
            ],
            capture_output=True,
            text=True
        )

        return result.returncode == 0
    finally:
        # Clean up list file
        if os.path.exists(list_path):
            os.remove(list_path)


async def process_multi_audio_job(job_id: str):
    """Background task to merge and process multiple audio files"""
    job = jobs.get(job_id)
    if not job:
        return

    # Run in thread pool to avoid blocking
    await asyncio.to_thread(run_multi_pipeline_sync, job_id, job)


def run_multi_pipeline_sync(job_id: str, job: dict):
    """
    Merge audio files and run the pipeline.
    """
    def progress_callback(progress: int, message: str):
        job['progress'] = progress
        job['message'] = message

    try:
        job['status'] = 'processing'
        job['progress'] = 2
        job['message'] = 'Merging audio tracks...'

        # Merge the audio files
        job_dir = Path(UPLOAD_DIR) / job_id
        merged_path = job_dir / "merged.mp3"

        if not merge_audio_files(job['file_paths'], str(merged_path)):
            job['status'] = 'failed'
            job['message'] = 'Failed to merge audio files. Is ffmpeg installed?'
            job['result'] = {'error': 'Failed to merge audio files'}
            return

        job['merged_path'] = str(merged_path)
        job['progress'] = 5
        job['message'] = 'Tracks merged, starting processing...'

        # Create output directory
        job_output_dir = Path(OUTPUT_DIR) / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy merged file to output for playback
        merged_output = job_output_dir / "merged.mp3"
        shutil.copy(merged_path, merged_output)

        # Initialize pipeline
        pipeline = CaptionSyncPipeline(
            scripts_dir=SCRIPTS_DIR,
            search_index_path=SEARCH_INDEX,
            openai_api_key=OPENAI_API_KEY,
            use_local_whisper=not bool(OPENAI_API_KEY)
        )

        # Process the merged audio
        result = pipeline.process_audio(
            audio_path=str(merged_path),
            output_dir=str(job_output_dir),
            progress_callback=progress_callback
        )

        if result.success:
            job['status'] = 'completed'
            job['progress'] = 100
            job['message'] = 'Processing complete!'
            job['result'] = {
                'episode_title': result.episode_title,
                'part_number': result.part_number,
                'release_number': result.release_number,
                'match_confidence': result.match_confidence,
                'captions_count': result.captions_count,
                'webvtt_url': f"/api/output/{job_id}/{Path(result.webvtt_path).name}",
                'json_url': f"/api/output/{job_id}/{Path(result.json_path).name}",
                'merged_audio_url': f"/api/output/{job_id}/merged.mp3",
                'warnings': result.warnings
            }
        else:
            job['status'] = 'failed'
            job['progress'] = 100
            job['message'] = f'Processing failed: {result.error}'
            job['result'] = {'error': result.error}

        # Clean up uploaded files (keep merged in output)
        try:
            job_dir = Path(UPLOAD_DIR) / job_id
            shutil.rmtree(job_dir, ignore_errors=True)
        except:
            pass

    except Exception as e:
        job['status'] = 'failed'
        job['progress'] = 100
        job['message'] = f'Error: {str(e)}'
        job['result'] = {'error': str(e)}


def run_pipeline_sync(job_id: str, job: dict, manual_episode: str = None, manual_part: int = None):
    """
    Run the pipeline synchronously (called from thread pool).
    This is blocking code that shouldn't run on the event loop.
    """
    def progress_callback(progress: int, message: str):
        """Update job progress from pipeline"""
        job['progress'] = progress
        job['message'] = message

    try:
        job['status'] = 'processing'
        job['progress'] = 5
        job['message'] = 'Starting processing...'

        # Create output directory for this job
        job_output_dir = Path(OUTPUT_DIR) / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline
        pipeline = CaptionSyncPipeline(
            scripts_dir=SCRIPTS_DIR,
            search_index_path=SEARCH_INDEX,
            openai_api_key=OPENAI_API_KEY,
            use_local_whisper=not bool(OPENAI_API_KEY)
        )

        # Process the audio with progress callback
        result = pipeline.process_audio(
            audio_path=job['file_path'],
            output_dir=str(job_output_dir),
            manual_episode=manual_episode,
            manual_part=manual_part,
            progress_callback=progress_callback
        )

        if result.success:
            job['status'] = 'completed'
            job['progress'] = 100
            job['message'] = 'Processing complete!'
            job['result'] = {
                'episode_title': result.episode_title,
                'part_number': result.part_number,
                'release_number': result.release_number,
                'match_confidence': result.match_confidence,
                'captions_count': result.captions_count,
                'webvtt_url': f"/api/output/{job_id}/{Path(result.webvtt_path).name}",
                'json_url': f"/api/output/{job_id}/{Path(result.json_path).name}",
                'warnings': result.warnings
            }
        else:
            job['status'] = 'failed'
            job['progress'] = 100
            job['message'] = f'Processing failed: {result.error}'
            job['result'] = {'error': result.error}

        # Clean up uploaded file
        try:
            os.remove(job['file_path'])
        except:
            pass

    except Exception as e:
        job['status'] = 'failed'
        job['progress'] = 100
        job['message'] = f'Error: {str(e)}'
        job['result'] = {'error': str(e)}


async def process_audio_job(job_id: str):
    """Background task to process uploaded audio"""
    job = jobs.get(job_id)
    if not job:
        return

    # Run the blocking pipeline work in a thread pool
    await asyncio.to_thread(run_pipeline_sync, job_id, job)


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'result': job.get('result')
    }


@app.get("/api/output/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str):
    """Download an output file (VTT or JSON)"""
    file_path = Path(OUTPUT_DIR) / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = "text/vtt" if filename.endswith(".vtt") else "application/json"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/api/captions/{job_id}")
async def get_captions(job_id: str):
    """Get captions data for the player"""
    job = jobs.get(job_id)
    if not job or job['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Captions not ready")
    
    # Find the JSON file
    job_output_dir = Path(OUTPUT_DIR) / job_id
    json_files = list(job_output_dir.glob("*.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Caption file not found")
    
    # Load and return caption data
    with open(json_files[0], encoding='utf-8') as f:
        data = json.load(f)
    
    return {
        'job_id': job_id,
        'episode': job['result']['episode_title'],
        'part': job['result']['part_number'],
        'captions': data['captions'],
        'gaps': data.get('gaps', [])
    }


@app.post("/api/manual-select")
async def manual_episode_select(
    file: UploadFile = File(...),
    script_id: str = None,
    part_number: int = None,
    background_tasks: BackgroundTasks = None
):
    """
    Upload with manual episode selection (bypass auto-detection).
    """
    if not script_id or not part_number:
        raise HTTPException(
            status_code=400,
            detail="script_id and part_number are required for manual selection"
        )
    
    # Similar to upload, but with manual override
    job_id = str(uuid.uuid4())[:8]
    upload_path = Path(UPLOAD_DIR) / f"{job_id}_{file.filename}"
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    jobs[job_id] = {
        'status': 'pending',
        'progress': 0,
        'message': 'Queued for processing',
        'file_path': str(upload_path),
        'manual_episode': script_id,
        'manual_part': part_number,
        'created_at': datetime.now().isoformat(),
        'result': None
    }
    
    background_tasks.add_task(process_manual_job, job_id)
    
    return {"job_id": job_id, "status": "pending"}


async def process_manual_job(job_id: str):
    """Process with manual episode selection"""
    job = jobs.get(job_id)
    if not job:
        return

    # Run the blocking pipeline work in a thread pool
    await asyncio.to_thread(
        run_pipeline_sync,
        job_id,
        job,
        job.get('manual_episode'),
        job.get('manual_part')
    )


# Serve index.html with no-cache headers for development
@app.get("/")
async def serve_index():
    """Serve the main page with no-cache headers"""
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(
        index_path,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# Mount static files for other assets
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
