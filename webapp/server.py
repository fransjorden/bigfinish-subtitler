#!/usr/bin/env python3
"""
Big Finish Caption Sync - Web API Server

FastAPI-based server for the caption sync web application.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our pipeline modules
from pipeline import CaptionSyncPipeline, ProcessingResult
from episode_matcher import match_episode

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
SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", "./parsed_scripts")
SEARCH_INDEX = os.environ.get("SEARCH_INDEX", "./parsed_scripts/search_index.json")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")

# Ensure directories exist
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Job tracking
jobs = {}


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


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Big Finish Caption Sync"}


@app.get("/api/episodes")
async def list_episodes():
    """List all available episodes"""
    try:
        with open(SEARCH_INDEX) as f:
            index = json.load(f)
        
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


async def process_audio_job(job_id: str):
    """Background task to process uploaded audio"""
    job = jobs.get(job_id)
    if not job:
        return
    
    try:
        job['status'] = 'processing'
        job['progress'] = 10
        job['message'] = 'Transcribing audio...'
        
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
        
        job['progress'] = 30
        job['message'] = 'Identifying episode...'
        
        # Process the audio
        result = pipeline.process_audio(
            audio_path=job['file_path'],
            output_dir=str(job_output_dir)
        )
        
        job['progress'] = 90
        job['message'] = 'Finalizing...'
        
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
    with open(json_files[0]) as f:
        data = json.load(f)
    
    return {
        'job_id': job_id,
        'episode': job['result']['episode_title'],
        'part': job['result']['part_number'],
        'captions': data['captions']
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
    
    try:
        job['status'] = 'processing'
        job['progress'] = 10
        job['message'] = 'Transcribing audio...'
        
        job_output_dir = Path(OUTPUT_DIR) / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = CaptionSyncPipeline(
            scripts_dir=SCRIPTS_DIR,
            search_index_path=SEARCH_INDEX,
            openai_api_key=OPENAI_API_KEY,
            use_local_whisper=not bool(OPENAI_API_KEY)
        )
        
        job['progress'] = 50
        job['message'] = 'Aligning with script...'
        
        result = pipeline.process_audio(
            audio_path=job['file_path'],
            output_dir=str(job_output_dir),
            manual_episode=job.get('manual_episode'),
            manual_part=job.get('manual_part')
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
            job['message'] = f'Failed: {result.error}'
            job['result'] = {'error': result.error}
        
        try:
            os.remove(job['file_path'])
        except:
            pass
            
    except Exception as e:
        job['status'] = 'failed'
        job['message'] = f'Error: {str(e)}'
        job['result'] = {'error': str(e)}


# Mount static files for frontend
# app.mount("/", StaticFiles(directory="webapp/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
