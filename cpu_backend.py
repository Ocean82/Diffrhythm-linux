"""
Production CPU-only music generation backend
Handles 20+ minute generation times with proper queue management
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import tempfile
import uuid
import json
from pathlib import Path
from datetime import datetime
import threading
import queue

app = FastAPI()

# Job storage
jobs_db = {}
job_queue = queue.Queue()
processing_lock = threading.Lock()
current_job = None

class GenerateRequest(BaseModel):
    lyrics: str
    style_prompt: str
    duration: int = 95

def worker():
    """Single worker thread - processes one job at a time"""
    global current_job
    
    while True:
        job_id = job_queue.get()
        
        with processing_lock:
            current_job = job_id
            jobs_db[job_id]["status"] = "processing"
            jobs_db[job_id]["started_at"] = datetime.now().isoformat()
        
        try:
            job = jobs_db[job_id]
            
            # Write lyrics
            lrc_file = Path(f"temp/{job_id}.lrc")
            lrc_file.parent.mkdir(exist_ok=True)
            lrc_file.write_text(job["lyrics"])
            
            # Output directory
            output_dir = Path(f"outputs/{job_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run inference
            result = subprocess.run([
                "python3", "infer/infer.py",
                "--lrc-path", str(lrc_file),
                "--ref-prompt", job["style_prompt"],
                "--audio-length", str(job["duration"]),
                "--output-dir", str(output_dir),
                "--chunked",
                "--batch-infer-num", "1"
            ], capture_output=True, text=True, timeout=2400)  # 40 min timeout
            
            if result.returncode == 0:
                jobs_db[job_id]["status"] = "completed"
                jobs_db[job_id]["output_file"] = f"{job_id}/output.wav"
            else:
                jobs_db[job_id]["status"] = "failed"
                jobs_db[job_id]["error"] = result.stderr[-500:]  # Last 500 chars
                
        except subprocess.TimeoutExpired:
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["error"] = "Generation timeout (40 minutes)"
        except Exception as e:
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["error"] = str(e)
        finally:
            jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
            with processing_lock:
                current_job = None
            job_queue.task_done()

# Start worker thread
threading.Thread(target=worker, daemon=True).start()

@app.post("/generate")
def generate(req: GenerateRequest):
    """Submit generation job"""
    job_id = str(uuid.uuid4())
    
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "lyrics": req.lyrics,
        "style_prompt": req.style_prompt,
        "duration": req.duration,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "output_file": None,
        "error": None
    }
    
    job_queue.put(job_id)
    queue_position = job_queue.qsize()
    
    return {
        "job_id": job_id,
        "queue_position": queue_position,
        "estimated_wait_minutes": queue_position * 20,
        "message": "Job queued. Check /status/{job_id} for progress"
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Check job status"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id].copy()
    job.pop("lyrics", None)  # Don't return full lyrics
    
    # Add queue info
    if job["status"] == "queued":
        queue_list = list(job_queue.queue)
        if job_id in queue_list:
            job["queue_position"] = queue_list.index(job_id) + 1
            job["estimated_wait_minutes"] = job["queue_position"] * 20
    
    return job

@app.get("/download/{job_id}")
def download(job_id: str):
    """Download generated audio"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    file_path = Path(f"outputs/{job['output_file']}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(file_path, media_type="audio/wav", filename="generated_song.wav")

@app.get("/queue")
def queue_status():
    """View queue status"""
    return {
        "queue_length": job_queue.qsize(),
        "current_job": current_job,
        "estimated_wait_minutes": job_queue.qsize() * 20
    }

@app.get("/")
def root():
    return {
        "service": "DiffRhythm CPU Backend",
        "note": "Generation takes ~20 minutes per song",
        "endpoints": {
            "POST /generate": "Submit job",
            "GET /status/{job_id}": "Check status",
            "GET /download/{job_id}": "Download audio",
            "GET /queue": "View queue"
        }
    }

# Run: uvicorn cpu_backend:app --host 0.0.0.0 --port 8000
