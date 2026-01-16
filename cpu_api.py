"""
CPU-friendly async music generation API
Handles long inference times with job queue
"""
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import json
from pathlib import Path
from datetime import datetime

app = FastAPI()

# Simple in-memory job store (use Redis/DB in production)
jobs = {}

class GenerateRequest(BaseModel):
    lyrics: str
    style_prompt: str
    duration: int = 95

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: str
    completed_at: str | None = None
    output_url: str | None = None
    error: str | None = None

def run_inference(job_id: str, lyrics: str, style_prompt: str, duration: int):
    """Background task for inference"""
    import subprocess
    import tempfile
    
    jobs[job_id]["status"] = "processing"
    
    try:
        # Write lyrics to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lrc', delete=False) as f:
            f.write(lyrics)
            lrc_path = f.name
        
        output_dir = f"outputs/{job_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run inference
        result = subprocess.run([
            "python3", "infer/infer.py",
            "--lrc-path", lrc_path,
            "--ref-prompt", style_prompt,
            "--audio-length", str(duration),
            "--output-dir", output_dir,
            "--chunked"
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_url"] = f"/download/{job_id}/output.wav"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = result.stderr
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Submit generation job"""
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "output_url": None,
        "error": None
    }
    
    background_tasks.add_task(
        run_inference, job_id, req.lyrics, req.style_prompt, req.duration
    )
    
    return {"job_id": job_id, "estimated_time_minutes": 20}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check job status"""
    if job_id not in jobs:
        return {"error": "Job not found"}, 404
    return jobs[job_id]

@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    """Download generated audio"""
    from fastapi.responses import FileResponse
    file_path = f"outputs/{job_id}/{filename}"
    if Path(file_path).exists():
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}, 404

# Usage:
# uvicorn cpu_api:app --host 0.0.0.0 --port 8000
# 
# POST /generate {"lyrics": "...", "style_prompt": "folk"}
# GET /status/{job_id}
# GET /download/{job_id}/output.wav
