#!/usr/bin/env python3
"""
Production-ready FastAPI wrapper for DiffRhythm
"""
import os
import sys
import uuid
import logging
import asyncio
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DiffRhythmAPI")

# Add current directory to path
sys.path.append(os.getcwd())

# Lazy imports to avoid startup crash if dependencies fail
try:
    from infer.infer_utils import (
        prepare_model, get_style_prompt, get_negative_style_prompt, get_lrc_token
    )
    from infer.infer import inference
    from infer.infer import save_audio_robust
except ImportError as e:
    logger.error(f"Failed to import DiffRhythm modules: {e}")
    sys.exit(1)

# Configuration from Environment Variables
DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU, can be set to 'cuda'
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_MAX_FRAMES = int(os.getenv("MODEL_MAX_FRAMES", "2048"))

# Use an absolute path for output relative to the current working directory, not hardcoded /app/output
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "output")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="DiffRhythm API", version="1.1.0")

# Mount output directory to serve files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Global state
class ModelManager:
    def __init__(self):
        self.cfm_model = None
        self.vae_model = None
        self.tokenizer = None
        self.muq_model = None
        self.device = DEVICE
        self.is_loaded = False

    def load(self):
        logger.info(f"Loading DiffRhythm models on {self.device}...")
        try:
            self.cfm_model, self.tokenizer, self.muq_model, self.vae_model = prepare_model(
                MODEL_MAX_FRAMES, self.device
            )
            self.is_loaded = True
            logger.info("Models loaded successfully!")
        except Exception as e:
            logger.critical(f"Failed to load models: {e}")
            raise RuntimeError("Model loading failed") from e

models = ModelManager()

class GenerationRequest(BaseModel):
    lyrics: str = Field(..., description="Lyrics for the song")
    style_prompt: str = Field(..., description="Description of the musical style")
    audio_length: int = Field(95, description="Length of audio in seconds (95 or 96-285)", example=95)
    batch_size: int = Field(1, ge=1, le=4, description="Number of generations")

class GenerationResponse(BaseModel):
    status: str
    output_id: str
    output_path: str
    audio_url: str
    audio_length: int

@app.on_event("startup")
async def startup_event():
    # Load models in a separate thread to not block header response if needed, 
    # but for simplicity we load synchronously or could use fastAPI lifespan
    try:
        models.load()
    except Exception:
        logger.error("Application started without models loaded properly.")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if models.is_loaded else "degraded"
    return {
        "status": status, 
        "models_loaded": models.is_loaded,
        "device": models.device,
        "output_dir": OUTPUT_DIR
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest):
    """Generate music from lyrics and style prompt"""
    if not models.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Validate audio length
    max_frames = 2048
    if request.audio_length == 95:
        max_frames = 2048
    elif 95 < request.audio_length <= 285:
        max_frames = 6144
    else:
        raise HTTPException(status_code=400, detail="Invalid audio_length. Must be 95 or between 96-285.")

    output_id = str(uuid.uuid4())
    logger.info(f"Received generation request {output_id} - Style: {request.style_prompt[:50]}...")

    try:
        # Running inference in a threadpool to avoid blocking event loop
        # Note: PyTorch logic is CPU intensive and might block GIL, so separate process is better for high load,
        # but threadpool is okay for simple deployments.
        
        loop = asyncio.get_event_loop()
        generated_filename = await loop.run_in_executor(None, _run_inference, request, max_frames, output_id)
        
        # Construct download URL (assuming client can reach this host)
        # In a real scenario, we might want to return a full URL based on request.base_url
        audio_url = f"/output/{generated_filename}"
        
        return {
            "status": "success",
            "output_id": output_id,
            "output_path": os.path.join(OUTPUT_DIR, generated_filename),
            "audio_url": audio_url,
            "audio_length": request.audio_length
        }

    except ValueError as e:
        logger.error(f"Validation error request {output_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error request {output_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during generation")

def _run_inference(request: GenerationRequest, max_frames: int, output_id: str) -> str:
    """Helper function to run inference implementation"""
    
    # Process inputs
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, request.lyrics, models.tokenizer, request.audio_length, models.device
    )

    style_prompt = get_style_prompt(models.muq_model, prompt=request.style_prompt)
    negative_style_prompt = get_negative_style_prompt(models.device)

    # Generate latent prompt (empty for new generation)
    latent_prompt = torch.zeros(1, max_frames, 64).to(models.device)
    pred_frames = [(0, max_frames)]

    # Run inference
    generated_songs = inference(
        cfm_model=models.cfm_model,
        vae_model=models.vae_model,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=end_frame,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        pred_frames=pred_frames,
        chunked=True,
        batch_infer_num=request.batch_size,
        song_duration=song_duration
    )

    if not generated_songs:
        raise ValueError("No audio generated")

    # Save output
    filename = f"{output_id}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    save_audio_robust(generated_songs[0], output_path, sample_rate=44100)
    
    logger.info(f"Generation {output_id} completed and saved to {output_path}")
    return filename

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)