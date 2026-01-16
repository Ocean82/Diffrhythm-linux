#!/usr/bin/env python3
"""
Simple FastAPI wrapper for DiffRhythm
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import torch
import tempfile
import uuid
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from infer.infer_utils import prepare_model, get_style_prompt, get_negative_style_prompt, get_lrc_token
from infer.infer import inference

app = FastAPI(title="DiffRhythm API", version="1.0.0")

# Global model variables
cfm_model = None
vae_model = None
tokenizer = None
muq_model = None

class GenerationRequest(BaseModel):
    lyrics: str
    style_prompt: str
    audio_length: int = 95
    batch_size: int = 1

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global cfm_model, vae_model, tokenizer, muq_model
    
    print("Loading DiffRhythm models...")
    device = "cpu"
    max_frames = 2048
    
    cfm_model, tokenizer, muq_model, vae_model = prepare_model(max_frames, device)
    print("Models loaded successfully!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": cfm_model is not None}

@app.post("/generate")
async def generate_music(request: GenerationRequest):
    """Generate music from lyrics and style prompt"""
    try:
        if cfm_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Validate audio length
        if request.audio_length == 95:
            max_frames = 2048
        elif 95 < request.audio_length <= 285:
            max_frames = 6144
        else:
            raise HTTPException(status_code=400, detail="Invalid audio_length")
        
        device = "cpu"
        
        # Process inputs
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            max_frames, request.lyrics, tokenizer, request.audio_length, device
        )
        
        style_prompt = get_style_prompt(muq_model, prompt=request.style_prompt)
        negative_style_prompt = get_negative_style_prompt(device)
        
        # Generate latent prompt (empty for new generation)
        latent_prompt = torch.zeros(1, max_frames, 64).to(device)
        pred_frames = [(0, max_frames)]
        
        # Run inference
        generated_songs = inference(
            cfm_model=cfm_model,
            vae_model=vae_model,
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
        
        # Save output
        output_id = str(uuid.uuid4())
        output_path = f"/app/output/{output_id}.wav"
        
        import torchaudio
        generated_song = generated_songs[0]
        torchaudio.save(output_path, generated_song, sample_rate=44100)
        
        return {
            "status": "success",
            "output_id": output_id,
            "output_path": output_path,
            "audio_length": request.audio_length
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DiffRhythm API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)