"""
Unified Production API for DiffRhythm
Combines best features from all API implementations with production-ready features
"""
import os
import sys
import uuid
import asyncio
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
import uvicorn
import hmac
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend modules
from backend.config import Config
from backend.exceptions import (
    DiffRhythmException,
    ModelNotLoadedError,
    InvalidRequestError,
    GenerationError,
    JobNotFoundError
)
from backend.logging_config import setup_logging, get_logger, RequestLogger
from backend.metrics import metrics
from backend.security import (
    limiter,
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
    get_api_key,
    check_api_key,
    get_cors_config,
    get_security_headers
)
from backend.payment_verification import verify_payment_intent, check_payment_required, STRIPE_AVAILABLE

# Import DiffRhythm modules
try:
    from infer.infer_utils import (
        prepare_model,
        get_style_prompt,
        get_negative_style_prompt,
        get_lrc_token,
        get_reference_latent
    )
    from infer.infer import inference, save_audio_robust
except ImportError as e:
    print(f"ERROR: Failed to import DiffRhythm modules: {e}")
    sys.exit(1)

# Setup logging
setup_logging()
logger = get_logger("DiffRhythmAPI")
request_logger = RequestLogger(logger)

# Ensure directories exist
Config.ensure_directories()

# Validate configuration
config_errors = Config.validate()
if config_errors:
    logger.error(f"Configuration errors: {config_errors}")
    raise ValueError(f"Invalid configuration: {config_errors}")


# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for music generation"""
    lyrics: str = Field(..., description="Lyrics in LRC format", min_length=10)
    style_prompt: str = Field(..., description="Musical style description", min_length=3)
    audio_length: int = Field(95, description="Audio length in seconds", ge=95, le=285)
    batch_size: int = Field(1, description="Number of generations", ge=1, le=4)
    steps: Optional[int] = Field(None, description="Override ODE integration steps (higher = better quality, slower)")
    cfg_strength: Optional[float] = Field(None, description="Override CFG strength (higher = more prompt adherence)")
    preset: Optional[str] = Field("high", description="Quality preset: preview, draft, standard, high, maximum, ultra")
    
    @validator("preset")
    def validate_preset(cls, v):
        if v is not None and v not in ["preview", "draft", "standard", "high", "maximum", "ultra"]:
            raise ValueError("preset must be one of: preview, draft, standard, high, maximum, ultra")
        return v
    auto_master: bool = Field(True, description="Automatically apply mastering to output")
    master_preset: str = Field("balanced", description="Mastering preset: subtle, balanced, loud, broadcast")
    payment_intent_id: Optional[str] = Field(None, description="Stripe payment intent ID for payment verification")
    
    @validator("audio_length")
    def validate_audio_length(cls, v):
        if v != 95 and not (96 <= v <= 285):
            raise ValueError("audio_length must be exactly 95 or between 96-285")
        return v
    
    @validator("batch_size")
    def validate_batch_size(cls, v):
        if v > Config.MAX_BATCH_SIZE:
            raise ValueError(f"batch_size cannot exceed {Config.MAX_BATCH_SIZE}")
        return v


class GenerationResponse(BaseModel):
    """Response model for generation submission"""
    job_id: str
    status: str
    queue_position: Optional[int] = None
    estimated_wait_minutes: Optional[int] = None
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str  # queued, processing, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None
    estimated_wait_minutes: Optional[int] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: bool
    device: str
    queue_length: int
    active_jobs: int
    version: str


# Model Manager
class ModelManager:
    """Manages model loading and state"""
    
    def __init__(self):
        self.cfm_model: Optional[torch.nn.Module] = None
        self.vae_model: Optional[torch.nn.Module] = None
        self.tokenizer = None
        self.muq_model: Optional[torch.nn.Module] = None
        self.device = Config.DEVICE
        self.is_loaded = False
        self._lock = threading.Lock()
    
    def load(self):
        """Load all models"""
        with self._lock:
            if self.is_loaded:
                logger.info("Models already loaded")
                return
            
            logger.info(f"Loading DiffRhythm models on {self.device}...")
            try:
                self.cfm_model, self.tokenizer, self.muq_model, self.vae_model = prepare_model(
                    Config.MODEL_MAX_FRAMES,
                    self.device
                )
                self.is_loaded = True
                metrics.set_models_loaded(True)
                logger.info("Models loaded successfully!")
            except Exception as e:
                logger.critical(f"Failed to load models: {e}", exc_info=True)
                metrics.set_models_loaded(False)
                raise ModelNotLoadedError(f"Model loading failed: {str(e)}")
    
    def unload(self):
        """Unload models to free memory"""
        with self._lock:
            if not self.is_loaded:
                return
            
            logger.info("Unloading models...")
            del self.cfm_model
            del self.vae_model
            del self.tokenizer
            del self.muq_model
            self.cfm_model = None
            self.vae_model = None
            self.tokenizer = None
            self.muq_model = None
            self.is_loaded = False
            metrics.set_models_loaded(False)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Models unloaded")


# Job Manager
class JobManager:
    """Manages generation jobs and queue"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_queue: queue.Queue = queue.Queue()
        self.processing_lock = threading.Lock()
        self.current_job: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
    
    def start_worker(self, model_manager: ModelManager):
        """Start the worker thread"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker,
                args=(model_manager,),
                daemon=True
            )
            self._worker_thread.start()
            logger.info("Job worker thread started")
    
    def _worker(self, model_manager: ModelManager):
        """Worker thread that processes jobs"""
        while True:
            try:
                job_id = self.job_queue.get(timeout=1)
                
                with self.processing_lock:
                    self.current_job = job_id
                    self.jobs[job_id]["status"] = "processing"
                    self.jobs[job_id]["started_at"] = datetime.utcnow().isoformat() + "Z"
                
                metrics.set_active_jobs(1)
                metrics.set_queue_length(self.job_queue.qsize())
                
                try:
                    self._process_job(job_id, model_manager)
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["error"] = str(e)
                    metrics.increment_error("generation_error")
                finally:
                    self.jobs[job_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
                    with self.processing_lock:
                        self.current_job = None
                    metrics.set_active_jobs(0)
                    self.job_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}", exc_info=True)
    
    def _process_job(self, job_id: str, model_manager: ModelManager):
        """Process a single generation job"""
        job = self.jobs[job_id]
        start_time = datetime.utcnow()
        
        logger.info(f"Processing job {job_id}: {job['style_prompt'][:50]}...")
        
        if not model_manager.is_loaded:
            raise ModelNotLoadedError("Models not loaded")
        
        # Determine max_frames based on audio length
        max_frames = 2048 if job["audio_length"] == 95 else 6144
        
        # Process inputs
        lrc_prompt, start_time_token, end_frame, song_duration = get_lrc_token(
            max_frames,
            job["lyrics"],
            model_manager.tokenizer,
            job["audio_length"],
            model_manager.device
        )
        
        style_prompt = get_style_prompt(
            model_manager.muq_model,
            prompt=job["style_prompt"]
        )
        negative_style_prompt = get_negative_style_prompt(model_manager.device)
        
        # Generate latent prompt
        latent_prompt = torch.zeros(1, max_frames, 64).to(model_manager.device)
        pred_frames = [(0, max_frames)]
        
        # Determine quality settings
        steps = job.get("steps")
        cfg_strength = job.get("cfg_strength")
        preset = job.get("preset", "high")  # Default to "high" for Suno-style quality
        
        # Apply quality preset
        try:
            from infer.quality_presets import get_preset
            quality_preset = get_preset(preset)
            if steps is None:
                steps = quality_preset.steps
            if cfg_strength is None:
                cfg_strength = quality_preset.cfg_strength
            logger.info(f"Using quality preset '{preset}': {steps} steps, CFG {cfg_strength}")
        except Exception as e:
            logger.warning(f"Failed to load preset '{preset}': {e}, using defaults")
            # Fallback to defaults
            if steps is None:
                steps = Config.CPU_STEPS if model_manager.device == "cpu" else 32
            if cfg_strength is None:
                cfg_strength = Config.CPU_CFG_STRENGTH if model_manager.device == "cpu" else 4.0
        
        # Use defaults if not specified (fallback)
        if steps is None:
            steps = Config.CPU_STEPS if model_manager.device == "cpu" else 32
        if cfg_strength is None:
            cfg_strength = Config.CPU_CFG_STRENGTH if model_manager.device == "cpu" else 4.0
        
        logger.info(f"Generation settings: steps={steps}, cfg_strength={cfg_strength}, device={model_manager.device}")
        
        # Run inference
        generated_songs = inference(
            cfm_model=model_manager.cfm_model,
            vae_model=model_manager.vae_model,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=end_frame,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time_token,
            pred_frames=pred_frames,
            chunked=True,
            batch_infer_num=job["batch_size"],
            song_duration=song_duration,
            steps=steps,
            cfg_strength=cfg_strength
        )
        
        if not generated_songs:
            raise GenerationError("No audio generated")
        
        # Save output
        output_dir = Config.OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output_fixed.wav"
        
        # Save raw audio at high quality (44.1kHz, 16-bit)
        save_audio_robust(generated_songs[0], str(output_path), sample_rate=44100)
        logger.info(f"Raw audio saved to {output_path}")
        
        # Apply mastering if requested
        final_output_path = output_path
        if job.get("auto_master", False):
            try:
                from post_processing.mastering import master_audio_file
                master_preset = job.get("master_preset", "balanced")
                mastered_path = output_dir / "output_mastered.wav"
                logger.info(f"Applying mastering with preset: {master_preset}")
                master_audio_file(str(output_path), str(mastered_path), preset=master_preset, verbose=False)
                final_output_path = mastered_path
                logger.info(f"Mastered audio saved to {mastered_path}")
            except ImportError:
                logger.warning("post_processing.mastering not available, skipping mastering")
            except Exception as e:
                logger.error(f"Mastering failed: {e}", exc_info=True)
                logger.info(f"Raw output still available at {output_path}")
        
        job["status"] = "completed"
        job["output_file"] = f"{job_id}/{final_output_path.name}"
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        metrics.record_generation_duration(duration)
        metrics.increment_generation("success")
        
        logger.info(f"Job {job_id} completed in {duration:.1f}s")
    
    def create_job(self, request: GenerationRequest) -> str:
        """Create a new generation job"""
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "lyrics": request.lyrics,
            "style_prompt": request.style_prompt,
            "audio_length": request.audio_length,
            "batch_size": request.batch_size,
            "steps": request.steps,
            "cfg_strength": request.cfg_strength,
            "preset": request.preset or "high",  # Default to "high" if None
            "auto_master": request.auto_master,
            "master_preset": request.master_preset,
            "payment_intent_id": request.payment_intent_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "started_at": None,
            "completed_at": None,
            "output_file": None,
            "error": None
        }
        
        self.job_queue.put(job_id)
        metrics.set_queue_length(self.job_queue.qsize())
        
        return job_id
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if job_id not in self.jobs:
            raise JobNotFoundError(job_id)
        
        job = self.jobs[job_id].copy()
        
        # Add queue position if queued
        if job["status"] == "queued":
            queue_list = list(self.job_queue.queue)
            if job_id in queue_list:
                job["queue_position"] = queue_list.index(job_id) + 1
                job["estimated_wait_minutes"] = job["queue_position"] * 20
        
        return job
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "queue_length": self.job_queue.qsize(),
            "current_job": self.current_job,
            "estimated_wait_minutes": self.job_queue.qsize() * 20
        }


# Global instances
model_manager = ModelManager()
job_manager = JobManager()


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting DiffRhythm API...")
    try:
        model_manager.load()
        job_manager.start_worker(model_manager)
        logger.info("API started successfully")
    except Exception as e:
        logger.critical(f"Failed to start API: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down DiffRhythm API...")
    model_manager.unload()


# Create FastAPI app
app = FastAPI(
    title="DiffRhythm API",
    version="1.0.0",
    description="Production-ready API for DiffRhythm music generation",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    **get_cors_config()
)

# Add rate limiting exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Middleware for request/response logging and metrics
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware for logging and metrics"""
    start_time = datetime.utcnow()
    
    # Log request
    request_logger.log_request(
        method=request.method,
        path=request.url.path
    )
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Update metrics
        metrics.increment_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        )
        metrics.record_request_duration(
            method=request.method,
            endpoint=request.url.path,
            duration=(datetime.utcnow() - start_time).total_seconds()
        )
        
        # Log response
        request_logger.log_response(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration
        )
        
        # Add security headers
        for header, value in get_security_headers().items():
            response.headers[header] = value
        
        return response
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Request error: {e}", exc_info=True)
        metrics.increment_error("request_error")
        raise


# API Key dependency
async def verify_api_key_dependency(api_key: Optional[str] = Depends(get_api_key)):
    """Dependency to verify API key"""
    if Config.API_KEY:
        check_api_key(api_key)
    return api_key


# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "DiffRhythm API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /api/v1/generate": "Submit generation job",
            "GET /api/v1/status/{job_id}": "Check job status",
            "GET /api/v1/download/{job_id}": "Download generated audio",
            "GET /api/v1/health": "Health check",
            "GET /api/v1/metrics": "Prometheus metrics",
            "GET /api/v1/queue": "Queue status",
            "GET /docs": "OpenAPI documentation"
        }
    }


@app.get(f"{Config.API_PREFIX}/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    queue_status = job_manager.get_queue_status()
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        models_loaded=model_manager.is_loaded,
        device=model_manager.device,
        queue_length=queue_status["queue_length"],
        active_jobs=1 if job_manager.current_job else 0,
        version="1.0.0"
    )


# Route alias for frontend compatibility (/api/generate -> /api/v1/generate)
@app.post(
    "/api/generate",
    response_model=GenerationResponse,
    tags=["Generation"],
    dependencies=[Depends(verify_api_key_dependency)],
    include_in_schema=False  # Don't show in docs, use /api/v1/generate instead
)
@limiter.limit(f"{Config.RATE_LIMIT_PER_HOUR}/hour")
async def generate_music_alias(
    request: GenerationRequest,
    api_request: Request
):
    """Alias for /api/v1/generate for frontend compatibility"""
    return await generate_music(request, api_request)


@app.post(
    f"{Config.API_PREFIX}/generate",
    response_model=GenerationResponse,
    tags=["Generation"],
    dependencies=[Depends(verify_api_key_dependency)]
)
@limiter.limit(f"{Config.RATE_LIMIT_PER_HOUR}/hour")
async def generate_music(
    request: GenerationRequest,
    api_request: Request
):
    """Submit a music generation job"""
    if not model_manager.is_loaded:
        raise ModelNotLoadedError("Models not loaded. Please wait for initialization.")
    
    # Payment verification if required
    if check_payment_required():
        if not request.payment_intent_id:
            raise HTTPException(
                status_code=402,
                detail="Payment required. Please provide a valid payment_intent_id."
            )
        
        is_valid, error_msg = verify_payment_intent(request.payment_intent_id)
        if not is_valid:
            logger.warning(f"Payment verification failed for {request.payment_intent_id}: {error_msg}")
            raise HTTPException(
                status_code=402,
                detail=f"Payment verification failed: {error_msg}"
            )
        logger.info(f"Payment verified for generation request: {request.payment_intent_id}")
    elif request.payment_intent_id:
        # Optional payment verification even if not required
        is_valid, error_msg = verify_payment_intent(request.payment_intent_id)
        if not is_valid:
            logger.warning(f"Optional payment verification failed: {error_msg}")
            # Don't fail the request if payment is optional, just log warning
    
    try:
        job_id = job_manager.create_job(request)
        queue_status = job_manager.get_queue_status()
        
        queue_position = queue_status["queue_length"]
        estimated_wait = queue_position * 20  # 20 minutes per job
        
        logger.info(f"Created generation job {job_id}")
        metrics.increment_generation("queued")
        
        return GenerationResponse(
            job_id=job_id,
            status="queued",
            queue_position=queue_position,
            estimated_wait_minutes=estimated_wait,
            message="Job queued successfully. Use /status/{job_id} to check progress."
        )
    except Exception as e:
        logger.error(f"Error creating job: {e}", exc_info=True)
        metrics.increment_error("job_creation_error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{Config.API_PREFIX}/status/{{job_id}}",
    response_model=JobStatusResponse,
    tags=["Generation"]
)
async def get_job_status(job_id: str):
    """Get the status of a generation job"""
    try:
        job = job_manager.get_job(job_id)
        return JobStatusResponse(**job)
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    f"{Config.API_PREFIX}/download/{{job_id}}",
    tags=["Generation"]
)
async def download_audio(job_id: str):
    """Download generated audio file"""
    try:
        job = job_manager.get_job(job_id)
        
        if job["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job status is {job['status']}. Only completed jobs can be downloaded."
            )
        
        if not job.get("output_file"):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        file_path = Config.OUTPUT_DIR / job["output_file"]
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename="generated_song.wav"
        )
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(f"{Config.API_PREFIX}/queue", tags=["Status"])
async def get_queue_status():
    """Get current queue status"""
    return job_manager.get_queue_status()


@app.get(Config.METRICS_PATH, tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not Config.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return metrics.get_metrics_response()


@app.post("/api/webhooks/stripe", tags=["Webhooks"])
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="stripe-signature")
):
    """
    Stripe webhook endpoint for payment events
    
    Handles:
    - payment_intent.succeeded: Payment completed successfully
    - payment_intent.payment_failed: Payment failed
    - payment_intent.canceled: Payment canceled
    """
    if not Config.STRIPE_WEBHOOK_SECRET:
        logger.warning("Stripe webhook secret not configured, rejecting webhook")
        raise HTTPException(status_code=403, detail="Webhook secret not configured")
    
    if not stripe_signature:
        logger.warning("Missing Stripe signature header")
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")
    
    try:
        body = await request.body()
        
        # Verify webhook signature
        try:
            if not STRIPE_AVAILABLE:
                raise HTTPException(status_code=500, detail="Stripe library not available")
            
            # Import stripe - it's already available via payment_verification import
            try:
                import stripe
            except ImportError:
                raise HTTPException(status_code=500, detail="Stripe library not available")
            
            event = stripe.Webhook.construct_event(
                body,
                stripe_signature,
                Config.STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Handle the event
        event_type = event.get("type")
        event_data = event.get("data", {}).get("object", {})
        
        logger.info(f"Received Stripe webhook: {event_type}")
        
        if event_type == "payment_intent.succeeded":
            payment_intent_id = event_data.get("id")
            amount = event_data.get("amount", 0)
            logger.info(f"Payment succeeded: {payment_intent_id} (${amount/100:.2f})")
            # Payment is verified, generation can proceed
            # Additional processing can be added here (e.g., update database, send notification)
            
        elif event_type == "payment_intent.payment_failed":
            payment_intent_id = event_data.get("id")
            logger.warning(f"Payment failed: {payment_intent_id}")
            # Handle failed payment (e.g., notify user, log for review)
            
        elif event_type == "payment_intent.canceled":
            payment_intent_id = event_data.get("id")
            logger.info(f"Payment canceled: {payment_intent_id}")
            # Handle canceled payment
            
        else:
            logger.info(f"Unhandled event type: {event_type}")
        
        return {"status": "success", "event_type": event_type}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Webhook processing failed")


# Error handlers
@app.exception_handler(DiffRhythmException)
async def diffrhythm_exception_handler(request: Request, exc: DiffRhythmException):
    """Handle DiffRhythm exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    metrics.increment_error("unhandled_exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if Config.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.api:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
