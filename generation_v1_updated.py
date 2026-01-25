"""
Generation API v1
Song generation endpoints with payment verification
"""

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    BackgroundTasks,
    Path as PathParam,
    Depends,
)
from pathlib import Path
from datetime import datetime
from typing import Optional

from ...schemas.generation import (
    GenerateSongRequest,
    GenerateSongResponse,
    GenerationStatusResponse,
)
from ...dependencies import (
    get_current_user,
    get_diffrhythm_service,
    get_database_service,
)
from ...services.job_queue import get_job_queue, JobStatus
from ...services.s3_service import get_s3_service
from ...services.ai.optimized_diffrhythm_service import OptimizedDiffRhythmService
from ...middleware.rate_limit import rate_limit
from ...config import settings
from ...utils.logger import logger
from ...utils.pricing import calculate_price_cents, validate_price_for_duration

router = APIRouter()


def verify_payment_intent(payment_intent_id: str, user_id: str, expected_price_cents: int) -> bool:
    """
    Verify that a payment intent is confirmed and matches expected price
    
    Args:
        payment_intent_id: Stripe payment intent ID
        user_id: User ID to verify ownership
        expected_price_cents: Expected price in cents
        
    Returns:
        True if payment is confirmed and valid, False otherwise
    """
    try:
        import stripe
        from ...config import settings
        
        if not stripe or not settings.STRIPE_SECRET_KEY:
            logger.warning("Stripe not configured, skipping payment verification")
            return False  # In development, allow without payment if Stripe not configured
        
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
        # Verify user owns this payment intent
        if payment_intent.metadata.get("user_id") != user_id:
            logger.error(f"Payment intent {payment_intent_id} does not belong to user {user_id}")
            return False
        
        # Verify payment is succeeded
        if payment_intent.status != "succeeded":
            logger.warning(f"Payment intent {payment_intent_id} status is {payment_intent.status}, not succeeded")
            return False
        
        # Verify amount matches (allow small variance for rounding)
        price_variance = max(1, int(expected_price_cents * 0.01))  # 1% variance
        if abs(payment_intent.amount - expected_price_cents) > price_variance:
            logger.error(
                f"Price mismatch: payment_intent.amount={payment_intent.amount}, "
                f"expected={expected_price_cents}"
            )
            return False
        
        logger.info(f"✅ Payment verified: {payment_intent_id} for ${expected_price_cents/100:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Payment verification error: {e}", exc_info=True)
        return False


def _generate_song_synchronously(job_id: str, request_data: dict) -> None:
    """
    Synchronous song generation fallback when Celery is not available.
    This function performs the same work as the Celery task but runs synchronously.
    """
    from ...services.job_queue import get_job_queue
    from ...services.cache import get_cache_service

    job_queue = get_job_queue()

    try:
        # Update job status
        job_queue.update_job(job_id, status=JobStatus.PROCESSING, progress=0.1)
        logger.info(f"Starting synchronous generation for job {job_id}")

        # Get services
        service = OptimizedDiffRhythmService(
            device="cpu",
            output_dir=settings.OUTPUT_DIR,
            use_audio_optimization=True,
            use_julius=True,
        )
        s3_service = get_s3_service()
        db_service = get_database_service()

        # Extract request data
        text_prompt = request_data.get("text_prompt")
        lyrics_path = request_data.get("lyrics_path")
        style = request_data.get("style")
        duration = request_data.get("duration", 95.0)
        negative_style = request_data.get("negative_style")
        chunked = request_data.get("chunked", True)
        user_id = request_data.get("user_id")

        # Generate song
        job_queue.update_job(job_id, progress=0.2)
        logger.info(f"Generating song for job {job_id}: {text_prompt[:50]}...")

        result = service.generate(
            text_prompt=text_prompt,
            lyrics_path=lyrics_path,
            style_prompt=style,
            duration=duration,
            negative_style=negative_style,
            chunked=chunked,
        )

        if not result.get("success"):
            error_msg = result.get("error", "Generation failed")
            logger.error(f"Generation failed for job {job_id}: {error_msg}")
            job_queue.update_job(job_id, status=JobStatus.FAILED, error=error_msg)
            return

        job_queue.update_job(job_id, progress=0.7)

        # Upload to S3
        audio_path = result.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            error_msg = "Generated file not found"
            logger.error(f"{error_msg} for job {job_id}")
            job_queue.update_job(job_id, status=JobStatus.FAILED, error=error_msg)
            return

        # Upload to S3
        s3_key = f"songs/{user_id}/{job_id}/{Path(audio_path).name}"
        try:
            s3_url = s3_service.upload_file(audio_path, s3_key)
            logger.info(f"✅ Uploaded to S3: {s3_url}")
        except Exception as s3_error:
            logger.error(f"Failed to upload to S3: {s3_error}", exc_info=True)
            # Continue even if S3 upload fails - file is still available locally

        job_queue.update_job(job_id, progress=0.9)

        # Save to database
        try:
            from ...models.database import Song

            song = Song(
                clerk_user_id=user_id,
                title=text_prompt[:100] if text_prompt else "Untitled",
                s3_key=s3_key,
                s3_url=s3_url if "s3_url" in locals() else None,
                local_path=str(audio_path),
                duration=result.get("duration", duration),
                genre=request_data.get("genre", "unknown"),
                status="completed",
            )
            db_service.session.add(song)
            db_service.session.commit()
            logger.info(f"✅ Saved song to database: {song.id}")
        except Exception as db_error:
            logger.error(f"Failed to save to database: {db_error}", exc_info=True)
            db_service.session.rollback()

        # Mark job as completed
        job_queue.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result={
                "audio_path": audio_path,
                "s3_url": s3_url if "s3_url" in locals() else None,
                "duration": result.get("duration", duration),
            },
        )

        # Invalidate cache
        cache = get_cache_service()
        cache.delete(f"cache:user:songs:{user_id}")

        logger.info(f"✅ Synchronous generation completed for job {job_id}")

    except Exception as e:
        logger.error(
            f"Synchronous generation failed for job {job_id}: {e}", exc_info=True
        )

        # Capture crash with context
        try:
            from ...services.crash_logger import capture_crash

            capture_crash(
                e,
                error_type="generation_failure",
                job_id=job_id,
                user_id=request_data.get("user_id"),
                request_params=request_data,
                model_state="generation_failed",
            )
        except ImportError:
            pass  # Crash logger not available
        except Exception as crash_error:
            logger.error(f"Failed to capture crash: {crash_error}")

        job_queue.update_job(job_id, status=JobStatus.FAILED, error=str(e))


@router.post(
    "",
    response_model=GenerateSongResponse,
    status_code=201,
    summary="Generate a full-length song",
    description="""
    Generate a complete song (music + vocals) from a text prompt.
    
    **Payment Required**: A confirmed payment intent must be provided.
    
    This endpoint starts a background job for song generation.
    Generation typically takes 5-15 minutes depending on duration.
    
    **Rate Limit**: 5 requests per 5 minutes per user
    
    **Authentication**: Required (Bearer token)
    
    **Process**:
    1. Payment intent is verified
    2. Price is validated against requested duration
    3. Request is validated and job created
    4. Background task starts generation
    5. Job ID returned immediately
    6. Use `/api/v1/generate/{job_id}/status` to check progress
    """,
    responses={
        201: {
            "description": "Generation job started successfully",
        },
        400: {
            "description": "Bad request (invalid payment, price mismatch)",
        },
        401: {
            "description": "Authentication required",
        },
        402: {
            "description": "Payment required or payment not confirmed",
        },
        503: {
            "description": "Service unavailable (models not loaded)",
        },
    },
)
@rate_limit(max_calls=5, time_period=300)  # 5 generations per 5 minutes
async def generate_song(
    request: Request,
    body: GenerateSongRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    diffrhythm_service=Depends(get_diffrhythm_service),
):
    """
    Generate a full-length song from text prompt

    This endpoint requires a confirmed payment intent before starting generation.
    Use the status endpoint to check progress.
    """
    from ...services.job_queue import get_job_queue

    # Try to import Celery task, with graceful fallback
    try:
        from ...tasks.generation_tasks import generate_song_task

        CELERY_AVAILABLE = True
    except ImportError as e:
        logger.warning(
            f"⚠️  Celery tasks not available: {e}. Generation will be synchronous."
        )
        CELERY_AVAILABLE = False
        generate_song_task = None

    try:
        logger.info(
            f"Received generation request from user {user_id}: {body.text_prompt[:50]}..."
        )

        # Verify models are loaded before accepting request
        if not diffrhythm_service.are_models_loaded():
            raise HTTPException(
                status_code=503,
                detail="AI models are not loaded. Please try again later or contact support if the issue persists.",
            )

        # Calculate expected price based on duration
        duration_int = int(body.duration)
        is_extended = duration_int > 120
        
        expected_price_cents = calculate_price_cents(
            duration_int,
            is_extended=is_extended,
            commercial_license=False,  # Can be added later if needed
            bulk_pack_size=None,  # Single song generation
        )
        
        logger.info(
            f"Expected price for {duration_int}s song: ${expected_price_cents/100:.2f}"
        )

        # Verify payment if payment_intent_id is provided
        # If Stripe is not configured, allow generation without payment (development mode)
        if body.payment_intent_id:
            payment_verified = verify_payment_intent(
                body.payment_intent_id, user_id, expected_price_cents
            )
            if not payment_verified:
                raise HTTPException(
                    status_code=402,
                    detail="Payment verification failed. Please ensure payment is confirmed and amount matches.",
                )
            logger.info(f"✅ Payment verified for generation: {body.payment_intent_id}")
        else:
            # Check if payment is required (based on settings)
            # In production, payment should be required
            require_payment = getattr(settings, 'REQUIRE_PAYMENT_FOR_GENERATION', False)
            if require_payment:
                raise HTTPException(
                    status_code=402,
                    detail="Payment required. Please create a payment intent before generating.",
                )
            else:
                logger.warning(
                    f"⚠️  Generation without payment for user {user_id} "
                    "(REQUIRE_PAYMENT_FOR_GENERATION is False or not set)"
                )

        # Create job
        job_queue = get_job_queue()
        job_id = job_queue.create_job(
            job_type="generation",
            metadata={
                "text_prompt": body.text_prompt,
                "genre": body.genre,
                "style": body.style,
                "duration": body.duration,
                "lyrics_path": body.lyrics_path,
                "negative_style": body.negative_style,
                "chunked": body.chunked,
                "user_id": user_id,
                "payment_intent_id": payment_intent_id,
                "expected_price_cents": expected_price_cents,
            },
        )

        # Prepare request data for Celery task
        request_data = {
            "text_prompt": body.text_prompt,
            "genre": body.genre,
            "style": body.style,
            "duration": body.duration,
            "lyrics_path": body.lyrics_path,
            "negative_style": body.negative_style,
            "chunked": body.chunked,
            "user_id": user_id,
            "payment_intent_id": payment_intent_id,
        }

        # Start Celery task (or fallback to synchronous if unavailable)
        if CELERY_AVAILABLE and generate_song_task and settings.CELERY_ENABLED:
            try:
                # Verify worker is running before queuing task
                try:
                    from ...services.celery_app import celery_app

                    inspect = celery_app.control.inspect()
                    active_workers = inspect.stats()
                    if not active_workers:
                        logger.warning(
                            f"⚠️  No Celery workers running for job {job_id}. "
                            "Task will be queued but won't process until worker starts."
                        )
                        if settings.CELERY_REQUIRED:
                            # Fail if Celery is required but no workers
                            job_queue.update_job(
                                job_id,
                                status=JobStatus.FAILED,
                                error="No Celery workers available. Please start the worker.",
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Background task processing unavailable. No workers running.",
                            )
                except Exception as worker_check_error:
                    logger.warning(
                        f"Could not verify Celery workers: {worker_check_error}. "
                        "Proceeding with task queue..."
                    )

                # Check if Celery is required
                if settings.CELERY_REQUIRED:
                    # In production, Celery is required
                    generate_song_task.delay(job_id, request_data)
                    logger.info(f"✅ Celery task queued for job {job_id}")
                else:
                    # In development, allow synchronous fallback
                    generate_song_task.delay(job_id, request_data)
                    logger.info(f"✅ Celery task queued for job {job_id}")

                # Invalidate user songs cache (new generation started)
                from ...services.cache import get_cache_service

                cache = get_cache_service()
                cache.delete(f"cache:user:songs:{user_id}")
            except Exception as celery_error:
                logger.error(
                    f"Failed to start Celery task: {celery_error}", exc_info=True
                )

                # If Celery is required, fail the request
                if settings.CELERY_REQUIRED:
                    job_queue.update_job(
                        job_id,
                        status=JobStatus.FAILED,
                        error=f"Failed to start background task: {str(celery_error)}",
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to start generation task. Celery is required but unavailable.",
                    )
                else:
                    # Fallback to synchronous generation
                    logger.warning(
                        f"⚠️  Celery failed, falling back to synchronous generation for job {job_id}"
                    )
        else:
            # Celery not available
            if settings.CELERY_REQUIRED:
                # Celery is required but not available - fail the request
                logger.error(
                    f"❌ Celery is required but not available for job {job_id}"
                )
                job_queue.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    error="Background task processing is required but not available. Please configure Celery.",
                )
                raise HTTPException(
                    status_code=503,
                    detail="Background task processing is required but not available. Please configure Celery worker.",
                )
            else:
                # Celery not required - use synchronous generation
                logger.info(
                    f"ℹ️  Celery not available, using synchronous generation for job {job_id}"
                )
                # Run synchronous generation in background (non-blocking)
                background_tasks.add_task(
                    _generate_song_synchronously, job_id, request_data
                )
                logger.info(f"✅ Started synchronous generation for job {job_id}")

                # Invalidate cache
                from ...services.cache import get_cache_service

                cache = get_cache_service()
                cache.delete(f"cache:user:songs:{user_id}")

        return GenerateSongResponse(
            success=True,
            job_id=job_id,
            message="Generation started. Use /api/v1/generate/{job_id}/status to check progress.",
            audio_path=None,
            duration=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)

        # Capture crash with context
        try:
            from ...services.crash_logger import capture_crash

            request_params = {
                "text_prompt": (
                    body.text_prompt if hasattr(body, "text_prompt") else None
                ),
                "genre": body.genre if hasattr(body, "genre") else None,
                "style": body.style if hasattr(body, "style") else None,
                "duration": body.duration if hasattr(body, "duration") else None,
            }
            capture_crash(
                e,
                error_type="generation_request_failure",
                user_id=user_id if "user_id" in locals() else None,
                request_params=request_params,
                model_state="request_failed",
            )
        except (ImportError, Exception):
            pass  # Crash logger not available or failed

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during generation request. Please try again or contact support.",
        )
