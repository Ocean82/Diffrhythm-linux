"""
Generation Schemas
Request and response models for song generation
"""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateSongRequest(BaseModel):
    """Request model for song generation"""

    text_prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text description of the song",
        examples=["A happy upbeat pop song about summer"],
    )
    genre: Optional[str] = Field(
        None,
        max_length=50,
        description="Music genre (pop, rock, jazz, etc.)",
        examples=["pop"],
    )
    style: Optional[str] = Field(
        None,
        max_length=100,
        description="Style description (happy, upbeat, slow, etc.)",
        examples=["upbeat"],
    )
    duration: float = Field(
        95.0,
        ge=95.0,
        le=285.0,
        description="Duration in seconds (95 for base, up to 285 for full)",
        examples=[95.0],
    )
    lyrics_path: Optional[str] = Field(
        None,
        max_length=500,
        description="Path to .lrc lyrics file (optional)",
        examples=[None],
    )
    negative_style: Optional[str] = Field(
        None, max_length=200, description="What to avoid in generation", examples=[None]
    )
    chunked: bool = Field(
        True,
        description="Use chunked decoding for CPU (required for CPU)",
        examples=[True],
    )
    payment_intent_id: Optional[str] = Field(
        None,
        description="Stripe payment intent ID (required if payment is enabled)",
        examples=["pi_xxx"],
    )

    def model_post_init(self, __context):
        """Sanitize text inputs after validation"""
        from ..middleware.validation import sanitize_text

        if self.text_prompt:
            self.text_prompt = sanitize_text(self.text_prompt, max_length=1000)
        if self.genre:
            self.genre = sanitize_text(self.genre, max_length=50)
        if self.style:
            self.style = sanitize_text(self.style, max_length=100)
        if self.negative_style:
            self.negative_style = sanitize_text(self.negative_style, max_length=200)

    class Config:
        json_schema_extra = {
            "example": {
                "text_prompt": "A happy upbeat pop song about summer",
                "genre": "pop",
                "style": "upbeat",
                "duration": 95.0,
                "lyrics_path": None,
                "negative_style": None,
                "chunked": True,
                "payment_intent_id": "pi_xxx",
            }
        }


class GenerateSongResponse(BaseModel):
    """Response model for song generation"""

    success: bool = Field(..., description="Generation job started successfully")
    job_id: Optional[str] = Field(None, description="Job ID for tracking progress")
    message: str = Field(..., description="Status message")
    audio_path: Optional[str] = Field(
        None, description="Path to generated audio (if immediate)"
    )
    duration: Optional[float] = Field(None, description="Duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "abc123-def456-ghi789",
                "message": "Generation started. Use /api/generate/{job_id}/status to check progress.",
                "audio_path": None,
                "duration": None,
            }
        }


class GenerationStatusResponse(BaseModel):
    """Response model for generation status"""

    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress (0.0-1.0)")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[dict] = Field(None, description="Result data if completed")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123-def456-ghi789",
                "status": "processing",
                "progress": 0.5,
                "error": None,
                "result": None,
            }
        }
