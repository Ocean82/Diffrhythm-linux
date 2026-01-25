"""
Payment Schemas
Request and response models for payment processing
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class CreatePaymentIntentRequest(BaseModel):
    """Request to create payment intent"""

    song_id: Optional[str] = Field(
        None,
        description="Song UUID to purchase (for existing songs)",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    duration_seconds: Optional[int] = Field(
        None,
        ge=95,
        le=240,
        description="Song duration in seconds (for new generation)",
        examples=[120],
    )
    amount_cents: int = Field(..., ge=1, description="Amount in cents", examples=[200])
    currency: str = Field(
        "usd", description="Currency code (ISO 4217)", examples=["usd"]
    )
    is_extended: bool = Field(
        False, description="Whether this is an extended song (121-240s)"
    )
    commercial_license: bool = Field(
        False, description="Whether commercial license is included"
    )
    bulk_pack_size: Optional[int] = Field(
        None,
        ge=10,
        le=50,
        description="Number of songs in bulk pack (10 or 50)",
        examples=[10, 50],
    )

    @field_validator("song_id")
    @classmethod
    def validate_song_id(cls, v):
        """Validate song ID format if provided"""
        if v is not None and (not v or len(v) < 10):
            raise ValueError("Invalid song ID format")
        return v
    
    @model_validator(mode='after')
    def validate_song_or_duration(self):
        """Validate that either song_id or duration_seconds is provided"""
        if not self.song_id and not self.duration_seconds:
            raise ValueError("Either song_id or duration_seconds must be provided")
        return self

    @field_validator("bulk_pack_size")
    @classmethod
    def validate_bulk_pack_size(cls, v):
        """Validate bulk pack size"""
        if v is not None and v not in [10, 50]:
            raise ValueError("bulk_pack_size must be 10 or 50")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code"""
        v = v.lower()
        if len(v) != 3:
            raise ValueError("Currency code must be 3 characters")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "duration_seconds": 120,
                "amount_cents": 200,
                "currency": "usd",
                "is_extended": False,
                "commercial_license": False,
            }
        }


class CreatePaymentIntentResponse(BaseModel):
    """Response with payment intent"""

    client_secret: str = Field(
        ..., description="Stripe client secret for payment confirmation"
    )
    payment_intent_id: str = Field(..., description="Stripe payment intent ID")
    amount_cents: int = Field(..., description="Amount in cents")
    currency: str = Field(..., description="Currency code")

    class Config:
        json_schema_extra = {
            "example": {
                "client_secret": "pi_xxx_secret_xxx",
                "payment_intent_id": "pi_xxx",
                "amount_cents": 200,
                "currency": "usd",
            }
        }


class CalculatePriceRequest(BaseModel):
    """Request to calculate price"""

    duration_seconds: int = Field(
        ..., ge=95, le=240, description="Song duration in seconds", examples=[120]
    )
    is_extended: bool = Field(
        False, description="Whether this is an extended song (121-240s)"
    )
    commercial_license: bool = Field(
        False, description="Whether commercial license is included"
    )
    bulk_pack_size: Optional[int] = Field(
        None,
        ge=10,
        le=50,
        description="Number of songs in bulk pack (10 or 50)",
        examples=[10, 50],
    )


class CalculatePriceResponse(BaseModel):
    """Response with calculated price"""

    duration_seconds: int
    song_type: str
    base_price_cents: int
    base_price_dollars: float
    with_commercial_license_cents: int
    with_commercial_license_dollars: float
    commercial_license_addon_cents: int
    commercial_license_addon_dollars: float
    bulk_10_price_cents: int
    bulk_10_price_dollars: float
    bulk_50_price_cents: int
    bulk_50_price_dollars: float
    currency: str = "usd"
