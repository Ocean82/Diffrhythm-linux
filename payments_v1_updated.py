"""
Payments API v1
Payment processing endpoints with duration-based pricing
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path as PathParam
from typing import Optional
import logging

try:
    import stripe
    from stripe import error as stripe_error
except ImportError:
    stripe = None
    stripe_error = None

from ...schemas.payments import (
    CreatePaymentIntentRequest,
    CreatePaymentIntentResponse,
    CalculatePriceRequest,
    CalculatePriceResponse,
)
from ...dependencies import get_current_user, get_database_service
from ...services.database_service import DatabaseService
from ...config import settings
from ...models.database import Song
from ...utils.pricing import (
    calculate_price_cents,
    calculate_price_for_duration,
    validate_price_for_duration,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize Stripe
if stripe and settings.STRIPE_SECRET_KEY:
    stripe.api_key = settings.STRIPE_SECRET_KEY
else:
    logger.warning("⚠️  Stripe not configured")


@router.get(
    "/calculate-price",
    response_model=CalculatePriceResponse,
    summary="Calculate price",
    description="Calculate price for a song based on duration and options",
)
async def calculate_price(
    duration: int = Query(..., ge=95, le=240, description="Song duration in seconds"),
    is_extended: bool = Query(False, description="Whether this is an extended song"),
    commercial_license: bool = Query(False, description="Include commercial license"),
    bulk_pack_size: Optional[int] = Query(
        None, ge=10, le=50, description="Bulk pack size (10 or 50)"
    ),
):
    """
    Calculate price for a song based on duration and purchase options

    Returns pricing information including:
    - Base price
    - Price with commercial license
    - Bulk pack prices
    """
    try:
        if bulk_pack_size and bulk_pack_size not in [10, 50]:
            raise HTTPException(
                status_code=400, detail="bulk_pack_size must be 10 or 50"
            )

        # Calculate price for single song
        price_info = calculate_price_for_duration(duration)

        # If bulk pack, calculate bulk price
        if bulk_pack_size:
            bulk_price_cents = calculate_price_cents(
                duration,
                is_extended=is_extended,
                commercial_license=commercial_license,
                bulk_pack_size=bulk_pack_size,
            )
            return {
                "duration_seconds": duration,
                "song_type": "extended" if is_extended or duration > 120 else "single",
                "base_price_cents": bulk_price_cents,
                "base_price_dollars": bulk_price_cents / 100,
                "with_commercial_license_cents": bulk_price_cents
                + 1500,  # Add commercial license
                "with_commercial_license_dollars": (bulk_price_cents + 1500) / 100,
                "commercial_license_addon_cents": 1500,
                "commercial_license_addon_dollars": 15.0,
                "bulk_10_price_cents": 1800,
                "bulk_10_price_dollars": 18.0,
                "bulk_50_price_cents": 8000,
                "bulk_50_price_dollars": 80.0,
                "currency": "usd",
            }

        return price_info

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Price calculation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error calculating price. Please try again."
        )


@router.post(
    "/create-intent",
    response_model=CreatePaymentIntentResponse,
    status_code=201,
    summary="Create payment intent",
    description="Create Stripe payment intent for song purchase or generation",
    responses={
        201: {"description": "Payment intent created successfully"},
        400: {"description": "Bad request (invalid amount, song already purchased)"},
        401: {"description": "Authentication required"},
        402: {"description": "Payment declined"},
        404: {"description": "Song not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        502: {"description": "Payment service error"},
        503: {"description": "Payment service unavailable"},
    },
)
async def create_payment_intent(
    request: CreatePaymentIntentRequest,
    user_id: str = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Create Stripe payment intent for song purchase or generation

    Supports:
    - Purchasing existing songs (requires song_id)
    - Pre-paying for new generation (requires duration_seconds)
    - Commercial license add-on
    - Bulk pack purchases
    """
    if not stripe or not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    try:
        # Determine if this is for existing song or new generation
        if request.song_id:
            # Existing song purchase
            if not db_service.session:
                raise HTTPException(
                    status_code=500, detail="Database not configured"
                )

            song = (
                db_service.session.query(Song)
                .filter(Song.id == request.song_id)
                .first()
            )

            if not song:
                raise HTTPException(status_code=404, detail="Song not found")

            # Check if already purchased
            has_purchase = db_service.verify_purchase(user_id, request.song_id)
            if has_purchase:
                raise HTTPException(
                    status_code=400, detail="Song already purchased"
                )

            # Validate price (legacy file-size based for existing songs)
            from ...utils.pricing import validate_price_for_file_size

            file_size_mb = float(song.file_size_mb) if song.file_size_mb else None
            is_valid, error_message = validate_price_for_file_size(
                request.amount_cents, file_size_mb
            )
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_message)

            metadata = {
                "song_id": request.song_id,
                "user_id": user_id,
                "file_size_mb": str(song.file_size_mb) if song.file_size_mb else "unknown",
                "purchase_type": "existing_song",
            }

        elif request.duration_seconds:
            # New generation - validate duration-based pricing
            is_extended = (
                request.is_extended
                or request.duration_seconds > 120
            )

            # Calculate expected price
            expected_price = calculate_price_cents(
                request.duration_seconds,
                is_extended=is_extended,
                commercial_license=request.commercial_license,
                bulk_pack_size=request.bulk_pack_size,
            )

            # Validate price matches
            is_valid, error_message = validate_price_for_duration(
                request.amount_cents,
                request.duration_seconds,
                is_extended=is_extended,
                commercial_license=request.commercial_license,
                bulk_pack_size=request.bulk_pack_size,
            )

            if not is_valid:
                logger.warning(
                    f"Price validation failed: expected={expected_price}, "
                    f"received={request.amount_cents}, error={error_message}"
                )
                raise HTTPException(status_code=400, detail=error_message)

            metadata = {
                "user_id": user_id,
                "duration_seconds": str(request.duration_seconds),
                "is_extended": str(is_extended),
                "commercial_license": str(request.commercial_license),
                "bulk_pack_size": str(request.bulk_pack_size) if request.bulk_pack_size else "none",
                "purchase_type": "generation",
                "expected_price_cents": str(expected_price),
            }

        else:
            raise HTTPException(
                status_code=400,
                detail="Either song_id or duration_seconds must be provided",
            )

        # Create payment intent
        payment_intent = stripe.PaymentIntent.create(
            amount=request.amount_cents,
            currency=request.currency,
            metadata=metadata,
            automatic_payment_methods={"enabled": True},
        )

        logger.info(
            f"✅ Created payment intent: {payment_intent.id} "
            f"for user {user_id}, amount=${request.amount_cents/100:.2f}"
        )

        return CreatePaymentIntentResponse(
            client_secret=payment_intent.client_secret,
            payment_intent_id=payment_intent.id,
            amount_cents=request.amount_cents,
            currency=request.currency,
        )

    except HTTPException:
        raise
    except stripe_error.CardError as e:
        logger.error(
            f"❌ Stripe CardError: {e.user_message if hasattr(e, 'user_message') else str(e)}"
        )
        raise HTTPException(
            status_code=402,
            detail=(
                e.user_message
                if hasattr(e, "user_message") and e.user_message
                else "Your card was declined. Please check your card details or try a different payment method."
            ),
        )
    except stripe_error.RateLimitError as e:
        logger.error(f"❌ Stripe RateLimitError: {e}")
        raise HTTPException(
            status_code=429,
            detail="Too many payment requests. Please wait a moment and try again.",
        )
    except stripe_error.InvalidRequestError as e:
        logger.error(f"❌ Stripe InvalidRequestError: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payment request: {e.user_message if hasattr(e, 'user_message') and e.user_message else 'Please check your payment details and try again.'}",
        )
    except stripe_error.AuthenticationError as e:
        logger.error(f"❌ Stripe AuthenticationError: {e}")
        raise HTTPException(
            status_code=500,
            detail="Payment service authentication failed. Please contact support.",
        )
    except stripe_error.APIConnectionError as e:
        logger.error(f"❌ Stripe APIConnectionError: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to payment service. Please try again in a moment.",
        )
    except stripe_error.APIError as e:
        logger.error(f"❌ Stripe APIError: {e}")
        raise HTTPException(
            status_code=502,
            detail="Payment service error. Please try again or contact support if the issue persists.",
        )
    except stripe_error.StripeError as e:
        logger.error(f"❌ Stripe error: {e}")
        user_message = (
            getattr(e, "user_message", None) if hasattr(e, "user_message") else None
        )
        raise HTTPException(
            status_code=500,
            detail=(
                user_message
                if user_message
                else "Payment processing error. Please try again or contact support."
            ),
        )
    except Exception as e:
        logger.error(f"❌ Payment intent creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again or contact support.",
        )


@router.get(
    "/verify-payment/{payment_intent_id}",
    summary="Verify payment intent",
    description="Verify that a payment intent is confirmed and ready for generation",
)
async def verify_payment_intent(
    payment_intent_id: str = PathParam(..., description="Stripe payment intent ID"),
    user_id: str = Depends(get_current_user),
):
    """
    Verify that a payment intent is confirmed and ready for generation

    Returns payment intent status and metadata
    """
    if not stripe or not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    try:
        payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)

        # Verify user owns this payment intent
        if payment_intent.metadata.get("user_id") != user_id:
            raise HTTPException(
                status_code=403, detail="Payment intent does not belong to this user"
            )

        # Check payment status
        if payment_intent.status == "succeeded":
            return {
                "payment_intent_id": payment_intent_id,
                "status": "succeeded",
                "amount_cents": payment_intent.amount,
                "currency": payment_intent.currency,
                "metadata": payment_intent.metadata,
                "ready_for_generation": True,
            }
        elif payment_intent.status == "requires_payment_method":
            return {
                "payment_intent_id": payment_intent_id,
                "status": "requires_payment_method",
                "ready_for_generation": False,
                "message": "Payment method required",
            }
        elif payment_intent.status == "requires_confirmation":
            return {
                "payment_intent_id": payment_intent_id,
                "status": "requires_confirmation",
                "ready_for_generation": False,
                "message": "Payment requires confirmation",
            }
        else:
            return {
                "payment_intent_id": payment_intent_id,
                "status": payment_intent.status,
                "ready_for_generation": False,
                "message": f"Payment status: {payment_intent.status}",
            }

    except stripe_error.InvalidRequestError as e:
        raise HTTPException(status_code=404, detail="Payment intent not found")
    except Exception as e:
        logger.error(f"Payment verification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error verifying payment. Please try again."
        )
