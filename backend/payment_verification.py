"""
Payment verification module for Stripe payment intents
"""
import os
import logging
from typing import Optional
from backend.config import Config

logger = logging.getLogger(__name__)

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    stripe = None
    STRIPE_AVAILABLE = False
    logger.warning("Stripe library not available. Payment verification will be disabled.")


def verify_payment_intent(payment_intent_id: str, expected_price_cents: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """
    Verify that a payment intent is confirmed and valid
    
    Args:
        payment_intent_id: Stripe payment intent ID
        expected_price_cents: Optional expected price in cents for validation
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if not STRIPE_AVAILABLE:
        if Config.REQUIRE_PAYMENT_FOR_GENERATION:
            return False, "Stripe library not available but payment is required"
        logger.warning("Stripe not available, skipping payment verification")
        return True, None
    
    if not Config.STRIPE_SECRET_KEY:
        if Config.REQUIRE_PAYMENT_FOR_GENERATION:
            return False, "Stripe secret key not configured but payment is required"
        logger.warning("Stripe secret key not configured, skipping payment verification")
        return True, None
    
    try:
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        # Retrieve payment intent from Stripe
        payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
        # Verify payment status is succeeded
        if payment_intent.status != "succeeded":
            return False, f"Payment intent status is '{payment_intent.status}', expected 'succeeded'"
        
        # Verify amount if expected price provided
        if expected_price_cents is not None:
            price_variance = max(1, int(expected_price_cents * 0.01))  # 1% variance
            if abs(payment_intent.amount - expected_price_cents) > price_variance:
                return False, (
                    f"Price mismatch: payment amount is ${payment_intent.amount/100:.2f}, "
                    f"expected ${expected_price_cents/100:.2f}"
                )
        
        logger.info(f"✅ Payment verified: {payment_intent_id} (${payment_intent.amount/100:.2f})")
        return True, None
        
    except stripe.error.InvalidRequestError as e:
        logger.error(f"Invalid payment intent ID: {e}")
        return False, f"Invalid payment intent ID: {str(e)}"
    except stripe.error.AuthenticationError as e:
        logger.error(f"Stripe authentication error: {e}")
        return False, "Stripe authentication failed"
    except Exception as e:
        logger.error(f"Payment verification error: {e}", exc_info=True)
        return False, f"Payment verification failed: {str(e)}"


def check_payment_required() -> bool:
    """Check if payment is required for generation"""
    return Config.REQUIRE_PAYMENT_FOR_GENERATION
