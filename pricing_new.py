"""
Pricing Utilities
Calculate and validate prices based on song duration and purchase options
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Pricing tiers based on duration
SINGLE_2MIN_PRICE_CENTS = 200  # $2.00 for 2-minute (120 seconds) song
EXTENDED_4MIN_PRICE_CENTS = 350  # $3.50 for extended song (up to 4 minutes / 240 seconds)
COMMERCIAL_LICENSE_ADDON_CENTS = 1500  # $15.00 commercial license add-on

# Bulk pack pricing
BULK_10_SONGS_PRICE_CENTS = 1800  # $18.00 for 10 songs (10% discount)
BULK_50_SONGS_PRICE_CENTS = 8000  # $80.00 for 50 songs (20% discount)

# Duration thresholds (in seconds)
SINGLE_SONG_MAX_DURATION = 120  # 2 minutes
EXTENDED_SONG_MAX_DURATION = 240  # 4 minutes

# Validation
MIN_DURATION_SECONDS = 95  # Minimum song duration
MAX_DURATION_SECONDS = 240  # Maximum song duration (4 minutes)

# Legacy constants for backward compatibility
PRICE_PER_MB_CENTS = 100  # $1.00 per MB (legacy)
MIN_PRICE_CENTS = 99  # $0.99 minimum (legacy)
MAX_PRICE_CENTS = 9999  # $99.99 maximum (legacy)


def calculate_price_cents(
    duration_seconds: int,
    is_extended: bool = False,
    commercial_license: bool = False,
    bulk_pack_size: Optional[int] = None
) -> int:
    """
    Calculate price in cents based on song duration and purchase options

    Pricing:
    - Single 2-Minute Song (up to 120s): $2.00
    - Extended Song (121-240s): $3.50
    - Commercial License Add-On: +$15.00
    - Bulk Pack (10 songs): $18.00 (10% off)
    - Bulk Pack (50 songs): $80.00 (20% off)

    Args:
        duration_seconds: Song duration in seconds
        is_extended: Whether this is an extended song (121-240s)
        commercial_license: Whether commercial license is included
        bulk_pack_size: Number of songs in bulk pack (10 or 50), None for single purchase

    Returns:
        Price in cents
    """
    # Handle bulk packs first
    if bulk_pack_size == 10:
        base_price = BULK_10_SONGS_PRICE_CENTS
        logger.info(f"Bulk pack 10 songs: ${base_price/100:.2f}")
    elif bulk_pack_size == 50:
        base_price = BULK_50_SONGS_PRICE_CENTS
        logger.info(f"Bulk pack 50 songs: ${base_price/100:.2f}")
    else:
        # Single song pricing
        if is_extended or duration_seconds > SINGLE_SONG_MAX_DURATION:
            if duration_seconds > EXTENDED_SONG_MAX_DURATION:
                raise ValueError(
                    f"Duration {duration_seconds}s exceeds maximum {EXTENDED_SONG_MAX_DURATION}s (4 minutes)"
                )
            base_price = EXTENDED_4MIN_PRICE_CENTS
            logger.info(f"Extended song ({duration_seconds}s): ${base_price/100:.2f}")
        else:
            if duration_seconds > SINGLE_SONG_MAX_DURATION:
                raise ValueError(
                    f"Duration {duration_seconds}s exceeds single song limit {SINGLE_SONG_MAX_DURATION}s. "
                    f"Use extended song option for durations up to {EXTENDED_SONG_MAX_DURATION}s."
                )
            base_price = SINGLE_2MIN_PRICE_CENTS
            logger.info(f"Single 2-minute song ({duration_seconds}s): ${base_price/100:.2f}")

    # Add commercial license if requested
    total_price = base_price
    if commercial_license:
        total_price += COMMERCIAL_LICENSE_ADDON_CENTS
        logger.info(f"Commercial license add-on: +${COMMERCIAL_LICENSE_ADDON_CENTS/100:.2f}")

    return total_price


def calculate_price_for_duration(duration_seconds: int) -> dict:
    """
    Calculate price options for a given duration

    Args:
        duration_seconds: Song duration in seconds

    Returns:
        Dictionary with pricing options
    """
    if duration_seconds < MIN_DURATION_SECONDS:
        raise ValueError(f"Duration {duration_seconds}s is below minimum {MIN_DURATION_SECONDS}s")
    
    if duration_seconds > MAX_DURATION_SECONDS:
        raise ValueError(f"Duration {duration_seconds}s exceeds maximum {MAX_DURATION_SECONDS}s")

    is_extended = duration_seconds > SINGLE_SONG_MAX_DURATION
    
    base_price_cents = calculate_price_cents(duration_seconds, is_extended=is_extended)
    with_commercial_cents = calculate_price_cents(
        duration_seconds, 
        is_extended=is_extended, 
        commercial_license=True
    )

    return {
        "duration_seconds": duration_seconds,
        "song_type": "extended" if is_extended else "single",
        "base_price_cents": base_price_cents,
        "base_price_dollars": base_price_cents / 100,
        "with_commercial_license_cents": with_commercial_cents,
        "with_commercial_license_dollars": with_commercial_cents / 100,
        "commercial_license_addon_cents": COMMERCIAL_LICENSE_ADDON_CENTS,
        "commercial_license_addon_dollars": COMMERCIAL_LICENSE_ADDON_CENTS / 100,
        "bulk_10_price_cents": BULK_10_SONGS_PRICE_CENTS,
        "bulk_10_price_dollars": BULK_10_SONGS_PRICE_CENTS / 100,
        "bulk_50_price_cents": BULK_50_SONGS_PRICE_CENTS,
        "bulk_50_price_dollars": BULK_50_SONGS_PRICE_CENTS / 100,
        "currency": "usd"
    }


def validate_price_for_duration(
    amount_cents: int,
    duration_seconds: int,
    is_extended: bool = False,
    commercial_license: bool = False,
    bulk_pack_size: Optional[int] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate that the price matches the duration and options

    Args:
        amount_cents: Price in cents being charged
        duration_seconds: Song duration in seconds
        is_extended: Whether this is an extended song
        commercial_license: Whether commercial license is included
        bulk_pack_size: Number of songs in bulk pack (10 or 50), None for single

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        expected_price = calculate_price_cents(
            duration_seconds,
            is_extended=is_extended,
            commercial_license=commercial_license,
            bulk_pack_size=bulk_pack_size
        )
    except ValueError as e:
        return False, str(e)

    # Allow small variance (rounding differences)
    price_variance = max(1, int(expected_price * 0.01))  # 1% variance or 1 cent minimum

    if abs(amount_cents - expected_price) > price_variance:
        expected_dollars = expected_price / 100
        actual_dollars = amount_cents / 100
        return False, (
            f"Price mismatch: Expected ${expected_dollars:.2f} "
            f"for {duration_seconds}s song (extended={is_extended}, "
            f"commercial={commercial_license}, bulk={bulk_pack_size}), "
            f"but received ${actual_dollars:.2f}."
        )

    return True, None


def get_price_tier(duration_seconds: int) -> str:
    """
    Get pricing tier name for a duration

    Args:
        duration_seconds: Song duration in seconds

    Returns:
        Tier name: "single", "extended", or "invalid"
    """
    if duration_seconds < MIN_DURATION_SECONDS:
        return "invalid"
    
    if duration_seconds <= SINGLE_SONG_MAX_DURATION:
        return "single"
    elif duration_seconds <= EXTENDED_SONG_MAX_DURATION:
        return "extended"
    else:
        return "invalid"


# Legacy functions for backward compatibility (file size based)
def calculate_price_cents_legacy(file_size_mb: Optional[float]) -> int:
    """
    Legacy function: Calculate price based on file size
    Kept for backward compatibility with existing code
    """
    logger.warning("Using legacy file-size based pricing. Consider migrating to duration-based pricing.")
    PRICE_PER_MB_CENTS = 100
    MIN_PRICE_CENTS = 99
    
    if file_size_mb is None or file_size_mb <= 0:
        return MIN_PRICE_CENTS
    
    calculated_price = int(round(file_size_mb * PRICE_PER_MB_CENTS))
    return max(calculated_price, MIN_PRICE_CENTS)


def validate_price_for_file_size(
    amount_cents: int, file_size_mb: Optional[float]
) -> tuple[bool, Optional[str]]:
    """
    Legacy function: Validate price against file size
    Kept for backward compatibility with existing song purchases
    
    Args:
        amount_cents: Price in cents being charged
        file_size_mb: File size in megabytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    logger.warning("Using legacy file-size based price validation. Consider migrating to duration-based pricing.")
    
    PRICE_PER_MB_CENTS = 100
    MIN_PRICE_CENTS = 99
    MAX_PRICE_CENTS = 9999
    
    if file_size_mb is None or file_size_mb <= 0:
        # If file size is unknown, allow minimum price
        if amount_cents < MIN_PRICE_CENTS:
            return False, f"Price must be at least ${MIN_PRICE_CENTS/100:.2f}"
        if amount_cents > MAX_PRICE_CENTS:
            return False, f"Price cannot exceed ${MAX_PRICE_CENTS/100:.2f}"
        return True, None

    # Calculate expected price
    expected_price = int(round(file_size_mb * PRICE_PER_MB_CENTS))
    expected_price = max(expected_price, MIN_PRICE_CENTS)
    expected_price = min(expected_price, MAX_PRICE_CENTS)

    # Allow small variance (rounding differences)
    price_variance = max(1, int(expected_price * 0.05))  # 5% variance or 1 cent minimum

    if abs(amount_cents - expected_price) > price_variance:
        expected_dollars = expected_price / 100
        actual_dollars = amount_cents / 100
        return False, (
            f"Price mismatch: Expected ${expected_dollars:.2f} "
            f"for {file_size_mb:.2f}MB file, but received ${actual_dollars:.2f}. "
            f"Price is calculated as $1.00 per MB."
        )

    return True, None
