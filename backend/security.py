"""
Security utilities for DiffRhythm API
"""
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.config import Config
from backend.exceptions import RateLimitError

# Re-export for convenience
__all__ = [
    "limiter",
    "RateLimitExceeded",
    "_rate_limit_exceeded_handler",
    "get_api_key",
    "check_api_key",
    "get_cors_config",
    "get_security_headers",
    "verify_api_key",
    "api_key_header"
]


# Rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=Config.ENABLE_RATE_LIMIT
)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = None) -> bool:
    """Verify API key if configured using constant-time comparison"""
    if Config.API_KEY is None:
        return True  # No API key required
    
    if api_key is None:
        return False
    
    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(api_key, Config.API_KEY)


async def get_api_key(request: Request) -> Optional[str]:
    """Extract API key from request"""
    return await api_key_header(request)


def check_api_key(api_key: Optional[str]):
    """Check API key and raise exception if invalid"""
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )


def get_cors_config() -> dict:
    """Get CORS configuration for frontend connections"""
    # Handle CORS_ORIGINS - support both string and list formats
    cors_origins = Config.CORS_ORIGINS
    
    # If it's a string, check if it's "*" or a comma-separated list
    if isinstance(cors_origins, str):
        if cors_origins.strip() == "*":
            allow_origins = ["*"]
        else:
            # Split by comma and clean up
            allow_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
            # If "*" is in the list, use ["*"]
            if "*" in allow_origins:
                allow_origins = ["*"]
    elif isinstance(cors_origins, list):
        # If it's already a list
        if "*" in cors_origins:
            allow_origins = ["*"]
        else:
            allow_origins = [origin.strip() if isinstance(origin, str) else str(origin) for origin in cors_origins]
    else:
        # Default to allow all
        allow_origins = ["*"]
    
    return {
        "allow_origins": allow_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        "allow_headers": ["*"],
        "expose_headers": ["*"],
    }


def get_security_headers() -> dict:
    """Get security headers for responses"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }
