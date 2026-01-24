"""
Custom exception classes for DiffRhythm API
"""
from typing import Optional


class DiffRhythmException(Exception):
    """Base exception for DiffRhythm API"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(DiffRhythmException):
    """Raised when models are not loaded"""
    def __init__(self, message: str = "Models not loaded"):
        super().__init__(message, status_code=503)


class InvalidRequestError(DiffRhythmException):
    """Raised when request validation fails"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, status_code=400, details=details)


class GenerationError(DiffRhythmException):
    """Raised when generation fails"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, status_code=500, details=details)


class JobNotFoundError(DiffRhythmException):
    """Raised when job is not found"""
    def __init__(self, job_id: str):
        super().__init__(f"Job {job_id} not found", status_code=404)


class RateLimitError(DiffRhythmException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)
