"""
Configuration management for DiffRhythm backend
"""
import os
from typing import Optional
from pathlib import Path


class Config:
    """Application configuration"""
    
    # API Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Device Configuration
    DEVICE: str = os.getenv("DEVICE", "cpu")  # cpu or cuda
    MODEL_MAX_FRAMES: int = int(os.getenv("MODEL_MAX_FRAMES", "2048"))
    
    # Model Paths
    MODEL_CACHE_DIR: str = os.getenv(
        "MODEL_CACHE_DIR",
        os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME") or "./pretrained"
    )
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # Job Queue Configuration
    USE_REDIS: bool = os.getenv("USE_REDIS", "false").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Generation Settings
    DEFAULT_AUDIO_LENGTH: int = int(os.getenv("DEFAULT_AUDIO_LENGTH", "95"))
    MAX_AUDIO_LENGTH: int = int(os.getenv("MAX_AUDIO_LENGTH", "285"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "4"))
    GENERATION_TIMEOUT: int = int(os.getenv("GENERATION_TIMEOUT", "2400"))  # 40 minutes
    
    # CPU Optimization (High Quality Defaults)
    # Defaults set to "high" quality preset for production
    CPU_STEPS: int = int(os.getenv("CPU_STEPS", "32"))  # High quality: 32 steps
    CPU_CFG_STRENGTH: float = float(os.getenv("CPU_CFG_STRENGTH", "4.0"))  # High quality: 4.0 CFG
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # json or text
    
    # Security
    ENABLE_RATE_LIMIT: bool = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "10"))
    API_KEY: Optional[str] = os.getenv("API_KEY", None)
    CORS_ORIGINS: str | list = os.getenv("CORS_ORIGINS", "*")
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PATH: str = os.getenv("METRICS_PATH", "/api/v1/metrics")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if cls.DEVICE not in ["cpu", "cuda"]:
            errors.append(f"Invalid DEVICE: {cls.DEVICE}. Must be 'cpu' or 'cuda'")
        
        if cls.MODEL_MAX_FRAMES not in [2048, 6144]:
            errors.append(f"Invalid MODEL_MAX_FRAMES: {cls.MODEL_MAX_FRAMES}. Must be 2048 or 6144")
        
        if cls.DEFAULT_AUDIO_LENGTH < 95 or cls.DEFAULT_AUDIO_LENGTH > cls.MAX_AUDIO_LENGTH:
            errors.append(f"Invalid DEFAULT_AUDIO_LENGTH: {cls.DEFAULT_AUDIO_LENGTH}")
        
        if cls.MAX_BATCH_SIZE < 1 or cls.MAX_BATCH_SIZE > 8:
            errors.append(f"Invalid MAX_BATCH_SIZE: {cls.MAX_BATCH_SIZE}. Must be 1-8")
        
        return errors
