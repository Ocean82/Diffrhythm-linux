"""
Metrics collection for DiffRhythm backend
"""
import time
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from backend.config import Config


# Prometheus metrics
request_count = Counter(
    "diffrhythm_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "diffrhythm_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"]
)

generation_count = Counter(
    "diffrhythm_generations_total",
    "Total number of generations",
    ["status"]
)

generation_duration = Histogram(
    "diffrhythm_generation_duration_seconds",
    "Generation duration in seconds"
)

queue_length = Gauge(
    "diffrhythm_queue_length",
    "Current job queue length"
)

active_jobs = Gauge(
    "diffrhythm_active_jobs",
    "Number of currently active generation jobs"
)

model_loaded = Gauge(
    "diffrhythm_models_loaded",
    "Whether models are loaded (1) or not (0)"
)

error_count = Counter(
    "diffrhythm_errors_total",
    "Total number of errors",
    ["error_type"]
)


class MetricsCollector:
    """Metrics collection and aggregation"""
    
    def __init__(self):
        self._lock = Lock()
        self._queue_length = 0
        self._active_jobs = 0
        self._models_loaded = 0
        
    def increment_request(self, method: str, endpoint: str, status: int):
        """Increment request counter"""
        request_count.labels(method=method, endpoint=endpoint, status=status).inc()
    
    def record_request_duration(self, method: str, endpoint: str, duration: float):
        """Record request duration"""
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def increment_generation(self, status: str):
        """Increment generation counter"""
        generation_count.labels(status=status).inc()
    
    def record_generation_duration(self, duration: float):
        """Record generation duration"""
        generation_duration.observe(duration)
    
    def set_queue_length(self, length: int):
        """Update queue length"""
        with self._lock:
            self._queue_length = length
            queue_length.set(length)
    
    def set_active_jobs(self, count: int):
        """Update active jobs count"""
        with self._lock:
            self._active_jobs = count
            active_jobs.set(count)
    
    def set_models_loaded(self, loaded: bool):
        """Update models loaded status"""
        with self._lock:
            self._models_loaded = 1 if loaded else 0
            model_loaded.set(self._models_loaded)
    
    def increment_error(self, error_type: str):
        """Increment error counter"""
        error_count.labels(error_type=error_type).inc()
    
    def get_metrics_response(self) -> Response:
        """Get Prometheus metrics response"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# Global metrics collector instance
metrics = MetricsCollector()
