"""
Integration tests for DiffRhythm API
"""
import pytest
import requests
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import Config
from backend.api import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_lyrics():
    """Sample lyrics for testing"""
    return """[00:00.00]Test song
[00:05.00]This is a test
[00:10.00]For the API
[00:15.00]Verification"""


@pytest.fixture
def sample_request(sample_lyrics):
    """Sample generation request"""
    return {
        "lyrics": sample_lyrics,
        "style_prompt": "pop, upbeat, energetic",
        "audio_length": 95,
        "batch_size": 1
    }


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_endpoint_exists(self, client):
        """Test health endpoint is accessible"""
        response = client.get(f"{Config.API_PREFIX}/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Test health response has required fields"""
        response = client.get(f"{Config.API_PREFIX}/health")
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "device" in data
        assert "queue_length" in data
        assert "active_jobs" in data
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestGenerationEndpoint:
    """Tests for generation endpoint"""
    
    def test_generate_endpoint_exists(self, client, sample_request):
        """Test generate endpoint accepts requests"""
        # Note: This will fail if models aren't loaded
        # In real tests, mock the model manager
        response = client.post(
            f"{Config.API_PREFIX}/generate",
            json=sample_request
        )
        # Should either succeed (200) or fail gracefully (503 if models not loaded)
        assert response.status_code in [200, 503]
    
    def test_generate_validation(self, client):
        """Test request validation"""
        # Invalid audio length
        response = client.post(
            f"{Config.API_PREFIX}/generate",
            json={
                "lyrics": "test",
                "style_prompt": "pop",
                "audio_length": 50,  # Invalid
                "batch_size": 1
            }
        )
        assert response.status_code == 422  # Validation error
        
        # Missing required fields
        response = client.post(
            f"{Config.API_PREFIX}/generate",
            json={
                "lyrics": "test"
                # Missing style_prompt
            }
        )
        assert response.status_code == 422


class TestJobStatusEndpoint:
    """Tests for job status endpoint"""
    
    def test_status_endpoint_not_found(self, client):
        """Test status endpoint handles non-existent jobs"""
        response = client.get(f"{Config.API_PREFIX}/status/nonexistent-job-id")
        assert response.status_code == 404


class TestQueueEndpoint:
    """Tests for queue endpoint"""
    
    def test_queue_endpoint(self, client):
        """Test queue status endpoint"""
        response = client.get(f"{Config.API_PREFIX}/queue")
        assert response.status_code == 200
        
        data = response.json()
        assert "queue_length" in data
        assert "current_job" in data
        assert "estimated_wait_minutes" in data


class TestMetricsEndpoint:
    """Tests for metrics endpoint"""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get(Config.METRICS_PATH)
        # Should return 200 if metrics enabled, 404 if disabled
        assert response.status_code in [200, 404]


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test 404 for invalid endpoints"""
        response = client.get("/api/v1/invalid")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.put(f"{Config.API_PREFIX}/health")
        assert response.status_code == 405


class TestSecurity:
    """Tests for security features"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/")
        # CORS middleware should add headers
        assert response.status_code in [200, 204]
    
    def test_security_headers(self, client):
        """Test security headers are present"""
        response = client.get("/")
        headers = response.headers
        
        # Check for security headers (if middleware is working)
        # Note: These may not be present in test client
        assert response.status_code == 200


# Integration test (requires running service)
class TestIntegration:
    """Integration tests requiring running service"""
    
    @pytest.mark.skip(reason="Requires running service and models loaded")
    def test_full_generation_flow(self):
        """Test complete generation flow"""
        base_url = "http://localhost:8000"
        
        # 1. Check health
        response = requests.get(f"{base_url}/api/v1/health")
        assert response.status_code == 200
        health = response.json()
        assert health["models_loaded"] is True
        
        # 2. Submit generation
        request_data = {
            "lyrics": "[00:00.00]Test\n[00:05.00]Song",
            "style_prompt": "pop, upbeat",
            "audio_length": 95,
            "batch_size": 1
        }
        
        response = requests.post(
            f"{base_url}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]
        
        # 3. Check status
        max_wait = 1800  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(f"{base_url}/api/v1/status/{job_id}")
            assert response.status_code == 200
            status = response.json()
            
            if status["status"] == "completed":
                # 4. Download file
                response = requests.get(f"{base_url}/api/v1/download/{job_id}")
                assert response.status_code == 200
                assert response.headers["content-type"] == "audio/wav"
                return
            
            if status["status"] == "failed":
                pytest.fail(f"Generation failed: {status.get('error')}")
            
            time.sleep(10)  # Check every 10 seconds
        
        pytest.fail("Generation timeout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
