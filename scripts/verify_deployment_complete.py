#!/usr/bin/env python3
"""
Comprehensive deployment verification script for DiffRhythm-LINUX
Checks Docker configuration, model loading, API routes, and inference pipeline
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class VerificationResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def add_info(self, msg: str):
        self.info.append(msg)
    
    def set_passed(self, passed: bool):
        self.passed = passed
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }

def verify_dockerfile(result: VerificationResult):
    """Verify Dockerfile.prod configuration"""
    dockerfile_path = project_root / "Dockerfile.prod"
    
    if not dockerfile_path.exists():
        result.add_error("Dockerfile.prod not found")
        return
    
    result.add_info("Dockerfile.prod exists")
    
    content = dockerfile_path.read_text()
    
    # Check entry point
    if "uvicorn backend.api:app" in content or 'uvicorn backend.api:app' in content:
        result.add_info("Entry point correctly set to 'uvicorn backend.api:app'")
    elif "uvicorn" in content and "backend.api" in content:
        result.add_info("Entry point correctly set to 'uvicorn backend.api:app'")
    else:
        result.add_error("Entry point not set correctly - expected 'uvicorn backend.api:app'")
    
    # Check health check
    if "HEALTHCHECK" in content:
        result.add_info("Health check configured")
    else:
        result.add_warning("Health check not configured")
    
    # Check non-root user
    if "USER appuser" in content:
        result.add_info("Non-root user configured")
    else:
        result.add_warning("Non-root user not configured")
    
    # Check multi-stage build
    if "FROM" in content and content.count("FROM") >= 2:
        result.add_info("Multi-stage build detected")
    else:
        result.add_warning("Single-stage build (not optimized)")
    
    result.set_passed(len(result.errors) == 0)

def verify_docker_compose(result: VerificationResult):
    """Verify docker-compose.prod.yml configuration"""
    compose_path = project_root / "docker-compose.prod.yml"
    
    if not compose_path.exists():
        result.add_error("docker-compose.prod.yml not found")
        return
    
    result.add_info("docker-compose.prod.yml exists")
    
    content = compose_path.read_text()
    
    # Check service name
    if "diffrhythm-api" in content:
        result.add_info("Service name 'diffrhythm-api' found")
    else:
        result.add_warning("Service name not found")
    
    # Check port mapping
    if "8000:8000" in content or "${PORT:-8000}:8000" in content:
        result.add_info("Port 8000 mapped correctly")
    else:
        result.add_error("Port mapping not found")
    
    # Check volumes
    if "/app/output" in content and "/app/pretrained" in content:
        result.add_info("Required volumes configured")
    else:
        result.add_warning("Some volumes may be missing")
    
    # Check health check
    if "healthcheck" in content.lower():
        result.add_info("Health check configured in compose")
    else:
        result.add_warning("Health check not in compose file")
    
    result.set_passed(len(result.errors) == 0)

def verify_backend_api(result: VerificationResult):
    """Verify backend/api.py structure"""
    api_path = project_root / "backend" / "api.py"
    
    if not api_path.exists():
        result.add_error("backend/api.py not found")
        return
    
    result.add_info("backend/api.py exists")
    
    content = api_path.read_text()
    
    # Check FastAPI app
    if "FastAPI" in content:
        result.add_info("FastAPI application found")
    else:
        result.add_error("FastAPI not found")
    
    # Check ModelManager
    if "class ModelManager" in content:
        result.add_info("ModelManager class found")
    else:
        result.add_error("ModelManager class not found")
    
    # Check JobManager
    if "class JobManager" in content:
        result.add_info("JobManager class found")
    else:
        result.add_error("JobManager class not found")
    
    # Check lifespan
    if "lifespan" in content:
        result.add_info("Lifespan manager found")
    else:
        result.add_warning("Lifespan manager not found")
    
    # Check routes
    routes = [
        ("/api/v1/health", "health"),
        ("/api/v1/generate", "generate"),
        ("/api/v1/status", "status"),
        ("/api/v1/download", "download"),
        ("/api/v1/queue", "queue")
    ]
    
    for route, keyword in routes:
        if route in content or keyword in content.lower():
            result.add_info(f"Route {route} found")
        else:
            result.add_warning(f"Route {route} not found")
    
    result.set_passed(len(result.errors) == 0)

def verify_model_loading(result: VerificationResult):
    """Verify model loading configuration"""
    infer_utils_path = project_root / "infer" / "infer_utils.py"
    
    if not infer_utils_path.exists():
        result.add_error("infer/infer_utils.py not found")
        return
    
    result.add_info("infer/infer_utils.py exists")
    
    content = infer_utils_path.read_text()
    
    # Check prepare_model function
    if "def prepare_model" in content:
        result.add_info("prepare_model function found")
    else:
        result.add_error("prepare_model function not found")
    
    # Check model repositories
    repos = [
        "ASLP-lab/DiffRhythm-1_2",
        "ASLP-lab/DiffRhythm-vae",
        "OpenMuQ/MuQ-MuLan-large"
    ]
    
    for repo in repos:
        if repo in content:
            result.add_info(f"Model repository {repo} referenced")
        else:
            result.add_warning(f"Model repository {repo} not found")
    
    # Check cache directory
    if "DEFAULT_CACHE_DIR" in content or "MODEL_CACHE_DIR" in content:
        result.add_info("Model cache directory configured")
    else:
        result.add_warning("Model cache directory not explicitly configured")
    
    result.set_passed(len(result.errors) == 0)

def verify_requirements(result: VerificationResult):
    """Verify requirements.txt files"""
    req_path = project_root / "requirements.txt"
    backend_req_path = project_root / "backend" / "requirements.txt"
    
    if not req_path.exists():
        result.add_error("requirements.txt not found")
        return
    
    result.add_info("requirements.txt exists")
    
    # Check critical dependencies
    content = req_path.read_text()
    critical_deps = [
        "fastapi",
        "uvicorn",
        "torch",
        "torchaudio",
        "transformers",
        "librosa"
    ]
    
    for dep in critical_deps:
        if dep.lower() in content.lower():
            result.add_info(f"Critical dependency {dep} found")
        else:
            result.add_warning(f"Critical dependency {dep} not found")
    
    if backend_req_path.exists():
        result.add_info("backend/requirements.txt exists")
        backend_content = backend_req_path.read_text()
        if "prometheus-client" in backend_content:
            result.add_info("Backend monitoring dependencies found")
    else:
        result.add_warning("backend/requirements.txt not found")
    
    result.set_passed(len(result.errors) == 0)

def verify_config(result: VerificationResult):
    """Verify backend configuration"""
    config_path = project_root / "backend" / "config.py"
    
    if not config_path.exists():
        result.add_error("backend/config.py not found")
        return
    
    result.add_info("backend/config.py exists")
    
    content = config_path.read_text()
    
    # Check Config class
    if "class Config" in content:
        result.add_info("Config class found")
    else:
        result.add_error("Config class not found")
    
    # Check critical settings
    settings = [
        "DEVICE",
        "MODEL_CACHE_DIR",
        "OUTPUT_DIR",
        "API_PREFIX"
    ]
    
    for setting in settings:
        if setting in content:
            result.add_info(f"Configuration {setting} found")
        else:
            result.add_warning(f"Configuration {setting} not found")
    
    result.set_passed(len(result.errors) == 0)

def verify_inference_pipeline(result: VerificationResult):
    """Verify inference pipeline components"""
    infer_path = project_root / "infer" / "infer.py"
    
    if not infer_path.exists():
        result.add_error("infer/infer.py not found")
        return
    
    result.add_info("infer/infer.py exists")
    
    content = infer_path.read_text()
    
    # Check inference function
    if "def inference" in content:
        result.add_info("inference function found")
    else:
        result.add_error("inference function not found")
    
    # Check save_audio_robust
    if "save_audio_robust" in content:
        result.add_info("save_audio_robust function found")
    else:
        result.add_warning("save_audio_robust function not found")
    
    # Check safe_normalize_audio
    if "safe_normalize_audio" in content:
        result.add_info("safe_normalize_audio function found")
    else:
        result.add_warning("safe_normalize_audio function not found")
    
    result.set_passed(len(result.errors) == 0)

def verify_security(result: VerificationResult):
    """Verify security configuration"""
    security_path = project_root / "backend" / "security.py"
    
    if not security_path.exists():
        result.add_error("backend/security.py not found")
        return
    
    result.add_info("backend/security.py exists")
    
    content = security_path.read_text()
    
    # Check CORS
    if "CORS" in content or "cors" in content.lower():
        result.add_info("CORS configuration found")
    else:
        result.add_warning("CORS configuration not found")
    
    # Check rate limiting
    if "limiter" in content.lower() or "rate" in content.lower():
        result.add_info("Rate limiting found")
    else:
        result.add_warning("Rate limiting not found")
    
    result.set_passed(len(result.errors) == 0)

def main():
    """Run all verification checks"""
    print("=" * 80)
    print("DiffRhythm-LINUX Deployment Verification")
    print("=" * 80)
    print()
    
    results: List[VerificationResult] = []
    
    # Phase 1: Docker Configuration
    print("Phase 1: Docker Configuration")
    print("-" * 80)
    
    result = VerificationResult("Dockerfile.prod")
    verify_dockerfile(result)
    results.append(result)
    print_result(result)
    
    result = VerificationResult("docker-compose.prod.yml")
    verify_docker_compose(result)
    results.append(result)
    print_result(result)
    
    # Phase 2: Backend API
    print("\nPhase 2: Backend API")
    print("-" * 80)
    
    result = VerificationResult("backend/api.py")
    verify_backend_api(result)
    results.append(result)
    print_result(result)
    
    result = VerificationResult("backend/config.py")
    verify_config(result)
    results.append(result)
    print_result(result)
    
    # Phase 3: Model Loading
    print("\nPhase 3: Model Loading")
    print("-" * 80)
    
    result = VerificationResult("Model Loading Configuration")
    verify_model_loading(result)
    results.append(result)
    print_result(result)
    
    # Phase 4: Dependencies
    print("\nPhase 4: Dependencies")
    print("-" * 80)
    
    result = VerificationResult("requirements.txt")
    verify_requirements(result)
    results.append(result)
    print_result(result)
    
    # Phase 5: Inference Pipeline
    print("\nPhase 5: Inference Pipeline")
    print("-" * 80)
    
    result = VerificationResult("Inference Pipeline")
    verify_inference_pipeline(result)
    results.append(result)
    print_result(result)
    
    # Phase 6: Security
    print("\nPhase 6: Security")
    print("-" * 80)
    
    result = VerificationResult("Security Configuration")
    verify_security(result)
    results.append(result)
    print_result(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    print(f"Total Checks: {total}")
    print(f"Passed: {passed} [OK]")
    print(f"Failed: {failed} [FAIL]")
    print()
    
    if failed > 0:
        print("Failed Checks:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")
                for error in r.errors:
                    print(f"    ERROR: {error}")
        print()
    
    # Save results to JSON
    output_path = project_root / "output" / "verification_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"Detailed results saved to: {output_path}")
    print()
    
    if failed == 0:
        print("[SUCCESS] All checks passed! Deployment configuration looks good.")
        return 0
    else:
        print("[FAILURE] Some checks failed. Please review the errors above.")
        return 1

def print_result(result: VerificationResult):
    """Print verification result"""
    status = "[PASS]" if result.passed else "[FAIL]"
    print(f"{status}: {result.name}")
    
    if result.info:
        for info in result.info:
            print(f"  [INFO] {info}")
    
    if result.warnings:
        for warning in result.warnings:
            print(f"  [WARN] {warning}")
    
    if result.errors:
        for error in result.errors:
            print(f"  [ERROR] {error}")

if __name__ == "__main__":
    sys.exit(main())
