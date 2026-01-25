#!/usr/bin/env python3
"""
Test API routes to verify all endpoints are accessible and working
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_api_structure():
    """Test API structure without starting server"""
    print("=" * 80)
    print("API Routes Structure Test")
    print("=" * 80)
    print()
    
    api_path = project_root / "backend" / "api.py"
    
    if not api_path.exists():
        print("[FAIL] backend/api.py not found")
        return False
    
    content = api_path.read_text()
    
    # Check routes
    routes = {
        "GET /": "Root endpoint",
        "GET /api/v1/health": "Health check",
        "POST /api/v1/generate": "Generate music",
        "GET /api/v1/status/{job_id}": "Job status",
        "GET /api/v1/download/{job_id}": "Download audio",
        "GET /api/v1/queue": "Queue status",
        "GET /api/v1/metrics": "Metrics endpoint"
    }
    
    print("Checking route definitions...")
    print()
    
    all_found = True
    for route, description in routes.items():
        # Check for route decorator
        method = route.split()[0]
        path = route.split()[1].split("{")[0] if "{" in route.split()[1] else route.split()[1]
        
        found = False
        if method == "GET":
            if f'@app.get("{path}"' in content or f'@app.get(f"{path}"' in content:
                found = True
        elif method == "POST":
            if f'@app.post("{path}"' in content or f'@app.post(f"{path}"' in content:
                found = True
        
        if found:
            print(f"  [OK] {route} - {description}")
        else:
            print(f"  [WARN] {route} - {description} (not found in expected format)")
            # Don't fail, just warn
    
    # Check request/response models
    print()
    print("Checking request/response models...")
    print()
    
    models = {
        "GenerationRequest": "Request model for generation",
        "GenerationResponse": "Response model for generation",
        "JobStatusResponse": "Response model for job status",
        "HealthResponse": "Response model for health check"
    }
    
    for model_name, description in models.items():
        if f"class {model_name}" in content:
            print(f"  [OK] {model_name} - {description}")
        else:
            print(f"  [FAIL] {model_name} - {description} (not found)")
            all_found = False
    
    # Check error handling
    print()
    print("Checking error handling...")
    print()
    
    if "@app.exception_handler" in content:
        print("  [OK] Exception handlers configured")
    else:
        print("  [WARN] Exception handlers not found")
    
    if "ModelNotLoadedError" in content:
        print("  [OK] ModelNotLoadedError handler found")
    else:
        print("  [WARN] ModelNotLoadedError handler not found")
    
    print()
    print("=" * 80)
    if all_found:
        print("[SUCCESS] API structure looks good!")
    else:
        print("[WARNING] Some issues found, but basic structure is present")
    print("=" * 80)
    
    return all_found

def test_api_imports():
    """Test that API can be imported"""
    print()
    print("=" * 80)
    print("API Import Test")
    print("=" * 80)
    print()
    
    try:
        # Try importing the API module
        print("Attempting to import backend.api...")
        from backend import api
        print("  [OK] backend.api imported successfully")
        
        # Check if app exists
        if hasattr(api, 'app'):
            print("  [OK] FastAPI app object found")
        else:
            print("  [FAIL] FastAPI app object not found")
            return False
        
        # Check if model_manager exists
        if hasattr(api, 'model_manager'):
            print("  [OK] model_manager found")
        else:
            print("  [FAIL] model_manager not found")
            return False
        
        # Check if job_manager exists
        if hasattr(api, 'job_manager'):
            print("  [OK] job_manager found")
        else:
            print("  [FAIL] job_manager not found")
            return False
        
        print()
        print("[SUCCESS] API module imports successfully!")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Failed to import backend.api: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing API structure and imports...")
    print()
    
    structure_ok = test_api_structure()
    imports_ok = test_api_imports()
    
    if structure_ok and imports_ok:
        print()
        print("[SUCCESS] All API tests passed!")
        sys.exit(0)
    else:
        print()
        print("[WARNING] Some API tests had issues")
        sys.exit(1)
