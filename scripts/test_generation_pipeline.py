#!/usr/bin/env python3
"""
Test the complete generation pipeline without actually generating audio
This verifies all components are working correctly
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all critical imports"""
    print("=" * 60)
    print("1. Testing Imports")
    print("=" * 60)
    
    errors = []
    
    # Backend imports
    try:
        from backend.config import Config
        print("✓ backend.config")
    except Exception as e:
        print(f"✗ backend.config: {e}")
        errors.append("backend.config")
    
    try:
        from backend.api import app, model_manager, job_manager
        print("✓ backend.api")
    except Exception as e:
        print(f"✗ backend.api: {e}")
        errors.append("backend.api")
    
    # Core inference imports
    try:
        from infer.infer_utils import (
            prepare_model,
            get_lrc_token,
            get_style_prompt,
            get_negative_style_prompt
        )
        print("✓ infer.infer_utils")
    except Exception as e:
        print(f"✗ infer.infer_utils: {e}")
        errors.append("infer.infer_utils")
    
    try:
        from infer.infer import inference, save_audio_robust
        print("✓ infer.infer")
    except Exception as e:
        print(f"✗ infer.infer: {e}")
        errors.append("infer.infer")
    
    # Model imports
    try:
        from model.cfm import CFM
        print("✓ model.cfm")
    except Exception as e:
        print(f"✗ model.cfm: {e}")
        errors.append("model.cfm")
    
    # Optional imports
    try:
        from post_processing.mastering import master_audio_file
        print("✓ post_processing.mastering (optional)")
    except Exception as e:
        print(f"⚠ post_processing.mastering: {e} (optional)")
    
    try:
        from infer.quality_presets import get_preset, QUALITY_PRESETS
        print(f"✓ infer.quality_presets ({len(QUALITY_PRESETS)} presets)")
    except Exception as e:
        print(f"⚠ infer.quality_presets: {e} (optional)")
    
    if errors:
        print(f"\n✗ Import errors: {len(errors)}")
        return False
    else:
        print("\n✓ All critical imports successful")
        return True


def test_configuration():
    """Test configuration loading and validation"""
    print("\n" + "=" * 60)
    print("2. Testing Configuration")
    print("=" * 60)
    
    try:
        from backend.config import Config
        
        print(f"✓ HOST: {Config.HOST}")
        print(f"✓ PORT: {Config.PORT}")
        print(f"✓ API_PREFIX: {Config.API_PREFIX}")
        print(f"✓ DEVICE: {Config.DEVICE}")
        print(f"✓ CPU_STEPS: {Config.CPU_STEPS}")
        print(f"✓ CPU_CFG_STRENGTH: {Config.CPU_CFG_STRENGTH}")
        print(f"✓ OUTPUT_DIR: {Config.OUTPUT_DIR}")
        print(f"✓ MODEL_MAX_FRAMES: {Config.MODEL_MAX_FRAMES}")
        
        # Validate
        errors = Config.validate()
        if errors:
            print(f"\n✗ Configuration errors: {errors}")
            return False
        else:
            print("\n✓ Configuration valid")
            return True
            
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_presets():
    """Test quality preset system"""
    print("\n" + "=" * 60)
    print("3. Testing Quality Presets")
    print("=" * 60)
    
    try:
        from infer.quality_presets import get_preset, QUALITY_PRESETS
        
        print(f"✓ Found {len(QUALITY_PRESETS)} presets")
        
        for name in QUALITY_PRESETS.keys():
            preset = get_preset(name)
            print(f"  - {name}: {preset.steps} steps, {preset.cfg_strength} CFG")
        
        # Test invalid preset
        try:
            get_preset("invalid")
            print("\n✗ Should have raised error for invalid preset")
            return False
        except ValueError:
            print("\n✓ Invalid preset correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Quality presets test failed: {e}")
        return False


def test_api_structure():
    """Test API request/response models"""
    print("\n" + "=" * 60)
    print("4. Testing API Structure")
    print("=" * 60)
    
    try:
        from backend.api import (
            GenerationRequest,
            GenerationResponse,
            JobStatusResponse,
            HealthResponse,
            app
        )
        
        # Test request model
        test_request = GenerationRequest(
            lyrics="[00:00.00]Test lyrics",
            style_prompt="pop, upbeat",
            audio_length=95
        )
        print(f"✓ GenerationRequest model valid")
        print(f"  - lyrics length: {len(test_request.lyrics)}")
        print(f"  - style_prompt: {test_request.style_prompt}")
        print(f"  - audio_length: {test_request.audio_length}")
        
        # Test with quality preset
        test_request_preset = GenerationRequest(
            lyrics="[00:00.00]Test",
            style_prompt="pop",
            audio_length=95,
            preset="high",
            auto_master=True
        )
        print(f"✓ GenerationRequest with preset valid")
        print(f"  - preset: {test_request_preset.preset}")
        print(f"  - auto_master: {test_request_preset.auto_master}")
        
        # Test FastAPI app
        print(f"✓ FastAPI app initialized")
        print(f"  - title: {app.title}")
        print(f"  - version: {app.version}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test required files and directories"""
    print("\n" + "=" * 60)
    print("5. Testing File Structure")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        "backend/api.py",
        "backend/config.py",
        "infer/infer.py",
        "infer/infer_utils.py",
        "model/cfm.py",
    ]
    
    required_dirs = [
        "backend",
        "infer",
        "model",
        "output",
    ]
    
    all_ok = True
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} MISSING")
            all_ok = False
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✓ {dir_path}/ directory exists")
        else:
            print(f"✗ {dir_path}/ MISSING")
            all_ok = False
    
    return all_ok


def test_output_directory():
    """Test output directory setup"""
    print("\n" + "=" * 60)
    print("6. Testing Output Directory")
    print("=" * 60)
    
    try:
        from backend.config import Config
        
        Config.ensure_directories()
        
        output_dir = Config.OUTPUT_DIR
        if output_dir.exists():
            print(f"✓ Output directory exists: {output_dir}")
            
            # Test write permission
            test_file = output_dir / ".test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()
                print(f"✓ Output directory is writable")
            except Exception as e:
                print(f"✗ Output directory not writable: {e}")
                return False
        else:
            print(f"✗ Output directory missing: {output_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Output directory test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DiffRhythm Generation Pipeline Test")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Quality Presets", test_quality_presets()))
    results.append(("API Structure", test_api_structure()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Output Directory", test_output_directory()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All components verified successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
