#!/usr/bin/env python3
"""
Deep investigation into what's actually wrong with the setup
"""
import os
import sys
import time
import torch
import psutil
import threading
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

def check_process_activity():
    """Check if there are any hung processes"""
    print("="*60)
    print("CHECKING FOR HUNG PROCESSES")
    print("="*60)
    
    # Look for Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if python_processes:
        print("Found Python processes:")
        for proc in python_processes:
            print(f"  PID: {proc['pid']}")
            print(f"  Command: {' '.join(proc['cmdline'][:3]) if proc['cmdline'] else 'N/A'}")
            print(f"  CPU: {proc['cpu_percent']}%")
            print(f"  Memory: {proc['memory_info'].rss / 1024 / 1024:.1f} MB")
            print()
    else:
        print("No Python processes found")

def check_pytorch_performance():
    """Test basic PyTorch performance"""
    print("="*60)
    print("TESTING PYTORCH PERFORMANCE")
    print("="*60)
    
    # Test basic tensor operations
    print("Testing basic tensor operations...")
    start = time.time()
    
    # Create tensors
    a = torch.randn(1000, 1000)
    b = torch.randn(1000, 1000)
    
    # Matrix multiplication
    c = torch.mm(a, b)
    
    basic_time = time.time() - start
    print(f"  Basic operations: {basic_time:.3f}s")
    
    if basic_time > 1.0:
        print("  ⚠ WARNING: Basic PyTorch operations are very slow!")
    else:
        print("  ✓ Basic PyTorch operations are normal")
    
    # Test model loading speed
    print("\nTesting model creation...")
    start = time.time()
    
    try:
        # Simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100)
        )
        
        # Forward pass
        x = torch.randn(10, 1000)
        y = model(x)
        
        model_time = time.time() - start
        print(f"  Model operations: {model_time:.3f}s")
        
        if model_time > 5.0:
            print("  ⚠ WARNING: Model operations are very slow!")
        else:
            print("  ✓ Model operations are normal")
            
    except Exception as e:
        print(f"  ✗ ERROR: Model test failed: {e}")

def test_model_loading_components():
    """Test individual model loading components"""
    print("="*60)
    print("TESTING MODEL LOADING COMPONENTS")
    print("="*60)
    
    # Test 1: HuggingFace Hub download
    print("1. Testing HuggingFace Hub access...")
    start = time.time()
    try:
        from huggingface_hub import hf_hub_download
        
        # This should be instant if cached
        path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="./pretrained"
        )
        
        hub_time = time.time() - start
        print(f"  ✓ Hub access: {hub_time:.3f}s")
        
        if hub_time > 10.0:
            print("  ⚠ WARNING: Hub access is slow - network issue?")
        
        # Check file size
        file_size = os.path.getsize(path)
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
    except Exception as e:
        print(f"  ✗ ERROR: Hub access failed: {e}")
    
    # Test 2: JSON config loading
    print("\n2. Testing config loading...")
    start = time.time()
    try:
        import json
        with open("./config/diffrhythm-1b.json") as f:
            config = json.load(f)
        
        config_time = time.time() - start
        print(f"  ✓ Config loading: {config_time:.3f}s")
        
        if config_time > 1.0:
            print("  ⚠ WARNING: Config loading is slow")
            
    except Exception as e:
        print(f"  ✗ ERROR: Config loading failed: {e}")
    
    # Test 3: Model class import
    print("\n3. Testing model imports...")
    start = time.time()
    try:
        from model import DiT, CFM
        
        import_time = time.time() - start
        print(f"  ✓ Model imports: {import_time:.3f}s")
        
        if import_time > 5.0:
            print("  ⚠ WARNING: Model imports are slow")
            
    except Exception as e:
        print(f"  ✗ ERROR: Model imports failed: {e}")
    
    # Test 4: Model instantiation (without weights)
    print("\n4. Testing model instantiation...")
    start = time.time()
    try:
        from model import DiT, CFM
        import json
        
        with open("./config/diffrhythm-1b.json") as f:
            model_config = json.load(f)
        
        # Create model without loading weights
        dit_model = DiT(**model_config["model"], max_frames=2048)
        cfm = CFM(
            transformer=dit_model,
            num_channels=model_config["model"]["mel_dim"],
            max_frames=2048
        )
        
        instantiation_time = time.time() - start
        print(f"  ✓ Model instantiation: {instantiation_time:.3f}s")
        
        if instantiation_time > 30.0:
            print("  ⚠ WARNING: Model instantiation is very slow!")
        elif instantiation_time > 10.0:
            print("  ⚠ Model instantiation is slow")
        else:
            print("  ✓ Model instantiation is normal")
            
    except Exception as e:
        print(f"  ✗ ERROR: Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()

def test_checkpoint_loading():
    """Test the actual checkpoint loading that's hanging"""
    print("="*60)
    print("TESTING CHECKPOINT LOADING (THE SUSPECTED CULPRIT)")
    print("="*60)
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Get checkpoint path
        dit_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="./pretrained"
        )
        
        print(f"Checkpoint path: {dit_ckpt_path}")
        
        # Test loading checkpoint file
        print("Loading checkpoint file...")
        start = time.time()
        
        checkpoint = torch.load(dit_ckpt_path, weights_only=True, map_location='cpu')
        
        load_time = time.time() - start
        print(f"  ✓ Checkpoint loaded in {load_time:.1f}s")
        
        if load_time > 300:  # 5 minutes
            print("  ✗ CRITICAL: Checkpoint loading is extremely slow!")
            print("  This is likely the root cause of the hang!")
        elif load_time > 120:  # 2 minutes
            print("  ⚠ WARNING: Checkpoint loading is very slow")
        else:
            print("  ✓ Checkpoint loading time is reasonable")
        
        # Check checkpoint contents
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Model state dict has {len(state_dict)} parameters")
            
            # Test applying to model
            print("Testing state dict application...")
            start = time.time()
            
            from model import DiT, CFM
            import json
            
            with open("./config/diffrhythm-1b.json") as f:
                model_config = json.load(f)
            
            dit_model = DiT(**model_config["model"], max_frames=2048)
            cfm = CFM(
                transformer=dit_model,
                num_channels=model_config["model"]["mel_dim"],
                max_frames=2048
            )
            
            # This is where it might hang
            cfm.load_state_dict(state_dict, strict=False)
            
            apply_time = time.time() - start
            print(f"  ✓ State dict applied in {apply_time:.1f}s")
            
            if apply_time > 300:
                print("  ✗ CRITICAL: State dict application is extremely slow!")
            elif apply_time > 60:
                print("  ⚠ WARNING: State dict application is slow")
            else:
                print("  ✓ State dict application time is reasonable")
        
    except Exception as e:
        print(f"  ✗ ERROR: Checkpoint loading test failed: {e}")
        import traceback
        traceback.print_exc()

def check_system_resources():
    """Check system resources and potential bottlenecks"""
    print("="*60)
    print("SYSTEM RESOURCE ANALYSIS")
    print("="*60)
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"Memory:")
    print(f"  Total: {memory.total / 1024**3:.1f} GB")
    print(f"  Available: {memory.available / 1024**3:.1f} GB")
    print(f"  Used: {memory.percent:.1f}%")
    
    if memory.percent > 90:
        print("  ⚠ WARNING: Very high memory usage!")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"\nCPU:")
    print(f"  Usage: {cpu_percent:.1f}%")
    print(f"  Cores: {psutil.cpu_count()}")
    
    # Disk I/O
    disk_io = psutil.disk_io_counters()
    if disk_io:
        print(f"\nDisk I/O:")
        print(f"  Read: {disk_io.read_bytes / 1024**2:.1f} MB")
        print(f"  Write: {disk_io.write_bytes / 1024**2:.1f} MB")
    
    # Check if we're in WSL
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            version = f.read()
            if 'microsoft' in version.lower() or 'wsl' in version.lower():
                print(f"\n⚠ Running in WSL - this could cause performance issues")
                print(f"  Consider running directly in Linux or Windows")

def main():
    """Run deep investigation"""
    print("DiffRhythm Deep Investigation")
    print("="*40)
    print("Something is fundamentally wrong - let's find it!")
    print()
    
    try:
        check_process_activity()
        check_system_resources()
        check_pytorch_performance()
        test_model_loading_components()
        test_checkpoint_loading()
        
        print("\n" + "="*60)
        print("INVESTIGATION COMPLETE")
        print("="*60)
        print("Look for:")
        print("1. Extremely slow checkpoint loading (>5 minutes)")
        print("2. High memory/CPU usage from hung processes")
        print("3. WSL performance issues")
        print("4. PyTorch configuration problems")
        print("5. Network/disk I/O bottlenecks")
        
    except Exception as e:
        print(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()