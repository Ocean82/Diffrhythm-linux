#!/usr/bin/env python3
"""
Install missing dependencies for DiffRhythm on Windows
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"Running: {' '.join(cmd)}")
    if description:
        print(f"Purpose: {description}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Success")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False

def install_pip_package(package, description=""):
    """Install a pip package"""
    print(f"\n--- Installing {package} ---")
    if description:
        print(f"Purpose: {description}")
    
    return run_command([sys.executable, "-m", "pip", "install", package])

def download_and_install_espeak():
    """Download and install espeak-ng for Windows"""
    print("\n--- Installing eSpeak-NG for Windows ---")
    print("Purpose: Text-to-phoneme conversion for Chinese/English")
    
    # Check if espeak is already available
    if shutil.which("espeak") or shutil.which("espeak-ng"):
        print("✓ eSpeak already installed")
        return True
    
    print("Downloading eSpeak-NG installer...")
    
    # Download URL for latest eSpeak-NG Windows installer
    download_url = "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi"
    installer_path = "espeak-ng-installer.msi"
    
    try:
        urllib.request.urlretrieve(download_url, installer_path)
        print(f"✓ Downloaded: {installer_path}")
        
        print("Installing eSpeak-NG...")
        print("⚠ This will open an installer window - please follow the prompts")
        
        # Run the MSI installer
        result = subprocess.run(["msiexec", "/i", installer_path, "/quiet"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ eSpeak-NG installed successfully")
            
            # Add to PATH if needed
            espeak_path = r"C:\Program Files\eSpeak NG"
            if os.path.exists(espeak_path):
                current_path = os.environ.get("PATH", "")
                if espeak_path not in current_path:
                    print(f"⚠ Please add {espeak_path} to your PATH environment variable")
                    print("  Or restart your terminal/IDE after installation")
            
            # Clean up
            os.remove(installer_path)
            return True
        else:
            print(f"✗ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading/installing eSpeak: {e}")
        return False

def install_alternative_espeak():
    """Install alternative espeak packages for Windows"""
    print("\n--- Installing Alternative eSpeak Packages ---")
    
    packages = [
        ("espeak-phonemizer-windows", "Windows-specific eSpeak wrapper"),
        ("py-espeak-ng", "Python eSpeak-NG bindings"),
        ("espeakng", "Alternative eSpeak-NG package"),
    ]
    
    success = False
    for package, description in packages:
        if install_pip_package(package, description):
            success = True
            break
    
    return success

def check_current_setup():
    """Check what's currently installed"""
    print("="*60)
    print("CHECKING CURRENT SETUP")
    print("="*60)
    
    # Check Python packages
    packages_to_check = [
        "torch", "torchaudio", "einops", "phonemizer", 
        "jieba", "onnxruntime", "transformers"
    ]
    
    installed = []
    missing = []
    
    for package in packages_to_check:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    print("Installed packages:")
    for pkg in installed:
        print(f"  ✓ {pkg}")
    
    print("Missing packages:")
    for pkg in missing:
        print(f"  ✗ {pkg}")
    
    # Check system commands
    commands_to_check = ["espeak", "espeak-ng"]
    
    print("\nSystem commands:")
    for cmd in commands_to_check:
        if shutil.which(cmd):
            print(f"  ✓ {cmd} available")
        else:
            print(f"  ✗ {cmd} not found")
    
    return len(missing) == 0

def install_missing_packages():
    """Install missing Python packages"""
    print("\n" + "="*60)
    print("INSTALLING MISSING PACKAGES")
    print("="*60)
    
    # Core packages that should be installed
    packages = [
        ("phonemizer", "Text-to-phoneme conversion"),
        ("jieba", "Chinese text segmentation"),
        ("onnxruntime", "ONNX model runtime"),
        ("transformers", "Hugging Face transformers"),
        ("librosa", "Audio processing"),
        ("soundfile", "Audio file I/O"),
    ]
    
    success_count = 0
    for package, description in packages:
        if install_pip_package(package, description):
            success_count += 1
    
    return success_count > 0

def main():
    """Main installation function"""
    print("DiffRhythm Dependency Installer for Windows")
    print("This will install missing dependencies for song generation")
    print()
    
    # Check current setup
    if check_current_setup():
        print("\n✓ All dependencies appear to be installed!")
        print("You can try running the generation test now.")
        return
    
    print("\nSome dependencies are missing. Installing...")
    
    # Install Python packages
    packages_ok = install_missing_packages()
    
    # Install eSpeak
    print("\n" + "="*60)
    print("INSTALLING ESPEAK")
    print("="*60)
    
    espeak_ok = False
    
    # Try downloading and installing eSpeak-NG
    if download_and_install_espeak():
        espeak_ok = True
    else:
        print("Direct installation failed, trying alternative packages...")
        espeak_ok = install_alternative_espeak()
    
    # Final check
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    
    if packages_ok and espeak_ok:
        print("🎉 INSTALLATION SUCCESSFUL!")
        print()
        print("✓ Python packages installed")
        print("✓ eSpeak installed")
        print()
        print("Next steps:")
        print("1. Restart your terminal/IDE")
        print("2. Run: python test_song_generation.py")
        
    elif packages_ok:
        print("⚠ PARTIAL SUCCESS")
        print("✓ Python packages installed")
        print("✗ eSpeak installation failed")
        print()
        print("Manual eSpeak installation:")
        print("1. Download from: https://github.com/espeak-ng/espeak-ng/releases")
        print("2. Install the .msi file")
        print("3. Add installation directory to PATH")
        
    else:
        print("✗ INSTALLATION FAILED")
        print("Please check the errors above and try manual installation")
        print()
        print("Manual installation steps:")
        print("1. pip install phonemizer jieba onnxruntime transformers")
        print("2. Install eSpeak from GitHub releases")

if __name__ == "__main__":
    main()