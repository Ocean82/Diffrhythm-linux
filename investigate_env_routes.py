#!/usr/bin/env python3
"""
Comprehensive investigation of environment variables and route configurations
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

def find_env_files(root_dir: Path) -> List[Path]:
    """Find all .env files in the project"""
    env_files = []
    for path in root_dir.rglob("*.env"):
        if ".git" not in str(path) and "node_modules" not in str(path):
            env_files.append(path)
    return env_files

def find_cursorignore_files(root_dir: Path) -> List[Path]:
    """Find all .cursorignore files"""
    cursorignore_files = []
    for path in root_dir.rglob(".cursorignore"):
        cursorignore_files.append(path)
    return cursorignore_files

def analyze_env_loading() -> Dict[str, any]:
    """Analyze how env variables are loaded"""
    results = {
        "uses_dotenv": False,
        "dotenv_imports": [],
        "os_getenv_usage": [],
        "config_files": []
    }
    
    # Check backend/config.py
    backend_config = Path("backend/config.py")
    if backend_config.exists():
        with open(backend_config) as f:
            content = f.read()
            if "load_dotenv" in content or "from dotenv" in content:
                results["uses_dotenv"] = True
                results["dotenv_imports"].append(str(backend_config))
            if "os.getenv" in content:
                results["os_getenv_usage"].append(str(backend_config))
            results["config_files"].append(str(backend_config))
    
    # Check backend/api.py
    backend_api = Path("backend/api.py")
    if backend_api.exists():
        with open(backend_api) as f:
            content = f.read()
            if "load_dotenv" in content or "from dotenv" in content:
                results["uses_dotenv"] = True
                results["dotenv_imports"].append(str(backend_api))
    
    # Check api.py (root)
    root_api = Path("api.py")
    if root_api.exists():
        with open(root_api) as f:
            content = f.read()
            if "os.getenv" in content:
                results["os_getenv_usage"].append(str(root_api))
    
    return results

def analyze_routes() -> Dict[str, any]:
    """Analyze route configurations"""
    results = {
        "route_files": [],
        "route_prefixes": [],
        "route_conflicts": []
    }
    
    # Check backend/api.py for routes
    backend_api = Path("backend/api.py")
    if backend_api.exists():
        with open(backend_api) as f:
            content = f.read()
            if "app.post" in content or "app.get" in content or "include_router" in content:
                results["route_files"].append(str(backend_api))
                # Extract route prefixes
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "API_PREFIX" in line or "prefix=" in line:
                        results["route_prefixes"].append({
                            "file": str(backend_api),
                            "line": i + 1,
                            "content": line.strip()
                        })
    
    return results

def main():
    """Main investigation"""
    root_dir = Path(".")
    
    print("=" * 80)
    print("ENVIRONMENT VARIABLE & ROUTE INVESTIGATION")
    print("=" * 80)
    
    # Find .env files
    print("\n1. ENVIRONMENT FILES:")
    print("-" * 80)
    env_files = find_env_files(root_dir)
    if env_files:
        for env_file in env_files:
            print(f"  [OK] Found: {env_file}")
            # Check if it's in .gitignore
            gitignore = Path(".gitignore")
            if gitignore.exists():
                with open(gitignore) as f:
                    gitignore_content = f.read()
                    if env_file.name in gitignore_content or str(env_file) in gitignore_content:
                        print(f"    -> In .gitignore: YES")
                    else:
                        print(f"    -> In .gitignore: NO ([WARN] Should be ignored!)")
    else:
        print("  ✓ No .env files found")
    
    # Find .cursorignore files
    print("\n2. CURSORIGNORE FILES:")
    print("-" * 80)
    cursorignore_files = find_cursorignore_files(root_dir)
    if cursorignore_files:
        for cursorignore_file in cursorignore_files:
            print(f"  [OK] Found: {cursorignore_file}")
            with open(cursorignore_file) as f:
                content = f.read()
                print(f"    Content ({len(content)} chars):")
                for line in content.split("\n")[:10]:  # First 10 lines
                    if line.strip():
                        print(f"      {line}")
    else:
        print("  [OK] No .cursorignore files found")
    
    # Analyze env loading
    print("\n3. ENVIRONMENT VARIABLE LOADING:")
    print("-" * 80)
    env_analysis = analyze_env_loading()
    if env_analysis["uses_dotenv"]:
        print(f"  [OK] Uses python-dotenv: YES")
        print(f"    Files: {', '.join(env_analysis['dotenv_imports'])}")
    else:
        print(f"  [WARN] Uses python-dotenv: NO (using os.getenv directly)")
        print(f"    Files using os.getenv: {', '.join(env_analysis['os_getenv_usage'])}")
    
    # Analyze routes
    print("\n4. ROUTE CONFIGURATIONS:")
    print("-" * 80)
    route_analysis = analyze_routes()
    if route_analysis["route_files"]:
        print(f"  [OK] Route files found: {', '.join(route_analysis['route_files'])}")
        if route_analysis["route_prefixes"]:
            print(f"  [OK] Route prefixes:")
            for prefix_info in route_analysis["route_prefixes"]:
                print(f"    - {prefix_info['file']}:{prefix_info['line']} - {prefix_info['content']}")
    
    # Check for conflicts
    print("\n5. POTENTIAL ISSUES:")
    print("-" * 80)
    issues = []
    
    # Check if .env files are properly ignored
    env_files_not_ignored = []
    if env_files:
        gitignore = Path(".gitignore")
        if gitignore.exists():
            with open(gitignore) as f:
                gitignore_content = f.read()
                for env_file in env_files:
                    if env_file.name not in gitignore_content and str(env_file) not in gitignore_content:
                        env_files_not_ignored.append(env_file)
    
    if env_files_not_ignored:
        issues.append(f"[WARN] .env files not in .gitignore: {[str(f) for f in env_files_not_ignored]}")
    
    # Check for load_dotenv usage
    if not env_analysis["uses_dotenv"] and env_files:
        issues.append("[WARN] .env files exist but load_dotenv() is not used - env vars won't be loaded from .env files")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  [OK] No obvious issues found")
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
