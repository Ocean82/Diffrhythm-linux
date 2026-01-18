import time
import sys
import os

def test_import(module_name):
    print(f"Testing import of {module_name}...", end="", flush=True)
    start = time.time()
    try:
        if module_name == "muq":
            from muq import MuQMuLan
        else:
            __import__(module_name)
        print(f" [OK] (took {time.time() - start:.2f}s)")
    except Exception as e:
        print(f" [FAIL] {e}")

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")
    print("-" * 40)
    test_import("torch")
    test_import("torchaudio")
    test_import("librosa")
    test_import("huggingface_hub")
    test_import("muq")
    print("-" * 40)
    print("All tests finished.")
