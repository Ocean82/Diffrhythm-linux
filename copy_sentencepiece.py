#!/usr/bin/env python3
import shutil
from pathlib import Path

# Copy the blob file to replace the symlink
source = Path("pretrained/models--xlm-roberta-base/blobs/6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb")
target = Path("pretrained/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model")

if source.exists():
    shutil.copy2(source, target)
    print(f"✅ Copied sentencepiece model: {target}")
    print(f"Size: {target.stat().st_size / (1024*1024):.1f} MB")
else:
    print(f"❌ Source not found: {source}")