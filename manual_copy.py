#!/usr/bin/env python3
import shutil

# Copy the blob to the target location
shutil.copy2(
    "pretrained/models--xlm-roberta-base/blobs/6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb",
    "pretrained/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model"
)
print("Copied sentencepiece model")