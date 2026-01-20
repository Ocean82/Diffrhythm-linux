#!/bin/bash
echo "Starting DiffRhythm Checkpoint Pipeline..."
cd /mnt/d/EMBERS-BANK/DiffRhythm-LINUX
source .venv/bin/activate
python checkpoint_pipeline.py