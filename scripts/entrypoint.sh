#!/bin/bash
set -e

# Define model directory
MODEL_DIR="/app/pretrained"

# Check if models exist (basic check for one key file)
if [ ! -f "$MODEL_DIR/vae_model.pt" ]; then
    echo "Models not found in $MODEL_DIR. Downloading..."
    python3 download_models.py
else
    echo "Models found. Skipping download."
fi

# Start the application
echo "Starting DiffRhythm API..."
exec "$@"
