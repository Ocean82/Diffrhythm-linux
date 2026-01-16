FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    espeak-ng \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code (excluding pretrained models)
COPY --exclude=pretrained . .

# Create directories
RUN mkdir -p /app/output /app/pretrained

# Set Python path
ENV PYTHONPATH=/app

# Download models at runtime
COPY download_models.py .
RUN python3 download_models.py

# Expose port for web app
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python3", "api.py"]