#!/bin/bash
# DiffRhythm Docker Build and Deploy Script

set -e

echo "🐳 DiffRhythm Docker Deployment"
echo "================================"

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t diffrhythm:latest .

# Test the image
echo "🧪 Testing Docker image..."
docker run --rm diffrhythm:latest python3 test_core_models.py

# Start the services
echo "🚀 Starting DiffRhythm services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check health
echo "🔍 Checking service health..."
docker-compose ps

echo "✅ DiffRhythm is ready!"
echo "🌐 Access your app at: http://localhost:8000"
echo "📊 View logs: docker-compose logs -f diffrhythm"
echo "🛑 Stop services: docker-compose down"