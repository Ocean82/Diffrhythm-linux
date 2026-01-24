# Quick Start - Production Deployment

## Local Testing (Docker)

```bash
# Build production image
docker build -f Dockerfile.prod -t diffrhythm:prod .

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Test API
curl http://localhost:8000/api/v1/health
```

## Local Testing (Direct)

```bash
# Activate virtual environment
source .venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run API
python3 -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

## AWS EC2 Deployment

1. **Initial Setup** (one-time):
```bash
sudo bash scripts/ec2-setup.sh
```

2. **Deploy Application**:
```bash
cd /opt/diffrhythm
sudo bash scripts/deploy.sh
```

3. **Configure**:
```bash
sudo cp config/ec2-config.env .env
sudo nano .env  # Edit configuration
```

4. **Start Service**:
```bash
sudo docker-compose -f docker-compose.prod.yml up -d
# OR
sudo systemctl start diffrhythm-api
```

## API Usage

### Submit Generation Job
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "lyrics": "[00:00.00]Your lyrics here",
    "style_prompt": "pop, upbeat, energetic",
    "audio_length": 95,
    "batch_size": 1
  }'
```

### Check Job Status
```bash
curl http://localhost:8000/api/v1/status/{job_id}
```

### Download Audio
```bash
curl http://localhost:8000/api/v1/download/{job_id} -o output.wav
```

## Configuration

Edit `.env` file or set environment variables:

```bash
# Required
DEVICE=cpu
API_KEY=your-secure-api-key

# Optional
RATE_LIMIT_PER_HOUR=10
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Monitoring

- **Health**: `curl http://localhost:8000/api/v1/health`
- **Metrics**: `curl http://localhost:8000/api/v1/metrics`
- **Queue**: `curl http://localhost:8000/api/v1/queue`
- **Logs**: `docker-compose -f docker-compose.prod.yml logs -f`

## Troubleshooting

1. **Models not loading**: Check `pretrained/` directory has model files
2. **Port already in use**: Change `PORT` in `.env`
3. **Permission errors**: Check Docker user permissions
4. **Memory issues**: Increase instance size or reduce batch size

For detailed information, see `DEPLOYMENT.md`.
