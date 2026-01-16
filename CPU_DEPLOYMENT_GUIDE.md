# CPU-Only Production Deployment

## Setup

```bash
# In WSL
cd /mnt/d/EMBERS-BANK/DiffRhythm-Linux
source .venv/bin/activate

# Install backend dependencies
pip install -r backend_requirements.txt

# Create directories
mkdir -p outputs temp

# Start server
uvicorn cpu_backend:app --host 0.0.0.0 --port 8000
```

## Usage

### Submit Job
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Hello world\n[00:05.00]This is a test",
    "style_prompt": "folk, acoustic guitar",
    "duration": 95
  }'

# Response:
# {
#   "job_id": "abc-123",
#   "queue_position": 1,
#   "estimated_wait_minutes": 20
# }
```

### Check Status
```bash
curl http://localhost:8000/status/abc-123

# Response:
# {
#   "job_id": "abc-123",
#   "status": "processing",  # queued, processing, completed, failed
#   "created_at": "2025-01-15T10:00:00",
#   "started_at": "2025-01-15T10:05:00"
# }
```

### Download
```bash
curl http://localhost:8000/download/abc-123 -o song.wav
```

### Check Queue
```bash
curl http://localhost:8000/queue

# Response:
# {
#   "queue_length": 3,
#   "current_job": "abc-123",
#   "estimated_wait_minutes": 60
# }
```

## Performance

- **Generation time**: 15-25 minutes per song
- **Concurrent jobs**: Queued (one at a time)
- **Memory**: 8-16GB RAM
- **CPU**: 100% utilization during generation

## Production Considerations

### 1. Persistence
Jobs are in-memory. Add database:
```python
# Use SQLite/PostgreSQL
import sqlite3
```

### 2. Cleanup
```python
# Add cleanup job
@app.on_event("startup")
async def cleanup_old_files():
    # Delete files older than 24 hours
    pass
```

### 3. Rate Limiting
```python
from slowapi import Limiter
limiter = Limiter(key_func=lambda: "global")

@app.post("/generate")
@limiter.limit("5/hour")  # 5 jobs per hour
def generate(...):
    pass
```

### 4. Monitoring
```bash
# Check logs
tail -f uvicorn.log

# Monitor CPU
htop
```

## Docker Deployment

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip espeak-ng

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r backend_requirements.txt

EXPOSE 8000
CMD ["uvicorn", "cpu_backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t diffrhythm-cpu .
docker run -p 8000:8000 diffrhythm-cpu
```

## Limitations

- **20 minutes per song** - Users must wait
- **No concurrent processing** - Queue system only
- **Not scalable** - Single worker thread
- **Memory intensive** - 8-16GB per generation

## This is the reality of CPU-only inference.
