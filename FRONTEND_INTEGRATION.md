# Frontend Integration Guide

## Backend API Endpoints

The DiffRhythm backend is deployed at: **http://52.0.207.242:8000**

### Base URL
```
http://52.0.207.242:8000
```

### API Prefix
All API endpoints are prefixed with `/api/v1`

## Available Endpoints

### 1. Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy" | "degraded",
  "models_loaded": true | false,
  "device": "cpu" | "cuda",
  "queue_length": 0,
  "active_jobs": 0,
  "version": "1.0.0"
}
```

### 2. Submit Generation Job
```http
POST /api/v1/generate
Content-Type: application/json
X-API-Key: your-api-key (if configured)
```

**Request Body:**
```json
{
  "lyrics": "[00:00.00]First line\n[00:05.00]Second line",
  "style_prompt": "pop, upbeat, energetic",
  "audio_length": 95,
  "batch_size": 1
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "queue_position": 0,
  "estimated_wait_minutes": 0,
  "message": "Job queued successfully. Use /status/{job_id} to check progress."
}
```

### 3. Check Job Status
```http
GET /api/v1/status/{job_id}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued" | "processing" | "completed" | "failed",
  "created_at": "2025-01-23T12:00:00Z",
  "started_at": "2025-01-23T12:05:00Z" | null,
  "completed_at": "2025-01-23T12:25:00Z" | null,
  "output_file": "uuid-string/output_fixed.wav" | null,
  "error": null | "error message",
  "queue_position": 0 | null,
  "estimated_wait_minutes": 0 | null
}
```

### 4. Download Generated Audio
```http
GET /api/v1/download/{job_id}
```

**Response:** Binary WAV file (audio/wav)

### 5. Queue Status
```http
GET /api/v1/queue
```

**Response:**
```json
{
  "queue_length": 0,
  "current_job": "uuid-string" | null,
  "estimated_wait_minutes": 0
}
```

### 6. API Documentation
```http
GET /docs
```
Interactive Swagger/OpenAPI documentation

## Frontend Integration Examples

### JavaScript/TypeScript (Fetch API)

```javascript
const API_BASE_URL = 'http://52.0.207.242:8000/api/v1';
const API_KEY = 'your-api-key'; // Optional if not configured

// Submit generation job
async function generateSong(lyrics, stylePrompt, audioLength = 95) {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY && { 'X-API-Key': API_KEY })
    },
    body: JSON.stringify({
      lyrics: lyrics,
      style_prompt: stylePrompt,
      audio_length: audioLength,
      batch_size: 1
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Check job status
async function checkJobStatus(jobId) {
  const response = await fetch(`${API_BASE_URL}/status/${jobId}`, {
    headers: {
      ...(API_KEY && { 'X-API-Key': API_KEY })
    }
  });
  
  return await response.json();
}

// Download audio
async function downloadAudio(jobId) {
  const response = await fetch(`${API_BASE_URL}/download/${jobId}`, {
    headers: {
      ...(API_KEY && { 'X-API-Key': API_KEY })
    }
  });
  
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }
  
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `generated_song_${jobId}.wav`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

// Poll for completion
async function waitForCompletion(jobId, onProgress) {
  const maxWaitTime = 1800000; // 30 minutes
  const pollInterval = 5000; // 5 seconds
  const startTime = Date.now();
  
  while (Date.now() - startTime < maxWaitTime) {
    const status = await checkJobStatus(jobId);
    
    if (onProgress) {
      onProgress(status);
    }
    
    if (status.status === 'completed') {
      return status;
    }
    
    if (status.status === 'failed') {
      throw new Error(`Generation failed: ${status.error}`);
    }
    
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }
  
  throw new Error('Generation timeout');
}

// Complete workflow example
async function generateAndDownload(lyrics, stylePrompt) {
  try {
    // 1. Submit job
    const job = await generateSong(lyrics, stylePrompt);
    console.log('Job submitted:', job.job_id);
    
    // 2. Wait for completion with progress updates
    const completed = await waitForCompletion(job.job_id, (status) => {
      console.log(`Status: ${status.status}, Queue: ${status.queue_position || 0}`);
    });
    
    // 3. Download audio
    await downloadAudio(job.job_id);
    console.log('Audio downloaded successfully');
    
    return completed;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

### React Example

```tsx
import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://52.0.207.242:8000/api/v1';

interface GenerationRequest {
  lyrics: string;
  style_prompt: string;
  audio_length?: number;
  batch_size?: number;
}

interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  queue_position?: number;
  estimated_wait_minutes?: number;
  error?: string;
}

export function useDiffRhythmGeneration() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async (request: GenerationRequest) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setJobId(data.job_id);
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;
    
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
        const data = await response.json();
        setStatus(data);
        
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error checking status:', err);
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [jobId]);

  const download = async () => {
    if (!jobId || status?.status !== 'completed') return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/download/${jobId}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `song_${jobId}.wav`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    }
  };

  return { generate, status, loading, error, download };
}
```

## CORS Configuration

The backend is configured to accept requests from any origin (`CORS_ORIGINS=*`). For production, update the environment variable to restrict to specific domains:

```bash
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## Error Handling

### Common HTTP Status Codes

- **200 OK**: Request successful
- **202 Accepted**: Job queued successfully
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Job not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error during generation
- **503 Service Unavailable**: Models not loaded

### Error Response Format

```json
{
  "error": "Error message",
  "details": {}
}
```

## Rate Limiting

Default rate limit: **10 requests per hour** per IP address.

To adjust, set `RATE_LIMIT_PER_HOUR` environment variable on the server.

## API Key Authentication (Optional)

If `API_KEY` is set in the server environment, include it in requests:

```javascript
headers: {
  'X-API-Key': 'your-api-key'
}
```

If not configured, API key is optional and requests will work without it.

## Testing the Connection

### Quick Test (cURL)

```bash
# Health check
curl http://52.0.207.242:8000/api/v1/health

# Submit job
curl -X POST http://52.0.207.242:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Test song",
    "style_prompt": "pop, upbeat",
    "audio_length": 95
  }'
```

### Browser Test

Open in browser: `http://52.0.207.242:8000/docs` for interactive API documentation.

## Notes

- Generation takes approximately **20-30 minutes** per song on CPU
- Jobs are processed sequentially (one at a time)
- Audio files are stored on the server and can be downloaded via the download endpoint
- Generated files may be cleaned up after a period (check server configuration)
