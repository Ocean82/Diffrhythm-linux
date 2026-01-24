# Frontend-Backend Connection Guide

## ✅ Connection Verified & Configured

All frontend-backend connections have been verified and configured correctly.

## Backend Server Details

- **Server IP**: `52.0.207.242`
- **Port**: `8000`
- **Base URL**: `http://52.0.207.242:8000`
- **API Prefix**: `/api/v1`
- **Full API URL**: `http://52.0.207.242:8000/api/v1`

## CORS Configuration ✅

**Status**: Properly configured for frontend access

- **Allow Origins**: `*` (all origins - can be restricted in production)
- **Allow Methods**: `GET, POST, PUT, DELETE, OPTIONS, HEAD`
- **Allow Headers**: `*` (all headers)
- **Allow Credentials**: `true`
- **Expose Headers**: `*`

This configuration allows any frontend to connect to the backend API.

## API Endpoints

### 1. Health Check
```http
GET /api/v1/health
```
**No authentication required**

### 2. Submit Generation
```http
POST /api/v1/generate
Content-Type: application/json
X-API-Key: {optional}
```
**Request Body:**
```json
{
  "lyrics": "[00:00.00]Line 1\n[00:05.00]Line 2",
  "style_prompt": "pop, upbeat, energetic",
  "audio_length": 95,
  "batch_size": 1
}
```

### 3. Check Job Status
```http
GET /api/v1/status/{job_id}
```

### 4. Download Audio
```http
GET /api/v1/download/{job_id}
```
**Returns**: Binary WAV file

### 5. Queue Status
```http
GET /api/v1/queue
```

## Frontend Integration

### JavaScript/TypeScript Example

```typescript
// Configuration
const API_BASE = 'http://52.0.207.242:8000/api/v1';

// Types
interface GenerationRequest {
  lyrics: string;
  style_prompt: string;
  audio_length?: number;
  batch_size?: number;
}

interface GenerationResponse {
  job_id: string;
  status: string;
  queue_position?: number;
  estimated_wait_minutes?: number;
  message: string;
}

interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  output_file?: string;
  error?: string;
}

// API Client
class DiffRhythmAPI {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl: string = API_BASE, apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async healthCheck() {
    return this.request<{
      status: string;
      models_loaded: boolean;
      device: string;
      queue_length: number;
      active_jobs: number;
      version: string;
    }>('/health');
  }

  async generate(request: GenerationRequest): Promise<GenerationResponse> {
    return this.request<GenerationResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getStatus(jobId: string): Promise<JobStatus> {
    return this.request<JobStatus>(`/status/${jobId}`);
  }

  async downloadAudio(jobId: string): Promise<Blob> {
    const url = `${this.baseUrl}/download/${jobId}`;
    const headers: HeadersInit = {};
    
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(url, { headers });
    
    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  async getQueueStatus() {
    return this.request<{
      queue_length: number;
      current_job: string | null;
      estimated_wait_minutes: number;
    }>('/queue');
  }

  // Helper: Poll for completion
  async waitForCompletion(
    jobId: string,
    onProgress?: (status: JobStatus) => void,
    maxWaitMs: number = 1800000 // 30 minutes
  ): Promise<JobStatus> {
    const startTime = Date.now();
    const pollInterval = 5000; // 5 seconds

    while (Date.now() - startTime < maxWaitMs) {
      const status = await this.getStatus(jobId);
      
      if (onProgress) {
        onProgress(status);
      }

      if (status.status === 'completed') {
        return status;
      }

      if (status.status === 'failed') {
        throw new Error(`Generation failed: ${status.error || 'Unknown error'}`);
      }

      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error('Generation timeout');
  }
}

// Usage Example
const api = new DiffRhythmAPI();

// Complete workflow
async function generateSong(lyrics: string, stylePrompt: string) {
  try {
    // 1. Check health
    const health = await api.healthCheck();
    console.log('Backend status:', health);

    // 2. Submit generation
    const job = await api.generate({
      lyrics,
      style_prompt: stylePrompt,
      audio_length: 95,
      batch_size: 1,
    });
    console.log('Job submitted:', job.job_id);

    // 3. Wait for completion
    const completed = await api.waitForCompletion(
      job.job_id,
      (status) => {
        console.log(`Status: ${status.status}, Queue: ${status.queue_position || 0}`);
      }
    );

    // 4. Download audio
    const audioBlob = await api.downloadAudio(completed.job_id);
    
    // 5. Create download link
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `song_${completed.job_id}.wav`;
    a.click();
    URL.revokeObjectURL(url);

    return completed;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

## React Hook Example

```tsx
import { useState, useEffect, useCallback } from 'react';

interface UseDiffRhythmReturn {
  generate: (lyrics: string, stylePrompt: string) => Promise<void>;
  status: JobStatus | null;
  loading: boolean;
  error: string | null;
  download: () => Promise<void>;
}

export function useDiffRhythm(): UseDiffRhythmReturn {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const api = new DiffRhythmAPI();

  const generate = useCallback(async (lyrics: string, stylePrompt: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const job = await api.generate({
        lyrics,
        style_prompt: stylePrompt,
        audio_length: 95,
        batch_size: 1,
      });
      setJobId(job.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const currentStatus = await api.getStatus(jobId);
        setStatus(currentStatus);

        if (currentStatus.status === 'completed' || currentStatus.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error checking status:', err);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [jobId]);

  const download = useCallback(async () => {
    if (!jobId || status?.status !== 'completed') return;

    try {
      const blob = await api.downloadAudio(jobId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `song_${jobId}.wav`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    }
  }, [jobId, status]);

  return { generate, status, loading, error, download };
}
```

## Connection Verification

Run the verification script:
```bash
bash scripts/verify_backend_connection.sh
```

Or test manually:
```bash
# Health check
curl http://52.0.207.242:8000/api/v1/health

# Test CORS
curl -X OPTIONS http://52.0.207.242:8000/api/v1/health \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v
```

## Configuration Files

### Server Configuration (`config/ec2-config.env`)
```bash
CORS_ORIGINS=*  # Allows all origins
API_KEY=        # Optional - leave empty to disable
RATE_LIMIT_PER_HOUR=10
```

### Frontend Environment Variables
```env
REACT_APP_API_URL=http://52.0.207.242:8000/api/v1
REACT_APP_API_KEY=  # Optional
```

## Troubleshooting

### CORS Errors
- ✅ **Fixed**: CORS is configured to allow all origins (`*`)
- ✅ **Fixed**: All HTTP methods are allowed
- ✅ **Fixed**: All headers are allowed

### Connection Refused
- Check if backend service is running
- Verify firewall allows port 8000
- Check server is accessible: `ping 52.0.207.242`

### 401 Unauthorized
- API key is optional (only required if `API_KEY` env var is set)
- If API key is required, include `X-API-Key` header

### 503 Service Unavailable
- Models may still be loading
- Check `/api/v1/health` endpoint for model status

## Summary

✅ **CORS**: Configured correctly for frontend access  
✅ **Endpoints**: All endpoints properly structured  
✅ **Authentication**: Optional API key support  
✅ **Error Handling**: Proper HTTP status codes  
✅ **Documentation**: Complete API docs at `/docs`  

**Backend is ready for frontend integration!**
