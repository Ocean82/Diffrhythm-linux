# Backend Connection Summary

## Server Information

- **Server IP**: `52.0.207.242`
- **API Base URL**: `http://52.0.207.242:8000`
- **API Prefix**: `/api/v1`
- **Full API URL**: `http://52.0.207.242:8000/api/v1`

## Connection Status

✅ **Backend files deployed** to `/opt/diffrhythm` on server  
⚠️ **Docker build pending** (disk space issue - 94% used, need cleanup)  
✅ **CORS configured** for frontend access  
✅ **API endpoints** ready for frontend integration  

## Frontend Connection Configuration

### CORS Settings
- **Current**: `CORS_ORIGINS=*` (allows all origins)
- **Status**: ✅ Configured correctly for development
- **Production**: Update to specific domains when ready

### API Endpoints for Frontend

All endpoints are prefixed with `/api/v1`:

1. **Health Check**
   ```
   GET http://52.0.207.242:8000/api/v1/health
   ```

2. **Submit Generation**
   ```
   POST http://52.0.207.242:8000/api/v1/generate
   Headers: Content-Type: application/json
            X-API-Key: {optional if API_KEY not set}
   Body: {
     "lyrics": "[00:00.00]...",
     "style_prompt": "...",
     "audio_length": 95,
     "batch_size": 1
   }
   ```

3. **Check Job Status**
   ```
   GET http://52.0.207.242:8000/api/v1/status/{job_id}
   ```

4. **Download Audio**
   ```
   GET http://52.0.207.242:8000/api/v1/download/{job_id}
   ```

5. **Queue Status**
   ```
   GET http://52.0.207.242:8000/api/v1/queue
   ```

### Frontend Integration Code

```javascript
// Configuration
const API_BASE_URL = 'http://52.0.207.242:8000/api/v1';

// Example: Submit generation
async function submitGeneration(lyrics, stylePrompt) {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      lyrics: lyrics,
      style_prompt: stylePrompt,
      audio_length: 95,
      batch_size: 1
    })
  });
  return await response.json();
}

// Example: Check status
async function checkStatus(jobId) {
  const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
  return await response.json();
}

// Example: Download audio
async function downloadAudio(jobId) {
  const response = await fetch(`${API_BASE_URL}/download/${jobId}`);
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `song_${jobId}.wav`;
  a.click();
  window.URL.revokeObjectURL(url);
}
```

## Current Configuration

### Backend API (`backend/api.py`)
- ✅ Endpoints: `/api/v1/*`
- ✅ CORS: Enabled for all origins (`*`)
- ✅ Methods: GET, POST, PUT, DELETE, OPTIONS
- ✅ Headers: All headers allowed
- ✅ API Key: Optional (not required if `API_KEY` env var not set)

### Security Settings
- **Rate Limiting**: 10 requests/hour (configurable)
- **API Key**: Optional (set `API_KEY` env var to enable)
- **CORS**: Open for development (`*`)

## Next Steps

1. **Clean up disk space** on server (currently 94% used)
2. **Complete Docker build** once space is available
3. **Start services**: `docker-compose -f docker-compose.prod.yml up -d`
4. **Test connection** from frontend
5. **Update CORS_ORIGINS** in production to specific domains

## Testing Connection

### From Browser Console
```javascript
fetch('http://52.0.207.242:8000/api/v1/health')
  .then(r => r.json())
  .then(console.log);
```

### From cURL
```bash
curl http://52.0.207.242:8000/api/v1/health
```

## Notes

- The backend is configured to accept requests from any origin (CORS: `*`)
- API key authentication is optional (only required if `API_KEY` is set)
- All endpoints return JSON except `/download/{job_id}` which returns binary WAV
- Generation jobs are processed sequentially (one at a time)
- Estimated generation time: 20-30 minutes per song on CPU
