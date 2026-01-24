#!/bin/bash
# Health check script for DiffRhythm API
# Can be used with monitoring systems or cron

API_URL="${API_URL:-http://localhost:8000}"
ENDPOINT="${ENDPOINT:-/api/v1/health}"

response=$(curl -s -w "\n%{http_code}" "${API_URL}${ENDPOINT}")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    status=$(echo "$body" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    models_loaded=$(echo "$body" | grep -o '"models_loaded":[^,]*' | cut -d':' -f2)
    
    if [ "$status" = "healthy" ] && [ "$models_loaded" = "true" ]; then
        echo "OK: Service is healthy"
        exit 0
    else
        echo "WARNING: Service is degraded (status: $status, models_loaded: $models_loaded)"
        exit 1
    fi
else
    echo "CRITICAL: Health check failed (HTTP $http_code)"
    exit 2
fi
