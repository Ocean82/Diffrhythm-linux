#!/bin/bash
# Verify backend connection and CORS configuration

echo "=========================================="
echo "Backend Connection Verification"
echo "=========================================="

API_URL="http://52.0.207.242:8000/api/v1"

echo ""
echo "1. Testing Health Endpoint..."
HEALTH_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/health" 2>&1)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | sed '/HTTP_CODE/d')

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✓ Health endpoint accessible"
    echo "   Response: $HEALTH_BODY" | head -3
else
    echo "   ✗ Health endpoint failed (HTTP $HTTP_CODE)"
    echo "   Response: $HEALTH_BODY"
fi

echo ""
echo "2. Testing CORS Headers..."
CORS_RESPONSE=$(curl -s -I -X OPTIONS "$API_URL/health" \
    -H "Origin: http://localhost:3000" \
    -H "Access-Control-Request-Method: POST" 2>&1)

if echo "$CORS_RESPONSE" | grep -q "access-control-allow-origin"; then
    echo "   ✓ CORS headers present"
    echo "$CORS_RESPONSE" | grep -i "access-control" | head -5
else
    echo "   ⚠ CORS headers not detected (may need service running)"
fi

echo ""
echo "3. Testing API Documentation..."
DOCS_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" "http://52.0.207.242:8000/docs" 2>&1)
DOCS_CODE=$(echo "$DOCS_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)

if [ "$DOCS_CODE" = "200" ]; then
    echo "   ✓ API documentation accessible"
    echo "   URL: http://52.0.207.242:8000/docs"
else
    echo "   ✗ API documentation not accessible (HTTP $DOCS_CODE)"
fi

echo ""
echo "4. Backend Connection Summary:"
echo "   Base URL: http://52.0.207.242:8000"
echo "   API Prefix: /api/v1"
echo "   Full URL: $API_URL"
echo ""
echo "   Endpoints:"
echo "   - GET  $API_URL/health"
echo "   - POST $API_URL/generate"
echo "   - GET  $API_URL/status/{job_id}"
echo "   - GET  $API_URL/download/{job_id}"
echo "   - GET  $API_URL/queue"
echo ""
echo "=========================================="
