# DiffRhythm Production Deployment Guide

## Overview

This guide covers deploying DiffRhythm as a production backend service on AWS EC2.

## Prerequisites

- AWS EC2 instance (t3.xlarge or larger recommended)
- Ubuntu 22.04 LTS
- SSH access to the instance
- Basic knowledge of Docker and Linux

## Instance Requirements

### Minimum Requirements
- **Instance Type**: t3.xlarge
- **vCPU**: 4
- **RAM**: 16 GB
- **Storage**: 50 GB (for models + application)
- **OS**: Ubuntu 22.04 LTS

### Recommended Requirements
- **Instance Type**: t3.2xlarge or larger
- **vCPU**: 8+
- **RAM**: 32 GB+
- **Storage**: 100 GB SSD
- **Network**: High bandwidth

## Step 1: EC2 Instance Setup

### 1.1 Launch EC2 Instance

1. Log into AWS Console
2. Launch EC2 instance with:
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: t3.xlarge or larger
   - Storage: 50+ GB
   - Security group: Allow SSH (22), HTTP (80), HTTPS (443), API (8000)

### 1.2 Initial Setup

SSH into your instance and run:

```bash
# Clone or upload your DiffRhythm project
cd /opt
sudo git clone <your-repo-url> diffrhythm
# OR upload files via SCP

# Run initial setup
cd diffrhythm
sudo bash scripts/ec2-setup.sh
```

## Step 2: Configuration

### 2.1 Environment Variables

Copy and edit the configuration file:

```bash
cd /opt/diffrhythm
sudo cp config/ec2-config.env .env
sudo nano .env
```

Key settings to configure:
- `API_KEY`: Set a secure API key for authentication
- `DEVICE`: Set to `cpu` (or `cuda` if using GPU instance)
- `MODEL_CACHE_DIR`: Path to store models
- `RATE_LIMIT_PER_HOUR`: Adjust based on your needs

### 2.2 Model Download

Models will be downloaded automatically on first run, or download manually:

```bash
cd /opt/diffrhythm
python3 download_models.py
```

## Step 3: Docker Deployment

### 3.1 Build Docker Image

```bash
cd /opt/diffrhythm
sudo docker build -f Dockerfile.prod -t diffrhythm:prod .
```

### 3.2 Start Services

```bash
# Start API service
sudo docker-compose -f docker-compose.prod.yml up -d

# Check status
sudo docker-compose -f docker-compose.prod.yml ps

# View logs
sudo docker-compose -f docker-compose.prod.yml logs -f
```

### 3.3 Systemd Service (Optional)

For automatic startup on boot:

```bash
sudo cp config/systemd/diffrhythm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable diffrhythm-api
sudo systemctl start diffrhythm-api
```

## Step 4: Nginx Reverse Proxy (Production)

### 4.1 Configure Nginx

```bash
# Edit nginx configuration
sudo nano config/nginx.conf

# Start nginx with production profile
sudo docker-compose -f docker-compose.prod.yml --profile production up -d nginx
```

### 4.2 SSL/TLS (Optional)

For HTTPS:

1. Obtain SSL certificate (Let's Encrypt recommended)
2. Place certificates in `config/nginx/ssl/`
3. Uncomment HTTPS server block in `config/nginx.conf`
4. Restart nginx

## Step 5: Verification

### 5.1 Health Check

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Expected response:
# {
#   "status": "healthy",
#   "models_loaded": true,
#   "device": "cpu",
#   "queue_length": 0,
#   "active_jobs": 0,
#   "version": "1.0.0"
# }
```

### 5.2 Test Generation

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "lyrics": "[00:00.00]Test song\n[00:05.00]This is a test",
    "style_prompt": "pop, upbeat, energetic",
    "audio_length": 95,
    "batch_size": 1
  }'
```

## Step 6: Monitoring

### 6.1 Health Check Script

Set up cron job for health monitoring:

```bash
# Add to crontab
sudo crontab -e

# Add line (runs every 5 minutes):
*/5 * * * * /opt/diffrhythm/scripts/health-check.sh
```

### 6.2 Metrics Endpoint

Access Prometheus metrics:

```bash
curl http://localhost:8000/api/v1/metrics
```

### 6.3 Logs

View application logs:

```bash
# Docker logs
sudo docker-compose -f docker-compose.prod.yml logs -f diffrhythm-api

# System logs (if using systemd)
sudo journalctl -u diffrhythm-api -f
```

## Step 7: Maintenance

### 7.1 Update Application

```bash
cd /opt/diffrhythm
sudo git pull  # or upload new files
sudo docker-compose -f docker-compose.prod.yml build
sudo docker-compose -f docker-compose.prod.yml up -d
```

### 7.2 Backup

Important files to backup:
- Models: `/opt/diffrhythm/pretrained/`
- Configuration: `/opt/diffrhythm/.env`
- Generated outputs: `/opt/diffrhythm/output/`

### 7.3 Cleanup

Clean old generated files:

```bash
# Remove files older than 7 days
find /opt/diffrhythm/output -type f -mtime +7 -delete
```

## Troubleshooting

### Models Not Loading

1. Check model files exist: `ls -lh /opt/diffrhythm/pretrained/`
2. Check disk space: `df -h`
3. Check logs: `sudo docker-compose logs diffrhythm-api`

### High Memory Usage

1. Reduce `MAX_BATCH_SIZE` in `.env`
2. Use `--chunked` flag (already enabled)
3. Consider larger instance type

### Generation Timeouts

1. Increase `GENERATION_TIMEOUT` in `.env`
2. Check CPU usage: `htop`
3. Verify instance has sufficient resources

### API Not Responding

1. Check service status: `sudo docker-compose ps`
2. Check health endpoint: `curl http://localhost:8000/api/v1/health`
3. Check firewall: `sudo ufw status`
4. Review logs for errors

## Performance Tuning

### CPU Optimization

Edit `.env`:
```bash
CPU_STEPS=16          # Lower = faster, lower quality
CPU_CFG_STRENGTH=2.0  # Lower = faster
```

### Memory Optimization

- Use chunked decoding (enabled by default)
- Limit batch size to 1-2
- Close other applications

## Security Best Practices

1. **API Key**: Always set a strong API key
2. **Firewall**: Restrict access to necessary ports only
3. **SSL/TLS**: Use HTTPS in production
4. **Updates**: Keep system and Docker images updated
5. **Logs**: Monitor logs for suspicious activity
6. **Rate Limiting**: Configure appropriate rate limits

## Scaling

### Horizontal Scaling

For multiple instances:
1. Use load balancer (AWS ALB)
2. Share Redis for job queue
3. Use shared storage (EFS) for models

### Vertical Scaling

For better performance:
1. Upgrade to larger instance type
2. Enable GPU if available
3. Increase memory allocation

## Support

For issues or questions:
- Check logs: `sudo docker-compose logs`
- Review health endpoint
- Check system resources: `htop`, `df -h`
- Review this documentation

## Quick Reference

```bash
# Start service
sudo docker-compose -f docker-compose.prod.yml up -d

# Stop service
sudo docker-compose -f docker-compose.prod.yml down

# View logs
sudo docker-compose -f docker-compose.prod.yml logs -f

# Restart service
sudo docker-compose -f docker-compose.prod.yml restart

# Check status
sudo docker-compose -f docker-compose.prod.yml ps

# Health check
curl http://localhost:8000/api/v1/health
```
