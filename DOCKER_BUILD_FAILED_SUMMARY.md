# Docker Build Failed - Disk Space Issue

## Build Status: **FAILED** ❌

### Error
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

### Build Progress
- **Reached**: Step 10 - Installing Python packages
- **Progress**: Successfully downloaded and built `muq` package
- **Failed**: During installation of all collected packages
- **Time**: ~7 minutes into build

### Root Cause
**Insufficient disk space during Docker build process**

The build process requires significant temporary space for:
1. Downloading packages (~5GB+)
2. Building wheels (muq, jieba)
3. Installing packages
4. Docker's overlay filesystem overhead

Even though `df -h` shows 7.6GB free, Docker's build process uses additional temporary space in `/var/lib/docker` that fills up during the installation phase.

### Current Disk Status
- **System Disk**: 85% usage (7.6GB free)
- **Docker Images**: 1.015GB (reclaimable)
- **Build Cache**: 1.045GB (reclaimable)
- **Total Reclaimable**: ~2GB

### Solutions

#### Option 1: Increase EC2 Storage (Recommended)
- **Action**: Increase EBS volume from 50GB to 100GB+
- **Benefit**: Provides sufficient space for Docker builds
- **Steps**: 
  1. Stop EC2 instance
  2. Modify EBS volume size in AWS Console
  3. Extend filesystem: `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1`
  4. Restart instance

#### Option 2: Build Locally and Push
- **Action**: Build Docker image on local machine with more space
- **Steps**:
  1. Build locally: `docker build -f Dockerfile.prod -t diffrhythm:prod .`
  2. Save image: `docker save diffrhythm:prod | gzip > diffrhythm-prod.tar.gz`
  3. Transfer to server: `scp diffrhythm-prod.tar.gz ubuntu@52.0.207.242:/tmp/`
  4. Load on server: `docker load < /tmp/diffrhythm-prod.tar.gz`

#### Option 3: Optimize Dockerfile
- **Action**: Split package installation into smaller batches
- **Benefit**: Reduces peak disk usage
- **Trade-off**: Slower build, but may fit in available space

#### Option 4: Use Docker BuildKit with External Cache
- **Action**: Configure BuildKit to use external cache location
- **Benefit**: Better space management during builds

### Immediate Actions Taken
1. ✅ Cleaned Docker build cache
2. ✅ Removed Docker images
3. ✅ Freed ~2GB of space

### Next Steps
1. **Recommended**: Increase EC2 storage to 100GB+
2. **Alternative**: Build image locally and transfer to server
3. **If keeping current storage**: Optimize Dockerfile for smaller peak usage

### Build Log Location
- `/tmp/docker_build.log` on server
- Last successful step: Built `muq` package
- Failed at: Installing all collected packages

### Disk Space Requirements
- **Minimum for build**: ~15-20GB free space
- **Current available**: ~7.6GB (insufficient)
- **Gap**: Need additional 7-12GB
