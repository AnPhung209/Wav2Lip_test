# Docker Deployment Guide
## Real-Time Lip-Syncing WebSocket API

This guide ensures the application **runs entirely inside Docker** with all dependencies and models.

## **Pre-Deployment Checklist**

### **1. Required Files**
```bash
# Verify these files exist:
Dockerfile                   # Container definition
docker-compose.yml           # Service orchestration  
.dockerignore                # Build optimization
requirements_updated.txt     # Python dependencies
app/                         # Application code
static/                      # Web interface
wav2lip/                     # AI model code
```

### **2. Model Files Setup**
```bash
# Download Wav2Lip models to checkpoints/
mkdir -p checkpoints

# Required model (choose one):
# - Wav2Lip-SD-NOGAN.pt (139MB) - Better accuracy 
# - Wav2Lip-SD-GAN.pt (139MB) - Better visual quality

# Verify model exists:
ls -la checkpoints/
```

## **Docker Deployment**

### **Docker Compose (Recommended)**

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### **Manual Docker Build (Alternative)**

```bash
# Build the image
docker build -t lip-sync-api .

# Run the container
docker run -d \
  --name lip-sync-api \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  --memory=3g \
  lip-sync-api

# Check logs
docker logs -f lip-sync-api

# Stop the container
docker stop lip-sync-api
docker rm lip-sync-api
```

## **Testing Docker Deployment**

### **Automatic Testing**
```bash
# Run comprehensive Docker tests
python test_docker.py

# Expected output:
# Health Check: PASS
# Web Interface: PASS  
# API Endpoints: PASS
# WebSocket: PASS
```

### **Manual Testing**
```bash
# 1. Test health endpoint
curl http://localhost:8000/health

# 2. Test web interface
open http://localhost:8000

# 3. Test API documentation
open http://localhost:8000/docs

# 4. Test WebSocket (using provided client)
python test_websocket_client.py
```

## **Docker Configuration Details**

### **Container Specifications**
- **Base Image**: `python:3.11-slim`
- **Memory**: 3GB limit, 1GB reserved
- **Port**: 8000 (HTTP/WebSocket)
- **Health Check**: Every 30s with 90s startup time
- **Restart Policy**: `unless-stopped`

### **Volume Mounts**
```yaml
volumes:
  - ./checkpoints:/app/checkpoints  # Model storage
```

**What gets mounted:**
- **checkpoints/** - Contains the Wav2Lip model files
- **Test files** (Donald_Trump_official_portrait.jpg, ttsMP3.com_VoiceText_2025-6-3_12-18-2.mp3) are copied into the container during build

### **Environment Variables**
```yaml
environment:
  - PYTHONPATH=/app
  - PYTHONUNBUFFERED=1
```

## **Troubleshooting**

### **Common Issues and Solutions**

#### **1. Container Fails to Start**
```bash
# Check logs
docker-compose logs lip-sync-api

# Common causes:
# - Missing model files
# - Insufficient memory
# - Port 8000 already in use
```

#### **2. Model Loading Fails**
```bash
# Verify model files
docker exec -it illuminus_lip-sync-api_1 ls -la /app/checkpoints/

# Check model permissions
docker exec -it illuminus_lip-sync-api_1 file /app/checkpoints/*.pt
```

#### **3. Health Check Fails**
```bash
# Check health status
docker ps

# Increase startup time if needed (in docker-compose.yml):
start_period: 120s  # For slow systems
```

#### **4. Memory Issues**
```bash
# Monitor memory usage
docker stats

# Increase memory limit (in docker-compose.yml):
limits:
  memory: 4G  # Increase if needed
```

## **Performance Optimization**

### **For Production Deployment**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  lip-sync-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    volumes:
      - ./checkpoints:/app/checkpoints:ro  # Read-only
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 60s
      timeout: 30s
      retries: 3
      start_period: 120s
```

### **For GPU Support** (Optional)
```dockerfile
# Add to Dockerfile for GPU support:
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Add to docker-compose.yml:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## **Verification Checklist**

After deployment, verify these items:

- [ ] **Container Status**: `docker ps` shows healthy container
- [ ] **Health Check**: `curl http://localhost:8000/health` returns 200
- [ ] **Web Interface**: Browser shows "Lip Sync Magic" interface  
- [ ] **WebSocket**: Can connect to `ws://localhost:8000/ws`
- [ ] **Model Loading**: Health endpoint shows `wav2lip_processor: true`
- [ ] **File Upload**: Web interface accepts image/audio files
- [ ] **Processing**: Can create lip-sync videos successfully

## **Success Criteria**

The application **runs entirely inside Docker** when:

**No local Python environment needed**  
**All dependencies included in container**  
**Models load automatically inside container**  
**Web interface accessible from host**  
**WebSocket API fully functional**  
**Can process real audio/image files**  
**Health checks pass consistently**  

## **Quick Commands Reference**

```bash
# Build and start
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Test deployment
python test_docker.py

# Clean up
docker-compose down
docker system prune
```

---
