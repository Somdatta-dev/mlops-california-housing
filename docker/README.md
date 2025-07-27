# Docker Configuration for MLOps California Housing Platform

This directory contains Docker configurations for containerizing the MLOps California Housing Prediction Platform with CUDA support, monitoring, and production optimizations.

## Overview

The Docker setup provides:
- **Multi-stage builds** for optimized production images
- **NVIDIA CUDA support** for GPU-accelerated model training and inference
- **Service orchestration** with Docker Compose
- **Production-ready configurations** with health checks and monitoring
- **Development environment** with hot reload and debugging tools

## Files Structure

```
docker/
├── README.md                 # This documentation
├── build.sh                  # Build script for Docker images
├── entrypoint.sh            # Container entrypoint with signal handling
├── nginx/
│   └── nginx.conf           # Nginx load balancer configuration
└── prometheus/
    ├── prometheus.yml       # Prometheus monitoring configuration
    └── alert_rules.yml      # Prometheus alerting rules
```

## Prerequisites

### System Requirements
- Docker Engine 20.10+ with Docker Compose v2
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU with CUDA 12.8+ support
- At least 8GB RAM and 4GB GPU memory
- PyTorch 2.7.0 with CUDA 12.8 support

### NVIDIA Docker Setup

1. **Install NVIDIA Container Toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Build the Docker Image

```bash
# Build production image
./docker/build.sh

# Build development image
./docker/build.sh latest development

# Build with specific version
./docker/build.sh v1.0.0 production
```

### 2. Run with Docker Compose

```bash
# Start all services (production)
docker-compose up -d

# Start development environment
docker-compose --profile development up -d

# Start with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. Access Services

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=info

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000

# Database Configuration
DATABASE_URL=sqlite:///app/data/mlops_platform.db

# Monitoring Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Production Configuration (for prod deployment)
POSTGRES_PASSWORD=your_secure_password
GRAFANA_ADMIN_PASSWORD=your_secure_password
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### Docker Compose Profiles

- **Default**: Production services (API, MLflow, Prometheus, Grafana)
- **Development**: Includes Jupyter, Adminer, and development API
- **Production**: Optimized for production with PostgreSQL, Nginx, SSL

## Usage Examples

### Development Workflow

```bash
# Start development environment
docker-compose --profile development up -d

# View logs
docker-compose logs -f mlops-api

# Execute commands in container
docker-compose exec mlops-api python -m pytest

# Access Jupyter notebook
# Navigate to http://localhost:8888
```

### Production Deployment

```bash
# Build production image
./docker/build.sh v1.0.0 production

# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale API service
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale mlops-api=3
```

### GPU Monitoring

```bash
# Check GPU usage in container
docker-compose exec mlops-api nvidia-smi

# View GPU metrics in Prometheus
# Navigate to http://localhost:9090/graph
# Query: nvidia_gpu_utilization_percent
```

## Health Checks and Monitoring

### Container Health Checks

All services include health checks:
- **API**: HTTP health endpoint check every 30s
- **MLflow**: HTTP health endpoint check every 30s
- **Prometheus**: HTTP ready endpoint check every 30s
- **Grafana**: HTTP health API check every 30s

### Monitoring Stack

1. **Prometheus** collects metrics from:
   - API endpoints and performance
   - GPU utilization and memory
   - System resources
   - MLflow server status

2. **Grafana** provides dashboards for:
   - API performance and request rates
   - GPU utilization and temperature
   - Model prediction metrics
   - System health overview

3. **Alerting** configured for:
   - High GPU utilization (>90%)
   - API errors and high latency
   - System resource exhaustion
   - Service downtime

## Troubleshooting

### Common Issues

1. **GPU not accessible in container:**
```bash
# Check NVIDIA Docker runtime
docker info | grep nvidia

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

2. **Container fails to start:**
```bash
# Check logs
docker-compose logs mlops-api

# Check resource usage
docker stats

# Verify environment variables
docker-compose config
```

3. **Model loading failures:**
```bash
# Check MLflow connection
docker-compose exec mlops-api curl http://mlflow:5000/health

# Verify model files
docker-compose exec mlops-api ls -la /app/models/
```

4. **High memory usage:**
```bash
# Monitor container resources
docker stats

# Check GPU memory
docker-compose exec mlops-api nvidia-smi

# Restart services
docker-compose restart mlops-api
```

### Performance Optimization

1. **Image Size Optimization:**
   - Multi-stage builds reduce final image size
   - .dockerignore excludes unnecessary files
   - Alpine-based images where possible

2. **Runtime Optimization:**
   - Gunicorn with multiple workers
   - Connection pooling and keep-alive
   - Resource limits and reservations

3. **GPU Optimization:**
   - CUDA memory management
   - Mixed precision training
   - Batch processing for inference

## Security Considerations

### Production Security

1. **Container Security:**
   - Non-root user execution
   - Read-only root filesystem where possible
   - Security scanning with tools like Trivy

2. **Network Security:**
   - Internal Docker network isolation
   - Nginx reverse proxy with rate limiting
   - SSL/TLS termination

3. **Secrets Management:**
   - Environment variables for configuration
   - Docker secrets for sensitive data
   - External secret management integration

### Security Scanning

```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image mlops-california-housing:latest

# Check for security best practices
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  docker/docker-bench-security
```

## Maintenance

### Regular Tasks

1. **Image Updates:**
```bash
# Rebuild with latest base images
./docker/build.sh latest production --no-cache

# Update dependencies
docker-compose pull
```

2. **Log Management:**
```bash
# View logs
docker-compose logs --tail=100 -f

# Clean up logs
docker system prune -f
```

3. **Backup:**
```bash
# Backup volumes
docker run --rm -v mlflow_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/mlflow_backup.tar.gz /data

# Backup database
docker-compose exec mlops-api sqlite3 /app/data/mlops_platform.db ".backup /app/data/backup.db"
```

## Support

For issues and questions:
1. Check container logs: `docker-compose logs [service]`
2. Verify GPU access: `docker-compose exec mlops-api nvidia-smi`
3. Check service health: `docker-compose ps`
4. Review monitoring dashboards in Grafana
5. Consult the main project documentation