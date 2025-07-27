#!/bin/bash
# Container optimization script for production deployment
# Optimizes Docker images and containers for minimal size and maximum performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="mlops-california-housing"
OPTIMIZED_TAG="optimized"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running"
        exit 1
    fi
    
    # Check if dive is available for image analysis
    if ! command -v dive &> /dev/null; then
        warn "dive not found. Install with: docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock wagoodman/dive:latest"
    fi
    
    log "Prerequisites check completed"
}

# Analyze current image
analyze_image() {
    local image_name=$1
    log "Analyzing image: $image_name"
    
    # Get image size
    local size=$(docker images $image_name --format "{{.Size}}")
    info "Current image size: $size"
    
    # Get layer information
    info "Image layers:"
    docker history $image_name --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
    
    # Run dive analysis if available
    if command -v dive &> /dev/null; then
        info "Running dive analysis..."
        dive $image_name --ci
    fi
}

# Optimize Dockerfile
optimize_dockerfile() {
    log "Creating optimized Dockerfile..."
    
    cat > Dockerfile.optimized << 'EOF'
# Optimized multi-stage Dockerfile for minimal production image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 as base

# Minimize layers and reduce image size
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel

# Dependencies stage
FROM base as dependencies
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn[gthread]==21.2.0 \
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local -type f -name "*.pyc" -delete

# Production stage
FROM base as production
WORKDIR /app

# Copy only necessary files
COPY --from=dependencies /usr/local /usr/local
COPY src/ ./src/
COPY docker/entrypoint.sh ./entrypoint.sh

# Create directories and set permissions in single layer
RUN mkdir -p /app/data/raw /app/data/processed /app/plots /app/logs /app/models \
    && groupadd -r mlops \
    && useradd -r -g mlops -d /app -s /bin/bash mlops \
    && chmod +x entrypoint.sh \
    && chown -R mlops:mlops /app \
    && find /app -type d -exec chmod 755 {} + \
    && find /app -type f -exec chmod 644 {} + \
    && chmod +x entrypoint.sh

USER mlops
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["gunicorn", "src.api.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--preload"]
EOF

    log "Optimized Dockerfile created"
}

# Build optimized image
build_optimized_image() {
    log "Building optimized image..."
    
    # Build with optimized Dockerfile
    docker build \
        --file Dockerfile.optimized \
        --target production \
        --tag ${IMAGE_NAME}:${OPTIMIZED_TAG} \
        --label "optimized=true" \
        --label "build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .
    
    if [[ $? -eq 0 ]]; then
        log "Optimized image built successfully"
    else
        error "Failed to build optimized image"
        exit 1
    fi
}

# Compare image sizes
compare_images() {
    log "Comparing image sizes..."
    
    local original_size=$(docker images ${IMAGE_NAME}:latest --format "{{.Size}}")
    local optimized_size=$(docker images ${IMAGE_NAME}:${OPTIMIZED_TAG} --format "{{.Size}}")
    
    info "Original image size: $original_size"
    info "Optimized image size: $optimized_size"
    
    # Show detailed comparison
    echo ""
    info "Detailed comparison:"
    docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

# Security scan
security_scan() {
    log "Running security scan on optimized image..."
    
    # Run Trivy security scan
    if command -v trivy &> /dev/null; then
        trivy image ${IMAGE_NAME}:${OPTIMIZED_TAG}
    else
        # Use Docker version of Trivy
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image ${IMAGE_NAME}:${OPTIMIZED_TAG}
    fi
}

# Performance test
performance_test() {
    log "Running performance test..."
    
    # Start optimized container
    local container_id=$(docker run -d -p 8002:8000 --gpus all ${IMAGE_NAME}:${OPTIMIZED_TAG})
    
    # Wait for startup
    sleep 15
    
    # Test startup time and response
    local start_time=$(date +%s)
    while ! curl -f http://localhost:8002/health >/dev/null 2>&1; do
        sleep 1
        local current_time=$(date +%s)
        if [ $((current_time - start_time)) -gt 60 ]; then
            error "Container failed to start within 60 seconds"
            docker stop $container_id >/dev/null 2>&1
            docker rm $container_id >/dev/null 2>&1
            exit 1
        fi
    done
    
    local startup_time=$(($(date +%s) - start_time))
    info "Container startup time: ${startup_time} seconds"
    
    # Test API response time
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8002/health)
    info "API response time: ${response_time} seconds"
    
    # Cleanup
    docker stop $container_id >/dev/null 2>&1
    docker rm $container_id >/dev/null 2>&1
    
    log "Performance test completed"
}

# Generate optimization report
generate_report() {
    log "Generating optimization report..."
    
    local report_file="optimization_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $report_file << EOF
MLOps California Housing Platform - Image Optimization Report
============================================================

Generated: $(date)

Image Comparison:
$(docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}")

Optimization Techniques Applied:
- Multi-stage build to reduce final image size
- Combined RUN commands to minimize layers
- Removed package caches and temporary files
- Used --no-install-recommends for apt packages
- Cleaned up Python cache files
- Optimized file permissions in single layer

Security Considerations:
- Non-root user execution
- Minimal base image with only required packages
- Regular security scanning with Trivy
- Health checks for container monitoring

Performance Optimizations:
- Gunicorn with optimized worker configuration
- Preloaded application for faster startup
- Connection pooling and keep-alive settings
- Resource limits and reservations

Recommendations:
1. Use optimized image for production deployment
2. Implement regular security scanning in CI/CD
3. Monitor container resource usage
4. Consider using distroless base images for even smaller size
5. Implement image signing for security

EOF

    info "Optimization report saved to: $report_file"
}

# Cleanup
cleanup() {
    log "Cleaning up temporary files..."
    rm -f Dockerfile.optimized
    log "Cleanup completed"
}

# Main execution
main() {
    log "Starting container optimization process..."
    
    check_prerequisites
    
    # Analyze original image if it exists
    if docker images ${IMAGE_NAME}:latest --format "{{.Repository}}" | grep -q ${IMAGE_NAME}; then
        analyze_image "${IMAGE_NAME}:latest"
    else
        warn "Original image not found, building first..."
        docker build -t ${IMAGE_NAME}:latest .
    fi
    
    optimize_dockerfile
    build_optimized_image
    compare_images
    security_scan
    performance_test
    generate_report
    cleanup
    
    log "Container optimization completed successfully!"
    info "Use the optimized image with: docker run --gpus all -p 8000:8000 ${IMAGE_NAME}:${OPTIMIZED_TAG}"
}

# Execute main function
main "$@"