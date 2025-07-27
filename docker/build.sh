#!/bin/bash
# Docker build script for MLOps California Housing Platform
# Provides easy building and tagging of Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="mlops-california-housing"
REGISTRY="your-registry.com"  # Change this to your registry
VERSION=${1:-latest}
BUILD_TARGET=${2:-production}

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

# Help function
show_help() {
    echo "Usage: $0 [VERSION] [TARGET]"
    echo ""
    echo "Arguments:"
    echo "  VERSION    Docker image version tag (default: latest)"
    echo "  TARGET     Build target (production|development) (default: production)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Build latest production image"
    echo "  $0 v1.0.0                  # Build v1.0.0 production image"
    echo "  $0 latest development       # Build latest development image"
    echo "  $0 v1.0.0 production       # Build v1.0.0 production image"
    echo ""
    echo "Environment Variables:"
    echo "  REGISTRY    Docker registry URL (default: your-registry.com)"
    echo "  NO_CACHE    Set to 'true' to disable build cache"
    echo "  PUSH        Set to 'true' to push image after build"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Validate build target
if [[ "$BUILD_TARGET" != "production" && "$BUILD_TARGET" != "development" ]]; then
    error "Invalid build target: $BUILD_TARGET. Must be 'production' or 'development'"
    exit 1
fi

# Build configuration
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
LOCAL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"

# Build arguments
BUILD_ARGS=""
if [[ "${NO_CACHE:-false}" == "true" ]]; then
    BUILD_ARGS="$BUILD_ARGS --no-cache"
fi

# Pre-build checks
pre_build_checks() {
    log "Running pre-build checks..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if NVIDIA Docker runtime is available (for GPU support)
    if docker info 2>/dev/null | grep -q nvidia; then
        log "NVIDIA Docker runtime detected"
    else
        warn "NVIDIA Docker runtime not detected - GPU features may not work"
    fi
    
    # Check if Dockerfile exists
    if [[ ! -f "Dockerfile" ]]; then
        error "Dockerfile not found in current directory"
        exit 1
    fi
    
    log "Pre-build checks completed"
}

# Build the Docker image
build_image() {
    log "Building Docker image..."
    info "Image: $LOCAL_IMAGE_NAME"
    info "Target: $BUILD_TARGET"
    info "Registry: $REGISTRY"
    
    # Build command
    docker build \
        $BUILD_ARGS \
        --target $BUILD_TARGET \
        --tag $LOCAL_IMAGE_NAME \
        --tag $FULL_IMAGE_NAME \
        --label "version=$VERSION" \
        --label "build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --label "build-target=$BUILD_TARGET" \
        --label "git-commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        .
    
    if [[ $? -eq 0 ]]; then
        log "Docker image built successfully"
    else
        error "Docker image build failed"
        exit 1
    fi
}

# Test the built image
test_image() {
    log "Testing built image..."
    
    # Basic container test
    if docker run --rm $LOCAL_IMAGE_NAME python --version >/dev/null 2>&1; then
        log "Basic container test passed"
    else
        error "Basic container test failed"
        exit 1
    fi
    
    # Health check test (if not development)
    if [[ "$BUILD_TARGET" == "production" ]]; then
        info "Starting container for health check test..."
        CONTAINER_ID=$(docker run -d -p 8001:8000 $LOCAL_IMAGE_NAME)
        
        # Wait for container to start
        sleep 10
        
        # Test health endpoint
        if curl -f http://localhost:8001/health >/dev/null 2>&1; then
            log "Health check test passed"
        else
            warn "Health check test failed - container may need more time to start"
        fi
        
        # Cleanup
        docker stop $CONTAINER_ID >/dev/null 2>&1
        docker rm $CONTAINER_ID >/dev/null 2>&1
    fi
    
    log "Image testing completed"
}

# Push image to registry
push_image() {
    if [[ "${PUSH:-false}" == "true" ]]; then
        log "Pushing image to registry..."
        
        # Login check
        if ! docker info | grep -q "Username:"; then
            warn "Not logged into Docker registry. Run 'docker login $REGISTRY' first"
        fi
        
        # Push image
        docker push $FULL_IMAGE_NAME
        
        if [[ $? -eq 0 ]]; then
            log "Image pushed successfully to $FULL_IMAGE_NAME"
        else
            error "Failed to push image"
            exit 1
        fi
    fi
}

# Show image information
show_image_info() {
    log "Build completed successfully!"
    echo ""
    info "Image Information:"
    echo "  Local Name:    $LOCAL_IMAGE_NAME"
    echo "  Registry Name: $FULL_IMAGE_NAME"
    echo "  Build Target:  $BUILD_TARGET"
    echo "  Version:       $VERSION"
    echo ""
    info "Image Size:"
    docker images $LOCAL_IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    echo ""
    info "To run the container:"
    if [[ "$BUILD_TARGET" == "production" ]]; then
        echo "  docker run -p 8000:8000 --gpus all $LOCAL_IMAGE_NAME"
    else
        echo "  docker run -p 8000:8000 --gpus all -v \$(pwd):/app $LOCAL_IMAGE_NAME"
    fi
    echo ""
    info "To run with Docker Compose:"
    echo "  docker-compose up"
}

# Main execution
main() {
    log "Starting Docker build process..."
    
    pre_build_checks
    build_image
    test_image
    push_image
    show_image_info
    
    log "Docker build process completed successfully!"
}

# Execute main function
main