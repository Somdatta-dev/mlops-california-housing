#!/bin/bash
# Docker entrypoint script for MLOps California Housing API
# Provides proper signal handling, health checks, and graceful shutdown

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Signal handlers
cleanup() {
    log "Received shutdown signal, performing graceful shutdown..."
    
    # Kill background processes
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$HEALTH_CHECK_PID" ]; then
        kill $HEALTH_CHECK_PID 2>/dev/null || true
    fi
    
    # Kill main application
    if [ ! -z "$APP_PID" ]; then
        log "Stopping main application (PID: $APP_PID)..."
        kill -TERM $APP_PID 2>/dev/null || true
        
        # Wait for graceful shutdown
        local count=0
        while kill -0 $APP_PID 2>/dev/null && [ $count -lt 30 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 $APP_PID 2>/dev/null; then
            warn "Application didn't shutdown gracefully, forcing termination..."
            kill -KILL $APP_PID 2>/dev/null || true
        fi
    fi
    
    log "Shutdown complete"
    exit 0
}

# Set up signal traps
trap cleanup SIGTERM SIGINT SIGQUIT

# Environment validation
validate_environment() {
    log "Validating environment..."
    
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    else
        warn "NVIDIA GPU not detected, running in CPU mode"
    fi
    
    # Check Python environment
    python --version
    
    # Check required directories
    mkdir -p /app/data/raw /app/data/processed /app/plots /app/logs /app/models
    
    # Set permissions
    chmod -R 755 /app/data /app/plots /app/logs /app/models 2>/dev/null || true
    
    log "Environment validation complete"
}

# Health check function
health_check() {
    while true; do
        sleep 30
        if ! curl -f http://localhost:8000/health >/dev/null 2>&1; then
            error "Health check failed"
        fi
    done
}

# GPU monitoring function
gpu_monitor() {
    while true; do
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits > /tmp/gpu_stats.txt 2>/dev/null || true
        fi
        sleep 10
    done
}

# Database initialization
init_database() {
    log "Initializing database..."
    
    # Run database migrations if needed
    if [ -f "/app/src/api/database_init.py" ]; then
        python -m src.api.database_init || warn "Database initialization failed"
    fi
    
    log "Database initialization complete"
}

# Model loading check
check_models() {
    log "Checking model availability..."
    
    # Check if models directory exists and has content
    if [ -d "/app/models" ] && [ "$(ls -A /app/models 2>/dev/null)" ]; then
        log "Models found in /app/models"
    else
        warn "No models found, will attempt to load from MLflow"
    fi
}

# Pre-flight checks
preflight_checks() {
    log "Running pre-flight checks..."
    
    validate_environment
    init_database
    check_models
    
    log "Pre-flight checks complete"
}

# Main execution
main() {
    log "Starting MLOps California Housing API container..."
    
    # Run pre-flight checks
    preflight_checks
    
    # Start background monitoring
    if [ "${ENABLE_GPU_MONITORING:-true}" = "true" ]; then
        gpu_monitor &
        GPU_MONITOR_PID=$!
        log "GPU monitoring started (PID: $GPU_MONITOR_PID)"
    fi
    
    if [ "${ENABLE_HEALTH_CHECK:-true}" = "true" ]; then
        health_check &
        HEALTH_CHECK_PID=$!
        log "Health check monitoring started (PID: $HEALTH_CHECK_PID)"
    fi
    
    # Start the main application
    log "Starting main application..."
    
    # Default to production command if no arguments provided
    if [ $# -eq 0 ]; then
        set -- gunicorn src.api.main:app \
            --bind 0.0.0.0:8000 \
            --workers ${WORKERS:-1} \
            --worker-class uvicorn.workers.UvicornWorker \
            --worker-connections ${WORKER_CONNECTIONS:-1000} \
            --max-requests ${MAX_REQUESTS:-1000} \
            --max-requests-jitter ${MAX_REQUESTS_JITTER:-100} \
            --timeout ${TIMEOUT:-120} \
            --keep-alive ${KEEP_ALIVE:-5} \
            --graceful-timeout ${GRACEFUL_TIMEOUT:-30} \
            --access-logfile - \
            --error-logfile - \
            --log-level ${LOG_LEVEL:-info} \
            --preload
    fi
    
    # Execute the main command
    exec "$@" &
    APP_PID=$!
    
    log "Main application started (PID: $APP_PID)"
    log "Container startup complete"
    
    # Wait for the main process
    wait $APP_PID
}

# Execute main function with all arguments
main "$@"