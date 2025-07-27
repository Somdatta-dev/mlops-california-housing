# Multi-stage Dockerfile for MLOps California Housing Prediction API
# Optimized for production deployment with CUDA support and minimal image size

# ============================================================================
# Stage 1: Base CUDA environment with Python
# ============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS cuda-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    curl \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Dependencies installation
# ============================================================================
FROM cuda-base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support first
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install other Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn[gthread]==21.2.0

# ============================================================================
# Stage 3: Application build
# ============================================================================
FROM dependencies AS app-build

# Copy application source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh ./entrypoint.sh
COPY .env.template ./.env.template

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/plots /app/logs /app/models

# Set proper permissions
RUN chmod +x scripts/*.py 2>/dev/null || true && \
    chmod +x entrypoint.sh

# ============================================================================
# Stage 4: Production runtime
# ============================================================================
FROM app-build AS production

# Create non-root user for security
RUN groupadd -r mlops && useradd -r -g mlops -d /app -s /bin/bash mlops

# Set ownership of application directory
RUN chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint for proper signal handling
ENTRYPOINT ["./entrypoint.sh"]

# Default command with proper signal handling
CMD ["gunicorn", "src.api.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--worker-connections", "1000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--preload"]

# ============================================================================
# Stage 5: Development runtime (optional)
# ============================================================================
FROM app-build AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    ipython==8.17.2 \
    debugpy==1.8.0

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops -d /app -s /bin/bash mlops
RUN chown -R mlops:mlops /app
USER mlops

WORKDIR /app
EXPOSE 8000

# Set entrypoint for proper signal handling
ENTRYPOINT ["./entrypoint.sh"]

# Development command with hot reload
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]