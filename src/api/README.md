# FastAPI Service Foundation

This directory contains the FastAPI service foundation for the MLOps California Housing Prediction platform. The service provides a comprehensive, production-ready API with GPU acceleration support, MLflow integration, and advanced monitoring capabilities.

## Components

### Core Modules

- **`config.py`** - Configuration management with environment variable support
- **`main.py`** - Main FastAPI application with middleware and error handling
- **`metrics.py`** - Prometheus metrics integration with GPU monitoring
- **`model_loader.py`** - MLflow Model Registry integration with caching and fallback
- **`health.py`** - Comprehensive health check endpoints
- **`run_server.py`** - Server startup script with command-line options

### Features

#### Configuration Management
- Environment-based configuration with sensible defaults
- Support for API, MLflow, database, and monitoring settings
- Structured logging with JSON format support
- Validation of configuration parameters

#### Prometheus Metrics
- API request metrics (duration, count, status codes)
- Prediction metrics (duration, values, model versions)
- GPU metrics (utilization, memory, temperature, power)
- System metrics and error tracking
- Background monitoring with configurable intervals

#### Model Loading
- MLflow Model Registry integration
- Model caching with TTL support
- Fallback mechanisms across model stages
- Model validation and performance checking
- Thread-safe operations

#### Health Checks
- Basic health status endpoint
- Detailed system information (CPU, memory, disk)
- GPU information (if available)
- Model status and performance metrics
- Dependency health checks (MLflow, database)
- Model reload functionality

#### Error Handling
- Comprehensive exception handling
- Structured error responses
- Request validation with detailed error messages
- Automatic error metrics collection
- Graceful degradation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

Create a `.env` file or set environment variables:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Model Configuration
MODEL_NAME=california-housing-model
MODEL_STAGE=Production

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Monitoring
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=8001
```

### 3. Run the Server

```bash
# Using the run script
python src/api/run_server.py

# With options
python src/api/run_server.py --host 127.0.0.1 --port 9000 --debug --reload

# Using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the Service

```bash
# Run the demo
python examples/fastapi_foundation_demo.py

# Test endpoints
curl http://localhost:8000/health/
curl http://localhost:8000/health/detailed
curl http://localhost:8000/metrics
```

## API Endpoints

### Health Checks
- `GET /health/` - Basic health status
- `GET /health/detailed` - Comprehensive health information
- `GET /health/model` - Model status
- `GET /health/system` - System information
- `GET /health/gpu` - GPU information
- `POST /health/model/reload` - Reload model

### Monitoring
- `GET /metrics` - Prometheus metrics
- `GET /info` - API information

### Documentation
- `GET /docs` - Swagger UI (debug mode only)
- `GET /redoc` - ReDoc documentation (debug mode only)

## Configuration Options

### API Settings
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `API_DEBUG` - Debug mode (default: false)
- `API_VERSION` - API version (default: 1.0.0)

### Model Settings
- `MODEL_NAME` - MLflow model name (default: california-housing-model)
- `MODEL_STAGE` - Model stage (default: Production)
- `MODEL_FALLBACK_STAGE` - Fallback stage (default: Staging)

### MLflow Settings
- `MLFLOW_TRACKING_URI` - Tracking server URI (default: http://localhost:5000)
- `MLFLOW_REGISTRY_URI` - Registry URI (optional)

### Monitoring Settings
- `ENABLE_PROMETHEUS` - Enable metrics (default: true)
- `PROMETHEUS_PORT` - Metrics server port (default: 8001)
- `LOG_LEVEL` - Logging level (default: INFO)

### Performance Settings
- `MAX_BATCH_SIZE` - Maximum batch size (default: 100)
- `REQUEST_TIMEOUT` - Request timeout (default: 30.0)

## Architecture

The FastAPI service follows a modular architecture:

```
FastAPI Application
├── Configuration Layer (config.py)
├── Metrics Layer (metrics.py)
├── Model Layer (model_loader.py)
├── Health Layer (health.py)
└── Application Layer (main.py)
```

### Key Design Patterns

1. **Dependency Injection** - FastAPI's dependency system for configuration and services
2. **Factory Pattern** - For creating configured instances
3. **Singleton Pattern** - For global metrics and model loader instances
4. **Observer Pattern** - For background monitoring
5. **Fallback Pattern** - For model loading and configuration

## Monitoring and Observability

### Prometheus Metrics

The service exposes comprehensive metrics:

- `api_requests_total` - Total API requests by method, endpoint, status
- `api_request_duration_seconds` - Request duration histogram
- `predictions_total` - Total predictions by model version and type
- `prediction_duration_seconds` - Prediction duration histogram
- `gpu_utilization_percent` - GPU utilization (if available)
- `gpu_memory_used_bytes` - GPU memory usage
- `model_status` - Current model status
- `errors_total` - Error count by type and endpoint

### Structured Logging

All operations are logged with structured JSON format including:
- Request/response details
- Performance metrics
- Error information
- System events

### Health Monitoring

Comprehensive health checks cover:
- System resources (CPU, memory, disk)
- GPU status and metrics
- Model availability and performance
- External dependencies (MLflow, database)

## Error Handling

The service implements comprehensive error handling:

1. **HTTP Exceptions** - Proper HTTP status codes and error messages
2. **Validation Errors** - Detailed field-level validation errors
3. **Model Errors** - Graceful handling of model loading/inference failures
4. **System Errors** - Resource and dependency error handling
5. **Fallback Mechanisms** - Automatic fallback for model loading and configuration

## Testing

Run the test suite:

```bash
# Run all API foundation tests
python -m pytest tests/test_api_foundation.py -v

# Run specific test classes
python -m pytest tests/test_api_foundation.py::TestAPIConfig -v
python -m pytest tests/test_api_foundation.py::TestPrometheusMetrics -v
```

## Development

### Adding New Endpoints

1. Create endpoint functions in appropriate modules
2. Add routers to `main.py`
3. Update health checks if needed
4. Add metrics collection
5. Write tests

### Adding New Metrics

1. Define metrics in `metrics.py`
2. Add collection points in relevant modules
3. Update background monitoring if needed
4. Test metric collection

### Configuration Changes

1. Update `APIConfig` or `ModelConfig` in `config.py`
2. Add environment variable support
3. Update validation if needed
4. Update documentation

## Production Deployment

For production deployment:

1. Set `API_DEBUG=false`
2. Configure proper CORS origins
3. Set up reverse proxy (nginx)
4. Configure monitoring and alerting
5. Set up log aggregation
6. Configure model registry access
7. Set resource limits

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check MLflow tracking URI
   - Verify model exists in registry
   - Check model stage configuration

2. **GPU Metrics Not Available**
   - Install nvidia-ml-py: `pip install nvidia-ml-py`
   - Ensure NVIDIA drivers are installed
   - Check GPU accessibility

3. **High Memory Usage**
   - Adjust model cache TTL
   - Monitor background processes
   - Check for memory leaks

4. **Slow Response Times**
   - Enable model caching
   - Check system resources
   - Monitor GPU utilization

### Logs and Debugging

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python src/api/run_server.py --debug
```

Check health endpoints:
```bash
curl http://localhost:8000/health/detailed
```

Monitor metrics:
```bash
curl http://localhost:8000/metrics
```