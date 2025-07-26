#!/usr/bin/env python3
"""
FastAPI Service Foundation Demo

This script demonstrates the FastAPI service foundation components
including configuration, metrics, model loading utilities, and health checks.
"""

import sys
import time
import logging
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from api.config import APIConfig, ModelConfig, setup_logging
from api.metrics import PrometheusMetrics
from api.model_loader import ModelCache
from api.health import get_system_info, get_gpu_info
from api.main import create_app


def demo_configuration():
    """Demonstrate configuration management."""
    print("=" * 60)
    print("CONFIGURATION DEMO")
    print("=" * 60)
    
    # API Configuration
    api_config = APIConfig()
    print(f"API Title: {api_config.title}")
    print(f"API Version: {api_config.version}")
    print(f"Host: {api_config.host}")
    print(f"Port: {api_config.port}")
    print(f"Debug Mode: {api_config.debug}")
    print(f"Model Name: {api_config.model_name}")
    print(f"Model Stage: {api_config.model_stage}")
    print(f"MLflow URI: {api_config.mlflow_tracking_uri}")
    
    # Model Configuration
    model_config = ModelConfig()
    print(f"\nFeature Names: {model_config.feature_names}")
    print(f"Performance Thresholds: {model_config.performance_thresholds}")
    
    # Setup logging
    setup_logging(api_config)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    
    print("\n✅ Configuration demo completed successfully")


def demo_metrics():
    """Demonstrate Prometheus metrics."""
    print("\n" + "=" * 60)
    print("PROMETHEUS METRICS DEMO")
    print("=" * 60)
    
    # Initialize metrics
    metrics = PrometheusMetrics()
    
    # Record some sample metrics
    print("Recording sample metrics...")
    
    # Record API requests
    metrics.record_request("GET", "/health", 200, 0.1)
    metrics.record_request("POST", "/predict", 200, 0.5)
    metrics.record_request("GET", "/health", 404, 0.05)
    
    # Record predictions
    metrics.record_prediction("v1.0", "single", 0.05, 2.5)
    metrics.record_prediction("v1.0", "batch", 0.2, 3.1)
    
    # Set model info
    metrics.set_model_info(
        "california-housing-model", "1.0", "Production", 
        "xgboost", ["MedInc", "HouseAge", "AveRooms"]
    )
    
    # Set model status
    metrics.set_model_status("ready")
    
    # Record errors
    metrics.record_error("validation_error", "/predict")
    
    # Update GPU metrics (if available)
    metrics.update_gpu_metrics()
    
    print("Sample metrics recorded:")
    print("- API requests: Recorded successfully")
    print("- Predictions: Recorded successfully") 
    print("- Errors: Recorded successfully")
    
    # Get metrics in Prometheus format
    metrics_output = metrics.get_metrics()
    print(f"\nMetrics output size: {len(metrics_output)} bytes")
    
    print("\n✅ Metrics demo completed successfully")


def demo_model_cache():
    """Demonstrate model cache functionality."""
    print("\n" + "=" * 60)
    print("MODEL CACHE DEMO")
    print("=" * 60)
    
    # Create cache with short TTL for demo
    cache = ModelCache(ttl_seconds=2)
    
    # Add items to cache
    print("Adding items to cache...")
    cache.put("model_v1", "Mock Model V1")
    cache.put("model_v2", "Mock Model V2")
    
    print(f"Cache size: {cache.size()}")
    
    # Retrieve items
    print("Retrieving items from cache...")
    model_v1 = cache.get("model_v1")
    model_v2 = cache.get("model_v2")
    
    print(f"Retrieved model_v1: {model_v1}")
    print(f"Retrieved model_v2: {model_v2}")
    
    # Test cache miss
    missing = cache.get("nonexistent")
    print(f"Missing item: {missing}")
    
    # Wait for expiration
    print("Waiting for cache expiration...")
    time.sleep(2.5)
    
    expired = cache.get("model_v1")
    print(f"Expired item: {expired}")
    print(f"Cache size after expiration: {cache.size()}")
    
    print("\n✅ Model cache demo completed successfully")


def demo_health_checks():
    """Demonstrate health check functionality."""
    print("\n" + "=" * 60)
    print("HEALTH CHECKS DEMO")
    print("=" * 60)
    
    # System information
    print("Getting system information...")
    system_info = get_system_info()
    
    print(f"Platform: {system_info.platform}")
    print(f"Python Version: {system_info.python_version}")
    print(f"CPU Count: {system_info.cpu_count}")
    print(f"Memory Total: {system_info.memory_total_gb:.2f} GB")
    print(f"Memory Available: {system_info.memory_available_gb:.2f} GB")
    print(f"Memory Usage: {system_info.memory_usage_percent:.1f}%")
    print(f"Disk Usage: {system_info.disk_usage_percent:.1f}%")
    
    # GPU information
    print("\nGetting GPU information...")
    gpu_info = get_gpu_info()
    
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            print(f"GPU {i}: {gpu.name}")
            print(f"  Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB ({gpu.memory_usage_percent:.1f}%)")
            print(f"  Utilization: {gpu.utilization_percent:.1f}%")
            if gpu.temperature_celsius:
                print(f"  Temperature: {gpu.temperature_celsius}°C")
            if gpu.power_usage_watts:
                print(f"  Power Usage: {gpu.power_usage_watts:.1f}W")
    else:
        print("No GPU information available")
    
    print("\n✅ Health checks demo completed successfully")


def demo_fastapi_app():
    """Demonstrate FastAPI application creation."""
    print("\n" + "=" * 60)
    print("FASTAPI APPLICATION DEMO")
    print("=" * 60)
    
    print("Creating FastAPI application...")
    app = create_app()
    
    print(f"App title: {app.title}")
    print(f"App version: {app.version}")
    print(f"App description: {app.description}")
    
    # List available routes
    print("\nAvailable routes:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ', '.join(route.methods)
            print(f"  {methods} {route.path}")
    
    print("\n✅ FastAPI application demo completed successfully")


def main():
    """Run all demos."""
    print("FastAPI Service Foundation Demo")
    print("This demo showcases the core components of the MLOps FastAPI service")
    
    try:
        demo_configuration()
        demo_metrics()
        demo_model_cache()
        demo_health_checks()
        demo_fastapi_app()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe FastAPI service foundation is working correctly.")
        print("You can now:")
        print("1. Start the server with: python src/api/run_server.py")
        print("2. Access health checks at: http://localhost:8000/health/")
        print("3. View API docs at: http://localhost:8000/docs")
        print("4. Check metrics at: http://localhost:8000/metrics")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()