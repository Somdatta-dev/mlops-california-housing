"""
Health Check Endpoints

This module provides comprehensive health check endpoints for the FastAPI service,
including system status, model availability, and dependency checks.
"""

import os
import time
import logging
import platform
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from .config import APIConfig, get_api_config
from .model_loader import ModelLoader, get_model_loader
from .metrics import PrometheusMetrics, get_metrics

try:
    import nvidia_ml_py as nvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    nvml = None

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")


class SystemInfo(BaseModel):
    """System information model."""
    platform: str = Field(..., description="Operating system platform")
    python_version: str = Field(..., description="Python version")
    cpu_count: int = Field(..., description="Number of CPU cores")
    memory_total_gb: float = Field(..., description="Total system memory in GB")
    memory_available_gb: float = Field(..., description="Available system memory in GB")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")


class GPUInfo(BaseModel):
    """GPU information model."""
    gpu_id: int = Field(..., description="GPU device ID")
    name: str = Field(..., description="GPU name")
    memory_total_mb: float = Field(..., description="Total GPU memory in MB")
    memory_used_mb: float = Field(..., description="Used GPU memory in MB")
    memory_usage_percent: float = Field(..., description="GPU memory usage percentage")
    utilization_percent: float = Field(..., description="GPU utilization percentage")
    temperature_celsius: Optional[float] = Field(None, description="GPU temperature in Celsius")
    power_usage_watts: Optional[float] = Field(None, description="GPU power usage in watts")


class ModelStatus(BaseModel):
    """Model status information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage")
    model_type: str = Field(..., description="Model type")
    load_time: datetime = Field(..., description="Model load timestamp")
    features: List[str] = Field(..., description="Model feature names")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")


class DependencyStatus(BaseModel):
    """Dependency status information."""
    name: str = Field(..., description="Dependency name")
    status: str = Field(..., description="Dependency status")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if dependency is unhealthy")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")
    system: SystemInfo = Field(..., description="System information")
    gpu: Optional[List[GPUInfo]] = Field(None, description="GPU information")
    model: Optional[ModelStatus] = Field(None, description="Model status")
    dependencies: List[DependencyStatus] = Field(..., description="Dependency status")


# Service start time for uptime calculation
_service_start_time = time.time()


def get_system_info() -> SystemInfo:
    """Get system information."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            cpu_count=psutil.cpu_count(),
            memory_total_gb=memory.total / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent
        )
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")


def get_gpu_info() -> Optional[List[GPUInfo]]:
    """Get GPU information if available."""
    if not NVIDIA_ML_AVAILABLE:
        return None
    
    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        gpu_info = []
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get basic info
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get optional info
            temperature = None
            power_usage = None
            
            try:
                temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            
            try:
                power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except Exception:
                pass
            
            gpu_info.append(GPUInfo(
                gpu_id=i,
                name=name,
                memory_total_mb=memory_info.total / (1024**2),
                memory_used_mb=memory_info.used / (1024**2),
                memory_usage_percent=(memory_info.used / memory_info.total) * 100,
                utilization_percent=utilization.gpu,
                temperature_celsius=temperature,
                power_usage_watts=power_usage
            ))
        
        return gpu_info
        
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


def get_model_status(model_loader: ModelLoader) -> Optional[ModelStatus]:
    """Get current model status."""
    try:
        model, model_info = model_loader.get_current_model()
        
        if model_info is None:
            return None
        
        return ModelStatus(
            name=model_info.name,
            version=model_info.version,
            stage=model_info.stage,
            model_type=model_info.model_type,
            load_time=model_info.load_time,
            features=model_info.features,
            performance_metrics=model_info.performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return None


def check_mlflow_dependency(config: APIConfig) -> DependencyStatus:
    """Check MLflow tracking server dependency."""
    import requests
    
    start_time = time.time()
    
    try:
        # Try to connect to MLflow tracking server
        response = requests.get(
            f"{config.mlflow_tracking_uri}/health",
            timeout=5.0
        )
        
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            return DependencyStatus(
                name="mlflow_tracking",
                status="healthy",
                response_time_ms=response_time
            )
        else:
            return DependencyStatus(
                name="mlflow_tracking",
                status="unhealthy",
                response_time_ms=response_time,
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return DependencyStatus(
            name="mlflow_tracking",
            status="unhealthy",
            response_time_ms=response_time,
            error=str(e)
        )


def check_database_dependency(config: APIConfig) -> DependencyStatus:
    """Check database dependency."""
    start_time = time.time()
    
    try:
        # For SQLite, just check if we can create a connection
        if config.database_url.startswith("sqlite"):
            import sqlite3
            db_path = config.database_url.replace("sqlite:///", "")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1")
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            return DependencyStatus(
                name="database",
                status="healthy",
                response_time_ms=response_time
            )
        else:
            # For other databases, you would implement appropriate checks
            return DependencyStatus(
                name="database",
                status="unknown",
                error="Database type not supported for health check"
            )
            
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return DependencyStatus(
            name="database",
            status="unhealthy",
            response_time_ms=response_time,
            error=str(e)
        )


@router.get("/", response_model=HealthStatus)
async def basic_health_check(config: APIConfig = Depends(get_api_config)) -> HealthStatus:
    """
    Basic health check endpoint.
    
    Returns basic service health status without detailed information.
    """
    uptime = time.time() - _service_start_time
    
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        version=config.version
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    config: APIConfig = Depends(get_api_config)
) -> DetailedHealthResponse:
    """
    Detailed health check endpoint.
    
    Returns comprehensive health information including system status,
    GPU information, model status, and dependency checks.
    """
    uptime = time.time() - _service_start_time
    
    # Get system information
    system_info = get_system_info()
    
    # Get GPU information
    gpu_info = get_gpu_info()
    
    # Get model status
    try:
        model_loader = get_model_loader()
        model_status = get_model_status(model_loader)
    except Exception as e:
        logger.warning(f"Failed to get model status: {e}")
        model_status = None
    
    # Check dependencies
    dependencies = [
        check_mlflow_dependency(config),
        check_database_dependency(config)
    ]
    
    # Determine overall status
    overall_status = "healthy"
    
    # Check if any critical dependencies are unhealthy
    for dep in dependencies:
        if dep.name in ["database"] and dep.status == "unhealthy":
            overall_status = "degraded"
        elif dep.name == "mlflow_tracking" and dep.status == "unhealthy":
            overall_status = "degraded"
    
    # Check if model is unavailable
    if model_status is None:
        overall_status = "degraded"
    
    # Check system resources
    if system_info.memory_usage_percent > 90:
        overall_status = "degraded"
    
    if system_info.disk_usage_percent > 95:
        overall_status = "unhealthy"
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        version=config.version,
        system=system_info,
        gpu=gpu_info,
        model=model_status,
        dependencies=dependencies
    )


@router.get("/model", response_model=Optional[ModelStatus])
async def model_health_check() -> Optional[ModelStatus]:
    """
    Model-specific health check endpoint.
    
    Returns information about the currently loaded model.
    """
    try:
        model_loader = get_model_loader()
        return get_model_status(model_loader)
    except Exception as e:
        logger.warning(f"Failed to get model status: {e}")
        return None


@router.get("/system", response_model=SystemInfo)
async def system_health_check() -> SystemInfo:
    """
    System-specific health check endpoint.
    
    Returns system resource information.
    """
    return get_system_info()


@router.get("/gpu", response_model=Optional[List[GPUInfo]])
async def gpu_health_check() -> Optional[List[GPUInfo]]:
    """
    GPU-specific health check endpoint.
    
    Returns GPU information if available.
    """
    return get_gpu_info()


@router.post("/model/reload")
async def reload_model(force: bool = False) -> Dict[str, Any]:
    """
    Reload the current model.
    
    Args:
        force: Whether to force reload even if current model exists
    
    Returns:
        Model reload status
    """
    try:
        start_time = time.time()
        
        model_loader = get_model_loader()
        model, model_info = model_loader.reload_model(force=force)
        
        reload_time = time.time() - start_time
        
        try:
            metrics = get_metrics()
            metrics.record_model_load_time(reload_time)
        except Exception:
            pass  # Metrics not available
        
        return {
            "status": "success",
            "message": f"Model {model_info.name} v{model_info.version} reloaded successfully",
            "model_info": {
                "name": model_info.name,
                "version": model_info.version,
                "stage": model_info.stage,
                "model_type": model_info.model_type,
                "load_time": model_info.load_time.isoformat(),
                "reload_duration_seconds": reload_time
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        
        try:
            metrics = get_metrics()
            metrics.record_error("model_reload_failed", "/health/model/reload")
        except Exception:
            pass  # Metrics not available
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )