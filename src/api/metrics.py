"""
Prometheus Metrics Integration

This module provides comprehensive Prometheus metrics collection for the FastAPI service,
including prediction metrics, system metrics, and GPU monitoring.
"""

import time
import logging
import threading
import asyncio
import schedule
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timedelta

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server
)

try:
    import nvidia_ml_py as nvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    nvml = None

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Comprehensive Prometheus metrics collector for the MLOps API.
    
    This class provides metrics collection for predictions, system performance,
    GPU utilization, and API operations with automatic GPU monitoring.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics collector.
        
        Args:
            registry: Optional custom registry. If None, uses default registry.
        """
        from prometheus_client import REGISTRY
        self.registry = registry or REGISTRY
        self._gpu_available = False
        self._gpu_handles = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._scheduled_tasks = []
        self._task_scheduler_thread = None
        self._last_metrics_update = datetime.now()
        
        # Initialize GPU monitoring
        self._init_gpu_monitoring()
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize custom metrics for model performance and system health
        self._init_custom_metrics()
        
        logger.info("Prometheus metrics initialized")
    
    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring if available."""
        if not NVIDIA_ML_AVAILABLE:
            logger.warning("nvidia-ml-py not available, GPU metrics disabled")
            return
        
        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                self._gpu_handles.append(handle)
            
            self._gpu_available = True
            logger.info(f"GPU monitoring initialized for {device_count} devices")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            self._gpu_available = False
    
    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        
        # API Request Metrics
        self.requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Prediction Metrics
        self.predictions_total = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['model_version', 'prediction_type'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'prediction_duration_seconds',
            'Time spent on model predictions',
            ['model_version'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.prediction_values = Histogram(
            'prediction_values',
            'Distribution of prediction values',
            ['model_version'],
            buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0],
            registry=self.registry
        )
        
        # Model Metrics
        self.model_info = Info(
            'model_info',
            'Information about the loaded model',
            registry=self.registry
        )
        
        self.model_load_duration = Histogram(
            'model_load_duration_seconds',
            'Time spent loading models',
            registry=self.registry
        )
        
        self.model_status = Enum(
            'model_status',
            'Current model status',
            states=['loading', 'ready', 'error', 'unavailable'],
            registry=self.registry
        )
        
        # System Metrics
        self.system_info = Info(
            'system_info',
            'System information',
            registry=self.registry
        )
        
        # GPU Metrics (always initialize, but only update if available)
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_total = Gauge(
            'gpu_memory_total_bytes',
            'Total GPU memory in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_power_usage = Gauge(
            'gpu_power_usage_watts',
            'GPU power usage in watts',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        
        # Database Metrics
        self.database_operations_total = Counter(
            'database_operations_total',
            'Total database operations',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.database_operation_duration = Histogram(
            'database_operation_duration_seconds',
            'Database operation duration',
            ['operation', 'table'],
            registry=self.registry
        )
    
    def _init_custom_metrics(self) -> None:
        """Initialize custom metrics for model performance and system health."""
        
        # Model Performance Metrics
        self.model_accuracy = Gauge(
            'model_accuracy_score',
            'Current model accuracy score',
            ['model_version', 'dataset'],
            registry=self.registry
        )
        
        self.model_rmse = Gauge(
            'model_rmse_score',
            'Current model RMSE score',
            ['model_version', 'dataset'],
            registry=self.registry
        )
        
        self.model_mae = Gauge(
            'model_mae_score',
            'Current model MAE score',
            ['model_version', 'dataset'],
            registry=self.registry
        )
        
        self.model_r2 = Gauge(
            'model_r2_score',
            'Current model R² score',
            ['model_version', 'dataset'],
            registry=self.registry
        )
        
        # System Health Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_memory_total = Gauge(
            'system_memory_total_bytes',
            'Total system memory in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        # API Health Metrics
        self.api_health_status = Gauge(
            'api_health_status',
            'API health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections_count',
            'Number of active connections',
            registry=self.registry
        )
        
        # Model Performance Over Time
        self.prediction_latency_p95 = Gauge(
            'prediction_latency_p95_seconds',
            '95th percentile prediction latency',
            ['model_version'],
            registry=self.registry
        )
        
        self.prediction_latency_p99 = Gauge(
            'prediction_latency_p99_seconds',
            '99th percentile prediction latency',
            ['model_version'],
            registry=self.registry
        )
        
        # Hardware Metrics (beyond GPU)
        self.hardware_temperature = Gauge(
            'hardware_temperature_celsius',
            'Hardware component temperature',
            ['component'],
            registry=self.registry
        )
        
        # Custom business metrics
        self.daily_predictions = Counter(
            'daily_predictions_total',
            'Total predictions made today',
            ['model_version'],
            registry=self.registry
        )
        
        self.model_drift_score = Gauge(
            'model_drift_score',
            'Model drift detection score',
            ['model_version', 'feature'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """
        Record API request metrics.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        self.requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_prediction(self, duration_ms: float, model_version: Optional[str] = None, 
                         prediction_value: Optional[float] = None) -> None:
        """
        Record single prediction metrics.
        
        Args:
            duration_ms: Prediction duration in milliseconds
            model_version: Optional version of the model used
            prediction_value: Optional prediction value for distribution tracking
        """
        # Convert milliseconds to seconds for Prometheus
        duration_seconds = duration_ms / 1000.0
        
        model_ver = model_version or "unknown"
        
        self.predictions_total.labels(
            model_version=model_ver,
            prediction_type="single"
        ).inc()
        
        self.prediction_duration.labels(
            model_version=model_ver
        ).observe(duration_seconds)
        
        if prediction_value is not None:
            self.prediction_values.labels(
                model_version=model_ver
            ).observe(prediction_value)
    
    def record_batch_prediction(self, batch_size: int, successful_predictions: int,
                              failed_predictions: int, total_processing_time_ms: float,
                              model_version: Optional[str] = None) -> None:
        """
        Record batch prediction metrics.
        
        Args:
            batch_size: Total number of predictions in batch
            successful_predictions: Number of successful predictions
            failed_predictions: Number of failed predictions
            total_processing_time_ms: Total processing time in milliseconds
            model_version: Optional version of the model used
        """
        # Convert milliseconds to seconds for Prometheus
        duration_seconds = total_processing_time_ms / 1000.0
        
        model_ver = model_version or "unknown"
        
        # Record batch prediction
        self.predictions_total.labels(
            model_version=model_ver,
            prediction_type="batch"
        ).inc()
        
        # Record batch duration
        self.prediction_duration.labels(
            model_version=model_ver
        ).observe(duration_seconds)
        
        # Record individual prediction counts
        for _ in range(successful_predictions):
            self.predictions_total.labels(
                model_version=model_ver,
                prediction_type="single"
            ).inc()
        
        # Record failed predictions as errors
        for _ in range(failed_predictions):
            self.errors_total.labels(
                error_type="prediction_failed",
                endpoint="predict_batch"
            ).inc()
    
    def record_original_prediction(self, model_version: str, prediction_type: str, 
                         duration: float, prediction_value: Optional[float] = None) -> None:
        """
        Record prediction metrics (original method for backward compatibility).
        
        Args:
            model_version: Version of the model used
            prediction_type: Type of prediction ('single' or 'batch')
            duration: Prediction duration in seconds
            prediction_value: Optional prediction value for distribution tracking
        """
        self.predictions_total.labels(
            model_version=model_version,
            prediction_type=prediction_type
        ).inc()
        
        self.prediction_duration.labels(
            model_version=model_version
        ).observe(duration)
        
        if prediction_value is not None:
            self.prediction_values.labels(
                model_version=model_version
            ).observe(prediction_value)
    
    def set_model_info(self, model_name: str, model_version: str, model_stage: str,
                      model_type: str, features: List[str]) -> None:
        """
        Set model information.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_stage: Stage of the model
            model_type: Type of the model
            features: List of feature names
        """
        self.model_info.info({
            'name': model_name,
            'version': model_version,
            'stage': model_stage,
            'type': model_type,
            'features': ','.join(features)
        })
    
    def set_model_status(self, status: str) -> None:
        """
        Set model status.
        
        Args:
            status: Model status ('loading', 'ready', 'error', 'unavailable')
        """
        self.model_status.state(status)
    
    def record_model_load_time(self, duration: float) -> None:
        """
        Record model loading time.
        
        Args:
            duration: Model loading duration in seconds
        """
        self.model_load_duration.observe(duration)
    
    def record_error(self, error_type: str, endpoint: str) -> None:
        """
        Record error occurrence.
        
        Args:
            error_type: Type of error
            endpoint: Endpoint where error occurred
        """
        self.errors_total.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()
    
    def record_database_operation(self, operation: str, table: str, duration: float) -> None:
        """
        Record database operation metrics.
        
        Args:
            operation: Database operation type
            table: Table name
            duration: Operation duration in seconds
        """
        self.database_operations_total.labels(
            operation=operation,
            table=table
        ).inc()
        
        self.database_operation_duration.labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    def record_database_query(self, query_type: str, duration_ms: float) -> None:
        """
        Record database query metrics.
        
        Args:
            query_type: Type of database query
            duration_ms: Query duration in milliseconds
        """
        self.database_operations_total.labels(
            operation="query",
            table=query_type
        ).inc()
        
        self.database_operation_duration.labels(
            operation="query",
            table=query_type
        ).observe(duration_ms / 1000.0)  # Convert to seconds
    
    def record_database_export(self, format: str, record_count: int, duration_ms: float) -> None:
        """
        Record database export metrics.
        
        Args:
            format: Export format (csv, json)
            record_count: Number of records exported
            duration_ms: Export duration in milliseconds
        """
        self.database_operations_total.labels(
            operation="export",
            table=format
        ).inc()
        
        self.database_operation_duration.labels(
            operation="export",
            table=format
        ).observe(duration_ms / 1000.0)  # Convert to seconds
    
    def update_gpu_metrics(self) -> None:
        """Update GPU metrics if available."""
        if not self._gpu_available:
            return
        
        try:
            for i, handle in enumerate(self._gpu_handles):
                # Get GPU name
                gpu_name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                gpu_id = str(i)
                
                # Get utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.labels(
                    gpu_id=gpu_id,
                    gpu_name=gpu_name
                ).set(utilization.gpu)
                
                # Get memory info
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory_used.labels(
                    gpu_id=gpu_id,
                    gpu_name=gpu_name
                ).set(memory_info.used)
                
                self.gpu_memory_total.labels(
                    gpu_id=gpu_id,
                    gpu_name=gpu_name
                ).set(memory_info.total)
                
                # Get temperature
                try:
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    self.gpu_temperature.labels(
                        gpu_id=gpu_id,
                        gpu_name=gpu_name
                    ).set(temperature)
                except Exception:
                    pass  # Temperature might not be available on all GPUs
                
                # Get power usage
                try:
                    power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    self.gpu_power_usage.labels(
                        gpu_id=gpu_id,
                        gpu_name=gpu_name
                    ).set(power_usage)
                except Exception:
                    pass  # Power usage might not be available on all GPUs
                
        except Exception as e:
            logger.error(f"Failed to update GPU metrics: {e}")
    
    def update_system_metrics(self) -> None:
        """Update system health metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            self.system_memory_total.set(memory.total)
            
            # Disk usage for root partition
            disk = psutil.disk_usage('/')
            self.system_disk_usage.labels(mount_point='/').set(disk.used)
            
            # Update last metrics update time
            self._last_metrics_update = datetime.now()
            
        except ImportError:
            logger.warning("psutil not available, system metrics disabled")
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_model_performance_metrics(self, model_version: str, dataset: str,
                                       accuracy: Optional[float] = None,
                                       rmse: Optional[float] = None,
                                       mae: Optional[float] = None,
                                       r2: Optional[float] = None) -> None:
        """
        Update model performance metrics.
        
        Args:
            model_version: Version of the model
            dataset: Dataset used for evaluation
            accuracy: Accuracy score
            rmse: RMSE score
            mae: MAE score
            r2: R² score
        """
        if accuracy is not None:
            self.model_accuracy.labels(
                model_version=model_version,
                dataset=dataset
            ).set(accuracy)
        
        if rmse is not None:
            self.model_rmse.labels(
                model_version=model_version,
                dataset=dataset
            ).set(rmse)
        
        if mae is not None:
            self.model_mae.labels(
                model_version=model_version,
                dataset=dataset
            ).set(mae)
        
        if r2 is not None:
            self.model_r2.labels(
                model_version=model_version,
                dataset=dataset
            ).set(r2)
    
    def update_api_health_status(self, component: str, is_healthy: bool) -> None:
        """
        Update API health status for a component.
        
        Args:
            component: Component name (e.g., 'database', 'model', 'gpu')
            is_healthy: Whether the component is healthy
        """
        self.api_health_status.labels(component=component).set(1 if is_healthy else 0)
    
    def update_prediction_latency_percentiles(self, model_version: str,
                                            p95_latency: float,
                                            p99_latency: float) -> None:
        """
        Update prediction latency percentiles.
        
        Args:
            model_version: Version of the model
            p95_latency: 95th percentile latency in seconds
            p99_latency: 99th percentile latency in seconds
        """
        self.prediction_latency_p95.labels(model_version=model_version).set(p95_latency)
        self.prediction_latency_p99.labels(model_version=model_version).set(p99_latency)
    
    def update_model_drift_score(self, model_version: str, feature: str, drift_score: float) -> None:
        """
        Update model drift detection score.
        
        Args:
            model_version: Version of the model
            feature: Feature name
            drift_score: Drift score (0-1, where 1 indicates high drift)
        """
        self.model_drift_score.labels(
            model_version=model_version,
            feature=feature
        ).set(drift_score)
    
    def record_daily_prediction(self, model_version: str) -> None:
        """
        Record a daily prediction count.
        
        Args:
            model_version: Version of the model used
        """
        self.daily_predictions.labels(model_version=model_version).inc()
    
    def update_active_connections(self, count: int) -> None:
        """
        Update active connections count.
        
        Args:
            count: Number of active connections
        """
        self.active_connections.set(count)
    
    def schedule_task(self, func: Callable, interval_seconds: int, task_name: str) -> None:
        """
        Schedule a recurring task for metrics collection.
        
        Args:
            func: Function to execute
            interval_seconds: Interval in seconds
            task_name: Name of the task for logging
        """
        def task_wrapper():
            try:
                logger.debug(f"Executing scheduled task: {task_name}")
                func()
            except Exception as e:
                logger.error(f"Error in scheduled task {task_name}: {e}")
        
        # Schedule the task
        schedule.every(interval_seconds).seconds.do(task_wrapper)
        self._scheduled_tasks.append((task_name, interval_seconds))
        logger.info(f"Scheduled task '{task_name}' to run every {interval_seconds} seconds")
    
    def start_background_monitoring(self, interval: float = 5.0) -> None:
        """
        Start background monitoring thread for GPU and system metrics.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_thread is not None:
            logger.warning("Background monitoring already started")
            return
        
        def monitor():
            while not self._stop_monitoring.wait(interval):
                try:
                    # Update GPU metrics if available
                    if self._gpu_available:
                        self.update_gpu_metrics()
                    
                    # Update system metrics
                    self.update_system_metrics()
                    
                except Exception as e:
                    logger.error(f"Error in background monitoring: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
        logger.info(f"Started background monitoring with {interval}s interval")
        
        # Start task scheduler if we have scheduled tasks
        if self._scheduled_tasks:
            self._start_task_scheduler()
    
    def _start_task_scheduler(self) -> None:
        """Start the task scheduler thread."""
        if self._task_scheduler_thread is not None:
            return
        
        def scheduler():
            while not self._stop_monitoring.is_set():
                try:
                    schedule.run_pending()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in task scheduler: {e}")
        
        self._task_scheduler_thread = threading.Thread(target=scheduler, daemon=True)
        self._task_scheduler_thread.start()
        logger.info("Started task scheduler thread")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring and task scheduler threads."""
        if self._monitoring_thread is not None or self._task_scheduler_thread is not None:
            self._stop_monitoring.set()
            
            if self._monitoring_thread is not None:
                self._monitoring_thread.join(timeout=10)
                self._monitoring_thread = None
            
            if self._task_scheduler_thread is not None:
                self._task_scheduler_thread.join(timeout=10)
                self._task_scheduler_thread = None
            
            # Clear scheduled tasks
            schedule.clear()
            self._scheduled_tasks.clear()
            
            logger.info("Stopped background monitoring and task scheduler")
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            labels: Optional labels for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            # This is a generic timing context manager
            # Specific metrics should be recorded by the caller
            logger.debug(f"Operation {operation_name} took {duration:.3f}s")
    
    def get_metrics(self) -> str:
        """
        Get current metrics in Prometheus format.
        
        Returns:
            Metrics string in Prometheus format
        """
        from prometheus_client import REGISTRY
        registry = self.registry or REGISTRY
        metrics_bytes = generate_latest(registry)
        return metrics_bytes.decode('utf-8')


# Decorator for automatic request timing
def time_request(metrics: PrometheusMetrics):
    """
    Decorator to automatically time API requests.
    
    Args:
        metrics: PrometheusMetrics instance
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Note: This is a basic implementation
                # In practice, you'd extract method, endpoint, and status from the request/response
                logger.debug(f"Request to {func.__name__} took {duration:.3f}s")
        return wrapper
    return decorator


def start_metrics_server(metrics: PrometheusMetrics, port: int = 8001) -> None:
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        metrics: PrometheusMetrics instance
        port: Port to serve metrics on
    """
    try:
        from prometheus_client import REGISTRY
        registry = metrics.registry or REGISTRY
        start_http_server(port, registry=registry)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise


# Global metrics instance
_metrics_instance: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """
    Get global metrics instance.
    
    Returns:
        PrometheusMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics()
    return _metrics_instance


def initialize_metrics(start_server: bool = True, server_port: int = 8001,
                      start_monitoring: bool = True, monitoring_interval: float = 5.0) -> PrometheusMetrics:
    """
    Initialize global metrics instance with optional server and monitoring.
    
    Args:
        start_server: Whether to start the metrics HTTP server
        server_port: Port for the metrics server
        start_monitoring: Whether to start background GPU monitoring
        monitoring_interval: GPU monitoring interval in seconds
    
    Returns:
        PrometheusMetrics instance
    """
    global _metrics_instance
    
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics()
        
        if start_monitoring:
            _metrics_instance.start_background_monitoring(monitoring_interval)
        
        if start_server:
            start_metrics_server(_metrics_instance, server_port)
    
    return _metrics_instance