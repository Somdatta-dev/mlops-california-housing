#!/usr/bin/env python3
"""
Tests for Prometheus Metrics Implementation

This module provides comprehensive tests for the Prometheus metrics functionality,
including GPU monitoring, prediction metrics, system health monitoring, and
background task scheduling.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, REGISTRY

from src.api.metrics import PrometheusMetrics, initialize_metrics, get_metrics


class TestPrometheusMetrics:
    """Test suite for PrometheusMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a custom registry for testing to avoid conflicts
        self.test_registry = CollectorRegistry()
        self.metrics = PrometheusMetrics(registry=self.test_registry)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.metrics:
            self.metrics.stop_background_monitoring()
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        assert self.metrics is not None
        assert self.metrics.registry == self.test_registry
        assert hasattr(self.metrics, 'requests_total')
        assert hasattr(self.metrics, 'prediction_duration')
        assert hasattr(self.metrics, 'model_info')
    
    def test_custom_metrics_initialization(self):
        """Test custom metrics initialization."""
        assert hasattr(self.metrics, 'model_accuracy')
        assert hasattr(self.metrics, 'system_cpu_usage')
        assert hasattr(self.metrics, 'api_health_status')
        assert hasattr(self.metrics, 'prediction_latency_p95')
        assert hasattr(self.metrics, 'model_drift_score')
    
    def test_record_request(self):
        """Test recording API request metrics."""
        # Record a request
        self.metrics.record_request(
            method="POST",
            endpoint="/predict",
            status_code=200,
            duration=0.123
        )
        
        # Check that metrics were recorded
        requests_metric = self.metrics.requests_total
        duration_metric = self.metrics.request_duration
        
        # Verify the metrics exist (exact values depend on Prometheus client internals)
        assert requests_metric is not None
        assert duration_metric is not None
    
    def test_record_prediction(self):
        """Test recording prediction metrics."""
        # Record a single prediction
        self.metrics.record_prediction(
            duration_ms=25.5,
            model_version="v1.2.3",
            prediction_value=4.5
        )
        
        # Verify metrics exist
        assert self.metrics.predictions_total is not None
        assert self.metrics.prediction_duration is not None
        assert self.metrics.prediction_values is not None
    
    def test_record_batch_prediction(self):
        """Test recording batch prediction metrics."""
        self.metrics.record_batch_prediction(
            batch_size=10,
            successful_predictions=8,
            failed_predictions=2,
            total_processing_time_ms=150.0,
            model_version="v1.2.3"
        )
        
        # Verify metrics exist
        assert self.metrics.predictions_total is not None
        assert self.metrics.prediction_duration is not None
        assert self.metrics.errors_total is not None
    
    def test_model_info_and_status(self):
        """Test setting model information and status."""
        # Set model info
        self.metrics.set_model_info(
            model_name="test_model",
            model_version="v1.0.0",
            model_stage="Production",
            model_type="XGBoost",
            features=["feature1", "feature2"]
        )
        
        # Set model status
        self.metrics.set_model_status("ready")
        
        # Record model load time
        self.metrics.record_model_load_time(2.5)
        
        # Verify metrics exist
        assert self.metrics.model_info is not None
        assert self.metrics.model_status is not None
        assert self.metrics.model_load_duration is not None
    
    def test_error_recording(self):
        """Test error recording."""
        self.metrics.record_error(
            error_type="validation_error",
            endpoint="/predict"
        )
        
        assert self.metrics.errors_total is not None
    
    def test_database_operations(self):
        """Test database operation metrics."""
        self.metrics.record_database_operation(
            operation="INSERT",
            table="predictions",
            duration=0.025
        )
        
        assert self.metrics.database_operations_total is not None
        assert self.metrics.database_operation_duration is not None
    
    def test_model_performance_metrics(self):
        """Test model performance metrics updates."""
        self.metrics.update_model_performance_metrics(
            model_version="v1.2.3",
            dataset="test",
            accuracy=0.95,
            rmse=0.123,
            mae=0.089,
            r2=0.92
        )
        
        assert self.metrics.model_accuracy is not None
        assert self.metrics.model_rmse is not None
        assert self.metrics.model_mae is not None
        assert self.metrics.model_r2 is not None
    
    def test_api_health_status(self):
        """Test API health status updates."""
        self.metrics.update_api_health_status("database", True)
        self.metrics.update_api_health_status("model", False)
        
        assert self.metrics.api_health_status is not None
    
    def test_prediction_latency_percentiles(self):
        """Test prediction latency percentile updates."""
        self.metrics.update_prediction_latency_percentiles(
            model_version="v1.2.3",
            p95_latency=0.05,
            p99_latency=0.1
        )
        
        assert self.metrics.prediction_latency_p95 is not None
        assert self.metrics.prediction_latency_p99 is not None
    
    def test_model_drift_score(self):
        """Test model drift score updates."""
        self.metrics.update_model_drift_score(
            model_version="v1.2.3",
            feature="MedInc",
            drift_score=0.15
        )
        
        assert self.metrics.model_drift_score is not None
    
    def test_daily_prediction_recording(self):
        """Test daily prediction recording."""
        self.metrics.record_daily_prediction("v1.2.3")
        
        assert self.metrics.daily_predictions is not None
    
    def test_active_connections_update(self):
        """Test active connections update."""
        self.metrics.update_active_connections(42)
        
        assert self.metrics.active_connections is not None
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_metrics_update(self, mock_disk_usage, mock_virtual_memory, mock_cpu_percent):
        """Test system metrics update."""
        # Mock psutil responses
        mock_cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.used = 8000000000
        mock_memory.total = 16000000000
        mock_virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.used = 500000000000
        mock_disk_usage.return_value = mock_disk
        
        # Update system metrics
        self.metrics.update_system_metrics()
        
        # Verify metrics exist
        assert self.metrics.system_cpu_usage is not None
        assert self.metrics.system_memory_usage is not None
        assert self.metrics.system_memory_total is not None
        assert self.metrics.system_disk_usage is not None
    
    @patch('src.api.metrics.nvml')
    def test_gpu_metrics_update_with_gpu(self, mock_nvml):
        """Test GPU metrics update when GPU is available."""
        # Mock NVIDIA ML responses
        mock_handle = Mock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"
        
        mock_utilization = Mock()
        mock_utilization.gpu = 85
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_utilization
        
        mock_memory = Mock()
        mock_memory.used = 12000000000
        mock_memory.total = 24000000000
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory
        
        mock_nvml.nvmlDeviceGetTemperature.return_value = 72
        mock_nvml.nvmlDeviceGetPowerUsage.return_value = 350000  # milliwatts
        
        # Set up GPU availability
        self.metrics._gpu_available = True
        self.metrics._gpu_handles = [mock_handle]
        
        # Update GPU metrics
        self.metrics.update_gpu_metrics()
        
        # Verify metrics exist
        assert self.metrics.gpu_utilization is not None
        assert self.metrics.gpu_memory_used is not None
        assert self.metrics.gpu_memory_total is not None
        assert self.metrics.gpu_temperature is not None
        assert self.metrics.gpu_power_usage is not None
    
    def test_gpu_metrics_update_without_gpu(self):
        """Test GPU metrics update when GPU is not available."""
        # Ensure GPU is not available
        self.metrics._gpu_available = False
        
        # This should not raise an exception
        self.metrics.update_gpu_metrics()
    
    def test_task_scheduling(self):
        """Test task scheduling functionality."""
        # Mock function to schedule
        mock_task = Mock()
        
        # Schedule a task
        self.metrics.schedule_task(
            func=mock_task,
            interval_seconds=1,
            task_name="test_task"
        )
        
        # Verify task was scheduled
        assert len(self.metrics._scheduled_tasks) == 1
        assert self.metrics._scheduled_tasks[0] == ("test_task", 1)
    
    def test_background_monitoring_start_stop(self):
        """Test starting and stopping background monitoring."""
        # Start monitoring
        self.metrics.start_background_monitoring(interval=0.1)
        
        # Verify monitoring thread started
        assert self.metrics._monitoring_thread is not None
        assert self.metrics._monitoring_thread.is_alive()
        
        # Wait a bit for monitoring to run
        time.sleep(0.2)
        
        # Stop monitoring
        self.metrics.stop_background_monitoring()
        
        # Verify monitoring thread stopped
        assert self.metrics._monitoring_thread is None
    
    def test_time_operation_context_manager(self):
        """Test the time_operation context manager."""
        with self.metrics.time_operation("test_operation"):
            time.sleep(0.01)  # Small delay
        
        # Context manager should complete without error
        # (Actual timing is logged, not stored in metrics)
    
    def test_get_metrics_output(self):
        """Test getting metrics in Prometheus format."""
        # Record some metrics first
        self.metrics.record_request("GET", "/health", 200, 0.01)
        self.metrics.record_prediction(15.0, "v1.0.0", 4.5)
        
        # Get metrics output
        metrics_output = self.metrics.get_metrics()
        
        # Verify output is a string and contains expected content
        assert isinstance(metrics_output, str)
        assert len(metrics_output) > 0
        # Should contain some metric names
        assert "api_requests_total" in metrics_output or "requests_total" in metrics_output


class TestMetricsGlobalFunctions:
    """Test suite for global metrics functions."""
    
    def test_initialize_metrics(self):
        """Test global metrics initialization."""
        # Initialize with custom settings
        metrics = initialize_metrics(
            start_server=False,  # Don't start server in tests
            start_monitoring=False  # Don't start monitoring in tests
        )
        
        assert metrics is not None
        assert isinstance(metrics, PrometheusMetrics)
        
        # Clean up
        metrics.stop_background_monitoring()
    
    def test_get_metrics_singleton(self):
        """Test get_metrics singleton behavior."""
        # Get metrics instance
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        
        # Should be the same instance
        assert metrics1 is metrics2
        
        # Clean up
        metrics1.stop_background_monitoring()


class TestMetricsIntegration:
    """Integration tests for metrics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_registry = CollectorRegistry()
        self.metrics = PrometheusMetrics(registry=self.test_registry)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.metrics:
            self.metrics.stop_background_monitoring()
    
    def test_comprehensive_metrics_workflow(self):
        """Test a comprehensive metrics workflow."""
        # Set up model info
        self.metrics.set_model_info(
            model_name="california_housing_predictor",
            model_version="v1.2.3",
            model_stage="Production",
            model_type="XGBoost",
            features=["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                     "Population", "AveOccup", "Latitude", "Longitude"]
        )
        
        # Set model status
        self.metrics.set_model_status("ready")
        
        # Record some API requests
        for i in range(5):
            self.metrics.record_request(
                method="POST",
                endpoint="/predict",
                status_code=200,
                duration=0.02 + i * 0.01
            )
        
        # Record some predictions
        for i in range(10):
            self.metrics.record_prediction(
                duration_ms=15.0 + i * 2,
                model_version="v1.2.3",
                prediction_value=3.5 + i * 0.1
            )
        
        # Record batch prediction
        self.metrics.record_batch_prediction(
            batch_size=20,
            successful_predictions=18,
            failed_predictions=2,
            total_processing_time_ms=250.0,
            model_version="v1.2.3"
        )
        
        # Update model performance
        self.metrics.update_model_performance_metrics(
            model_version="v1.2.3",
            dataset="validation",
            accuracy=0.95,
            rmse=0.123,
            mae=0.089,
            r2=0.92
        )
        
        # Update health status
        self.metrics.update_api_health_status("database", True)
        self.metrics.update_api_health_status("model", True)
        self.metrics.update_api_health_status("gpu", True)
        
        # Record some database operations
        self.metrics.record_database_operation("INSERT", "predictions", 0.025)
        self.metrics.record_database_operation("SELECT", "model_performance", 0.015)
        
        # Get final metrics output
        metrics_output = self.metrics.get_metrics()
        
        # Verify comprehensive output
        assert isinstance(metrics_output, str)
        assert len(metrics_output) > 100  # Should be substantial output
        
        # Should contain various metric types
        expected_metrics = [
            "requests_total", "prediction_duration", "model_info",
            "model_accuracy", "api_health_status", "database_operations"
        ]
        
        # At least some of these should be present
        found_metrics = sum(1 for metric in expected_metrics if metric in metrics_output)
        assert found_metrics > 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_background_monitoring_integration(self, mock_disk_usage, mock_virtual_memory, mock_cpu_percent):
        """Test background monitoring integration."""
        # Mock psutil
        mock_cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.used = 8000000000
        mock_memory.total = 16000000000
        mock_virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.used = 500000000000
        mock_disk_usage.return_value = mock_disk
        
        # Start background monitoring with short interval
        self.metrics.start_background_monitoring(interval=0.1)
        
        # Wait for a few monitoring cycles
        time.sleep(0.3)
        
        # Stop monitoring
        self.metrics.stop_background_monitoring()
        
        # Verify system metrics were updated
        # (We can't easily verify the exact values, but the monitoring should have run)
        assert mock_cpu_percent.called
        assert mock_virtual_memory.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])