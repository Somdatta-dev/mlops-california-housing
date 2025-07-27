#!/usr/bin/env python3
"""
Prometheus Metrics Implementation Demo

This script demonstrates the comprehensive Prometheus metrics implementation
for the MLOps platform, including GPU monitoring, prediction metrics, and
system health monitoring.

Usage:
    python examples/prometheus_metrics_demo.py
"""

import asyncio
import time
import random
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.api.metrics import PrometheusMetrics, initialize_metrics, get_metrics
    from src.api.config import get_api_config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


class MetricsDemo:
    """
    Demonstration class for Prometheus metrics functionality.
    """
    
    def __init__(self):
        """Initialize the metrics demo."""
        self.metrics = None
        self.demo_running = False
        
    async def initialize_metrics(self) -> None:
        """Initialize Prometheus metrics with full configuration."""
        logger.info("Initializing Prometheus metrics...")
        
        try:
            # Initialize metrics with background monitoring
            self.metrics = initialize_metrics(
                start_server=True,
                server_port=8001,
                start_monitoring=True,
                monitoring_interval=2.0  # More frequent for demo
            )
            
            # Set system information
            import platform
            self.metrics.system_info.info({
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'demo_version': '1.0.0',
                'gpu_available': str(self.metrics._gpu_available)
            })
            
            # Set initial model information
            self.metrics.set_model_info(
                model_name="california_housing_predictor",
                model_version="v1.2.3",
                model_stage="Production",
                model_type="XGBoost",
                features=[
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"
                ]
            )
            
            # Set model status
            self.metrics.set_model_status("ready")
            
            logger.info("✅ Prometheus metrics initialized successfully")
            logger.info(f"📊 Metrics server running on http://localhost:8001")
            logger.info(f"🔧 GPU monitoring: {'enabled' if self.metrics._gpu_available else 'disabled'}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize metrics: {e}")
            raise
    
    def simulate_api_requests(self, num_requests: int = 10) -> None:
        """
        Simulate API requests to demonstrate request metrics.
        
        Args:
            num_requests: Number of requests to simulate
        """
        logger.info(f"🌐 Simulating {num_requests} API requests...")
        
        endpoints = ["/predict", "/predict/batch", "/health", "/model/info"]
        methods = ["GET", "POST"]
        status_codes = [200, 200, 200, 200, 400, 422, 500]  # Mostly successful
        
        for i in range(num_requests):
            # Simulate request processing
            method = random.choice(methods)
            endpoint = random.choice(endpoints)
            status_code = random.choice(status_codes)
            duration = random.uniform(0.01, 0.5)  # 10ms to 500ms
            
            # Record request metrics
            self.metrics.record_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
            
            # Simulate some errors
            if status_code >= 400:
                error_types = ["validation_error", "model_error", "timeout_error"]
                self.metrics.record_error(
                    error_type=random.choice(error_types),
                    endpoint=endpoint
                )
            
            logger.info(f"  📝 Request {i+1}: {method} {endpoint} -> {status_code} ({duration:.3f}s)")
            time.sleep(0.1)  # Small delay between requests
        
        logger.info("✅ API request simulation completed")
    
    def simulate_predictions(self, num_predictions: int = 15) -> None:
        """
        Simulate model predictions to demonstrate prediction metrics.
        
        Args:
            num_predictions: Number of predictions to simulate
        """
        logger.info(f"🔮 Simulating {num_predictions} model predictions...")
        
        model_versions = ["v1.2.3", "v1.2.2", "v1.1.0"]
        
        for i in range(num_predictions):
            model_version = random.choice(model_versions)
            
            # Simulate single predictions
            if i % 3 != 0:  # 2/3 are single predictions
                duration_ms = random.uniform(5, 50)  # 5-50ms
                prediction_value = random.uniform(0.5, 8.0)  # Housing price range
                
                self.metrics.record_prediction(
                    duration_ms=duration_ms,
                    model_version=model_version,
                    prediction_value=prediction_value
                )
                
                logger.info(f"  🎯 Single prediction {i+1}: {prediction_value:.2f} ({duration_ms:.1f}ms)")
            
            else:  # 1/3 are batch predictions
                batch_size = random.randint(5, 20)
                successful = random.randint(batch_size - 2, batch_size)
                failed = batch_size - successful
                total_time_ms = random.uniform(50, 200)
                
                self.metrics.record_batch_prediction(
                    batch_size=batch_size,
                    successful_predictions=successful,
                    failed_predictions=failed,
                    total_processing_time_ms=total_time_ms,
                    model_version=model_version
                )
                
                logger.info(f"  📦 Batch prediction {i+1}: {successful}/{batch_size} successful ({total_time_ms:.1f}ms)")
            
            time.sleep(0.2)  # Small delay between predictions
        
        logger.info("✅ Prediction simulation completed")
    
    def simulate_database_operations(self, num_operations: int = 8) -> None:
        """
        Simulate database operations to demonstrate database metrics.
        
        Args:
            num_operations: Number of operations to simulate
        """
        logger.info(f"🗄️ Simulating {num_operations} database operations...")
        
        operations = ["INSERT", "SELECT", "UPDATE", "DELETE"]
        tables = ["predictions", "model_performance", "system_metrics"]
        
        for i in range(num_operations):
            operation = random.choice(operations)
            table = random.choice(tables)
            duration = random.uniform(0.001, 0.1)  # 1ms to 100ms
            
            self.metrics.record_database_operation(
                operation=operation,
                table=table,
                duration=duration
            )
            
            logger.info(f"  💾 DB Operation {i+1}: {operation} on {table} ({duration*1000:.1f}ms)")
            time.sleep(0.1)
        
        logger.info("✅ Database operation simulation completed")
    
    def simulate_model_operations(self) -> None:
        """Simulate model loading and status changes."""
        logger.info("🤖 Simulating model operations...")
        
        # Simulate model loading
        logger.info("  📥 Loading model...")
        self.metrics.set_model_status("loading")
        
        # Simulate loading time
        load_duration = random.uniform(2.0, 5.0)
        time.sleep(min(load_duration, 2.0))  # Cap sleep for demo
        self.metrics.record_model_load_time(load_duration)
        
        # Model ready
        self.metrics.set_model_status("ready")
        logger.info(f"  ✅ Model loaded successfully ({load_duration:.1f}s)")
        
        # Update model info
        self.metrics.set_model_info(
            model_name="california_housing_predictor_v2",
            model_version="v1.3.0",
            model_stage="Staging",
            model_type="LightGBM",
            features=[
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"
            ]
        )
        
        logger.info("✅ Model operations simulation completed")
    
    def display_gpu_metrics(self) -> None:
        """Display current GPU metrics if available."""
        if not self.metrics._gpu_available:
            logger.info("🚫 GPU metrics not available (nvidia-ml-py not installed or no GPU)")
            return
        
        logger.info("🎮 GPU Metrics:")
        
        try:
            # Force update GPU metrics
            self.metrics.update_gpu_metrics()
            
            # Get current metrics (this is a simplified display)
            logger.info("  📊 GPU metrics updated successfully")
            logger.info("  💡 Check http://localhost:8001 for detailed GPU metrics")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to update GPU metrics: {e}")
    
    def display_metrics_summary(self) -> None:
        """Display a summary of available metrics."""
        logger.info("📈 Metrics Summary:")
        logger.info("  🌐 API Request Metrics:")
        logger.info("    - api_requests_total (counter)")
        logger.info("    - api_request_duration_seconds (histogram)")
        logger.info("")
        logger.info("  🔮 Prediction Metrics:")
        logger.info("    - predictions_total (counter)")
        logger.info("    - prediction_duration_seconds (histogram)")
        logger.info("    - prediction_values (histogram)")
        logger.info("")
        logger.info("  🤖 Model Metrics:")
        logger.info("    - model_info (info)")
        logger.info("    - model_load_duration_seconds (histogram)")
        logger.info("    - model_status (enum)")
        logger.info("")
        logger.info("  🗄️ Database Metrics:")
        logger.info("    - database_operations_total (counter)")
        logger.info("    - database_operation_duration_seconds (histogram)")
        logger.info("")
        logger.info("  ⚠️ Error Metrics:")
        logger.info("    - errors_total (counter)")
        logger.info("")
        
        if self.metrics._gpu_available:
            logger.info("  🎮 GPU Metrics:")
            logger.info("    - gpu_utilization_percent (gauge)")
            logger.info("    - gpu_memory_used_bytes (gauge)")
            logger.info("    - gpu_memory_total_bytes (gauge)")
            logger.info("    - gpu_temperature_celsius (gauge)")
            logger.info("    - gpu_power_usage_watts (gauge)")
        else:
            logger.info("  🚫 GPU Metrics: Not available")
        
        logger.info("")
        logger.info("  ℹ️ System Info:")
        logger.info("    - system_info (info)")
    
    async def run_comprehensive_demo(self) -> None:
        """Run a comprehensive demonstration of all metrics features."""
        logger.info("🚀 Starting Comprehensive Prometheus Metrics Demo")
        logger.info("=" * 60)
        
        try:
            # Initialize metrics
            await self.initialize_metrics()
            
            # Display metrics summary
            self.display_metrics_summary()
            
            logger.info("=" * 60)
            logger.info("🎬 Running Simulations...")
            
            # Run simulations
            self.simulate_model_operations()
            logger.info("-" * 40)
            
            self.simulate_api_requests(12)
            logger.info("-" * 40)
            
            self.simulate_predictions(18)
            logger.info("-" * 40)
            
            self.simulate_database_operations(10)
            logger.info("-" * 40)
            
            self.display_gpu_metrics()
            
            logger.info("=" * 60)
            logger.info("✅ Demo completed successfully!")
            logger.info("")
            logger.info("📊 Metrics are now available at:")
            logger.info("  - Prometheus format: http://localhost:8001")
            logger.info("  - Or via API endpoint: http://localhost:8000/metrics")
            logger.info("")
            logger.info("🔧 To view metrics in Prometheus:")
            logger.info("  1. Install Prometheus")
            logger.info("  2. Configure scrape target: localhost:8001")
            logger.info("  3. Query metrics like: api_requests_total, gpu_utilization_percent")
            logger.info("")
            logger.info("📈 Sample Prometheus queries:")
            logger.info("  - rate(api_requests_total[5m])")
            logger.info("  - histogram_quantile(0.95, prediction_duration_seconds_bucket)")
            logger.info("  - gpu_utilization_percent")
            logger.info("  - increase(errors_total[1h])")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.metrics:
                self.metrics.stop_background_monitoring()
                logger.info("🛑 Background monitoring stopped")


async def main():
    """Main demo function."""
    demo = MetricsDemo()
    
    try:
        await demo.run_comprehensive_demo()
        
        # Keep the metrics server running for a bit
        logger.info("⏳ Keeping metrics server running for 30 seconds...")
        logger.info("   You can now check http://localhost:8001 for metrics")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))