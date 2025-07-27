#!/usr/bin/env python3
"""
Database Integration Demo

This script demonstrates the database integration and logging functionality
of the MLOps platform, including prediction logging, model performance tracking,
and system metrics collection.
"""

import sys
import os
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.api.config import APIConfig
from src.api.database import DatabaseManager, PredictionLogData
from src.api.database_init import initialize_database
from src.api.migrations import create_migrator


def demo_database_initialization():
    """Demonstrate database initialization and migration."""
    print("🔧 Database Initialization Demo")
    print("=" * 50)
    
    # Create config with demo database
    config = APIConfig()
    config.database_url = "sqlite:///./demo_database.db"
    
    print(f"Database URL: {config.database_url}")
    
    # Initialize database
    print("\n📦 Initializing database...")
    success = initialize_database(
        config=config,
        run_migrations=True,
        create_sample_data=False
    )
    
    if success:
        print("✅ Database initialized successfully")
    else:
        print("❌ Database initialization failed")
        return None
    
    # Show migration status
    migrator = create_migrator(config)
    current_version = migrator.get_current_version()
    applied_migrations = migrator.get_applied_migrations()
    
    print(f"\n📋 Migration Status:")
    print(f"  Current Version: {current_version}")
    print(f"  Applied Migrations: {len(applied_migrations)}")
    for migration in applied_migrations:
        print(f"    - {migration}")
    
    return DatabaseManager(config.database_url)


def demo_prediction_logging(db_manager: DatabaseManager):
    """Demonstrate prediction logging functionality."""
    print("\n📊 Prediction Logging Demo")
    print("=" * 50)
    
    # Sample housing data for predictions
    sample_predictions = [
        {
            "input_features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23
            },
            "prediction": 4.526,
            "processing_time_ms": 12.5
        },
        {
            "input_features": {
                "MedInc": 7.2574,
                "HouseAge": 21.0,
                "AveRooms": 5.734,
                "AveBedrms": 1.129,
                "Population": 2401.0,
                "AveOccup": 2.109,
                "Latitude": 37.86,
                "Longitude": -122.22
            },
            "prediction": 3.585,
            "processing_time_ms": 15.2
        },
        {
            "input_features": {
                "MedInc": 5.6431,
                "HouseAge": 52.0,
                "AveRooms": 5.817,
                "AveBedrms": 1.073,
                "Population": 558.0,
                "AveOccup": 2.547,
                "Latitude": 37.85,
                "Longitude": -122.25
            },
            "prediction": 3.521,
            "processing_time_ms": 18.7
        }
    ]
    
    # Log individual predictions
    print("📝 Logging individual predictions...")
    for i, pred_data in enumerate(sample_predictions):
        prediction_log = PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v1.2.3",
            model_stage="Production",
            input_features=pred_data["input_features"],
            prediction=pred_data["prediction"],
            confidence_lower=pred_data["prediction"] * 0.9,
            confidence_upper=pred_data["prediction"] * 1.1,
            confidence_score=0.85 + (i * 0.05),
            processing_time_ms=pred_data["processing_time_ms"],
            status="success"
        )
        
        success = db_manager.log_prediction(prediction_log)
        print(f"  Prediction {i+1}: {'✅ Logged' if success else '❌ Failed'}")
    
    # Log a batch prediction
    print("\n📦 Logging batch prediction...")
    batch_id = str(uuid.uuid4())
    
    for i in range(5):
        prediction_log = PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v1.2.3",
            model_stage="Production",
            input_features={"batch_item": i, "value": i * 1.5},
            prediction=i * 2.0,
            processing_time_ms=8.0 + i,
            batch_id=batch_id,
            status="success"
        )
        
        db_manager.log_prediction(prediction_log)
    
    print(f"  Batch prediction (5 items): ✅ Logged with batch_id: {batch_id[:8]}...")
    
    # Log a failed prediction
    print("\n❌ Logging failed prediction...")
    failed_prediction = PredictionLogData(
        request_id=str(uuid.uuid4()),
        model_version="v1.2.3",
        model_stage="Production",
        input_features={"invalid": "data"},
        prediction=0.0,
        processing_time_ms=5.0,
        status="error",
        error_message="Invalid input data format"
    )
    
    success = db_manager.log_prediction(failed_prediction)
    print(f"  Failed prediction: {'✅ Logged' if success else '❌ Failed'}")


def demo_model_performance_tracking(db_manager: DatabaseManager):
    """Demonstrate model performance tracking."""
    print("\n📈 Model Performance Tracking Demo")
    print("=" * 50)
    
    # Log performance metrics for different model versions
    model_versions = [
        ("v1.0.0", "Production", {
            "rmse": 0.654,
            "mae": 0.523,
            "r2_score": 0.789,
            "mape": 12.3
        }),
        ("v1.1.0", "Staging", {
            "rmse": 0.621,
            "mae": 0.498,
            "r2_score": 0.812,
            "mape": 11.8
        }),
        ("v1.2.0", "Development", {
            "rmse": 0.598,
            "mae": 0.475,
            "r2_score": 0.834,
            "mape": 11.2
        })
    ]
    
    print("📊 Logging model performance metrics...")
    for version, stage, metrics in model_versions:
        success = db_manager.log_model_performance(
            model_version=version,
            model_stage=stage,
            metrics=metrics,
            dataset_version="california_housing_v2"
        )
        
        print(f"  Model {version} ({stage}): {'✅ Logged' if success else '❌ Failed'}")
        for metric_name, metric_value in metrics.items():
            print(f"    - {metric_name}: {metric_value}")


def demo_system_metrics_logging(db_manager: DatabaseManager):
    """Demonstrate system metrics logging."""
    print("\n🖥️  System Metrics Logging Demo")
    print("=" * 50)
    
    # Sample system metrics
    system_metrics = [
        ("gpu_utilization", 75.5, {"device": "cuda:0", "gpu_name": "RTX 4090"}),
        ("gpu_memory_used", 8192.0, {"device": "cuda:0", "unit": "MB"}),
        ("gpu_temperature", 68.0, {"device": "cuda:0", "unit": "celsius"}),
        ("cpu_utilization", 45.2, {"cores": 16, "unit": "percent"}),
        ("memory_usage", 16384.0, {"total": 32768.0, "unit": "MB"}),
        ("api_response_time", 45.2, {"endpoint": "/predict", "method": "POST"}),
        ("prediction_throughput", 150.0, {"unit": "predictions_per_minute"}),
        ("model_inference_time", 12.8, {"model": "v1.2.3", "unit": "ms"})
    ]
    
    print("📊 Logging system metrics...")
    for metric_name, metric_value, labels in system_metrics:
        success = db_manager.log_system_metric(metric_name, metric_value, labels)
        print(f"  {metric_name}: {metric_value} {'✅ Logged' if success else '❌ Failed'}")


def demo_data_retrieval(db_manager: DatabaseManager):
    """Demonstrate data retrieval and statistics."""
    print("\n📋 Data Retrieval Demo")
    print("=" * 50)
    
    # Get recent predictions
    print("🔍 Retrieving recent predictions...")
    predictions = db_manager.get_predictions(limit=5)
    print(f"  Found {len(predictions)} recent predictions:")
    
    for pred in predictions[:3]:  # Show first 3
        print(f"    - ID: {pred.request_id[:8]}... | "
              f"Prediction: {pred.prediction:.3f} | "
              f"Status: {pred.status} | "
              f"Time: {pred.processing_time_ms:.1f}ms")
    
    if len(predictions) > 3:
        print(f"    ... and {len(predictions) - 3} more")
    
    # Get prediction statistics
    print("\n📊 Prediction Statistics:")
    stats = db_manager.get_prediction_stats()
    
    if stats:
        print(f"  Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"  Successful: {stats.get('successful_predictions', 0)}")
        print(f"  Failed: {stats.get('failed_predictions', 0)}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"  Average Processing Time: {stats.get('average_processing_time_ms', 0):.2f}ms")
    else:
        print("  No statistics available")
    
    # Get predictions from last hour
    print("\n🕐 Predictions from last hour:")
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_predictions = db_manager.get_predictions(
        start_time=one_hour_ago,
        limit=10
    )
    print(f"  Found {len(recent_predictions)} predictions in the last hour")


def demo_database_maintenance(db_manager: DatabaseManager):
    """Demonstrate database maintenance operations."""
    print("\n🧹 Database Maintenance Demo")
    print("=" * 50)
    
    # Health check
    print("🏥 Database Health Check:")
    is_healthy = db_manager.health_check()
    print(f"  Database Status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
    
    # Cleanup old records (simulate by setting days_to_keep to 0)
    print("\n🗑️  Cleanup Simulation (not actually deleting):")
    print("  In production, you would run:")
    print("  deleted_count = db_manager.cleanup_old_records(days_to_keep=30)")
    print("  This would remove records older than 30 days")


def main():
    """Main demo function."""
    print("🚀 MLOps Platform Database Integration Demo")
    print("=" * 60)
    
    try:
        # Initialize database
        db_manager = demo_database_initialization()
        if not db_manager:
            return 1
        
        # Wait a moment for dramatic effect
        time.sleep(1)
        
        # Run demos
        demo_prediction_logging(db_manager)
        time.sleep(1)
        
        demo_model_performance_tracking(db_manager)
        time.sleep(1)
        
        demo_system_metrics_logging(db_manager)
        time.sleep(1)
        
        demo_data_retrieval(db_manager)
        time.sleep(1)
        
        demo_database_maintenance(db_manager)
        
        print("\n🎉 Database Integration Demo Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Database initialization and migrations")
        print("  ✅ Prediction logging with batch support")
        print("  ✅ Model performance tracking")
        print("  ✅ System metrics collection")
        print("  ✅ Data retrieval and statistics")
        print("  ✅ Database health monitoring")
        
        print(f"\nDemo database created at: demo_database.db")
        print("You can inspect it using SQLite tools or the management script:")
        print("  python scripts/manage_database.py status --database-url sqlite:///./demo_database.db")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Demo cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup demo database
        try:
            if os.path.exists("demo_database.db"):
                print("\n🧹 Cleaning up demo database...")
                os.remove("demo_database.db")
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())