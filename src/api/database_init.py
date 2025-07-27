"""
Database Initialization and Setup

This module provides utilities for initializing the database,
running migrations, and setting up the database for first use.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .config import APIConfig, get_api_config
from .database import DatabaseManager, get_database_manager, initialize_database_manager
from .migrations import DatabaseMigrator, create_migrator

logger = logging.getLogger(__name__)


def initialize_database(config: Optional[APIConfig] = None, 
                       run_migrations: bool = True,
                       create_sample_data: bool = False) -> bool:
    """
    Initialize database with schema, migrations, and optional sample data.
    
    Args:
        config: Optional API configuration
        run_migrations: Whether to run database migrations
        create_sample_data: Whether to create sample data for testing
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        if config is None:
            config = get_api_config()
        
        logger.info("Initializing database...")
        
        # Initialize database manager
        db_manager = initialize_database_manager(config)
        
        # Run migrations if requested
        if run_migrations:
            logger.info("Running database migrations...")
            migrator = create_migrator(config)
            
            if not migrator.migrate_to_latest():
                logger.error("Database migration failed")
                return False
            
            logger.info("Database migrations completed successfully")
        
        # Validate schema
        logger.info("Validating database schema...")
        migrator = create_migrator(config)
        validation_result = migrator.validate_schema()
        
        if not validation_result.get("valid", False):
            logger.error(f"Database schema validation failed: {validation_result}")
            return False
        
        logger.info("Database schema validation passed")
        
        # Create sample data if requested
        if create_sample_data:
            logger.info("Creating sample data...")
            if not _create_sample_data(db_manager):
                logger.warning("Failed to create sample data, but continuing...")
        
        # Test database connectivity
        if not db_manager.health_check():
            logger.error("Database health check failed")
            return False
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def _create_sample_data(db_manager: DatabaseManager) -> bool:
    """
    Create sample data for testing and demonstration.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from datetime import datetime, timedelta
        import uuid
        from .database import PredictionLogData
        
        # Sample prediction data
        sample_predictions = [
            {
                "request_id": str(uuid.uuid4()),
                "model_version": "v1.0.0",
                "model_stage": "Production",
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
                "confidence_lower": 4.1,
                "confidence_upper": 4.9,
                "confidence_score": 0.85,
                "processing_time_ms": 12.5,
                "status": "success"
            },
            {
                "request_id": str(uuid.uuid4()),
                "model_version": "v1.0.0",
                "model_stage": "Production",
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
                "confidence_lower": 3.2,
                "confidence_upper": 4.0,
                "confidence_score": 0.78,
                "processing_time_ms": 15.2,
                "status": "success"
            }
        ]
        
        # Log sample predictions
        for pred_data in sample_predictions:
            prediction_log = PredictionLogData(**pred_data)
            db_manager.log_prediction(prediction_log)
        
        # Sample model performance metrics
        model_metrics = {
            "rmse": 0.654,
            "mae": 0.523,
            "r2_score": 0.789,
            "mape": 12.3
        }
        
        db_manager.log_model_performance(
            model_version="v1.0.0",
            model_stage="Production",
            metrics=model_metrics,
            dataset_version="california_housing_v1"
        )
        
        # Sample system metrics
        system_metrics = [
            ("gpu_utilization", 75.5, {"device": "cuda:0"}),
            ("gpu_memory_used", 8192.0, {"device": "cuda:0", "unit": "MB"}),
            ("api_response_time", 45.2, {"endpoint": "/predict", "method": "POST"}),
            ("prediction_throughput", 150.0, {"unit": "predictions_per_minute"})
        ]
        
        for metric_name, metric_value, labels in system_metrics:
            db_manager.log_system_metric(metric_name, metric_value, labels)
        
        logger.info("Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False


def reset_database(config: Optional[APIConfig] = None, 
                  confirm: bool = False) -> bool:
    """
    Reset database by dropping all tables and recreating schema.
    
    WARNING: This will delete all data!
    
    Args:
        config: Optional API configuration
        confirm: Must be True to actually perform reset
        
    Returns:
        True if successful, False otherwise
    """
    if not confirm:
        logger.error("Database reset requires explicit confirmation")
        return False
    
    try:
        if config is None:
            config = get_api_config()
        
        logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST!")
        
        # Create migrator and reset
        migrator = create_migrator(config)
        
        # Drop all tables
        from .database import Base
        Base.metadata.drop_all(bind=migrator.engine)
        
        # Recreate schema
        Base.metadata.create_all(bind=migrator.engine)
        
        # Run migrations
        if not migrator.migrate_to_latest():
            logger.error("Failed to run migrations after reset")
            return False
        
        logger.info("Database reset completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False


def backup_database(config: Optional[APIConfig] = None,
                   backup_path: Optional[str] = None) -> Optional[str]:
    """
    Create a backup of the database.
    
    Args:
        config: Optional API configuration
        backup_path: Optional backup file path
        
    Returns:
        Path to backup file if successful, None otherwise
    """
    try:
        if config is None:
            config = get_api_config()
        
        # Only works with SQLite databases
        if not config.database_url.startswith("sqlite"):
            logger.error("Database backup only supported for SQLite databases")
            return None
        
        # Extract database file path
        db_file = config.database_url.replace("sqlite:///", "")
        
        if not os.path.exists(db_file):
            logger.error(f"Database file not found: {db_file}")
            return None
        
        # Generate backup path if not provided
        if backup_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db_file}.backup_{timestamp}"
        
        # Copy database file
        import shutil
        shutil.copy2(db_file, backup_path)
        
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return None


def get_database_info(config: Optional[APIConfig] = None) -> Dict[str, Any]:
    """
    Get information about the current database state.
    
    Args:
        config: Optional API configuration
        
    Returns:
        Dictionary with database information
    """
    try:
        if config is None:
            config = get_api_config()
        
        db_manager = get_database_manager(config)
        migrator = create_migrator(config)
        
        # Get basic info
        info = {
            "database_url": config.database_url,
            "health_check": db_manager.health_check(),
            "current_version": migrator.get_current_version(),
            "applied_migrations": migrator.get_applied_migrations(),
            "schema_validation": migrator.validate_schema()
        }
        
        # Get table statistics
        try:
            from datetime import datetime, timedelta
            
            # Get prediction statistics
            prediction_stats = db_manager.get_prediction_stats()
            info["prediction_stats"] = prediction_stats
            
            # Get recent predictions count
            recent_predictions = db_manager.get_predictions(limit=10)
            info["recent_predictions_count"] = len(recent_predictions)
            
        except Exception as e:
            logger.warning(f"Failed to get database statistics: {e}")
            info["stats_error"] = str(e)
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    """Command-line interface for database initialization."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Database initialization utility")
    parser.add_argument("command", choices=["init", "reset", "backup", "info"],
                       help="Command to execute")
    parser.add_argument("--database-url", help="Database URL override")
    parser.add_argument("--no-migrations", action="store_true", 
                       help="Skip running migrations")
    parser.add_argument("--sample-data", action="store_true",
                       help="Create sample data")
    parser.add_argument("--confirm", action="store_true",
                       help="Confirm destructive operations")
    parser.add_argument("--backup-path", help="Backup file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    if args.command == "init":
        success = initialize_database(
            config=config,
            run_migrations=not args.no_migrations,
            create_sample_data=args.sample_data
        )
        sys.exit(0 if success else 1)
    
    elif args.command == "reset":
        success = reset_database(config=config, confirm=args.confirm)
        sys.exit(0 if success else 1)
    
    elif args.command == "backup":
        backup_path = backup_database(config=config, backup_path=args.backup_path)
        sys.exit(0 if backup_path else 1)
    
    elif args.command == "info":
        info = get_database_info(config=config)
        import json
        print(json.dumps(info, indent=2, default=str))