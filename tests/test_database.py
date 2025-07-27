"""
Tests for Database Integration and Logging

This module contains comprehensive tests for database operations,
migrations, and logging functionality.
"""

import os
import pytest
import tempfile
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.api.config import APIConfig
from src.api.database import (
    DatabaseManager, PredictionLogData, PredictionLog, 
    ModelPerformance, SystemMetrics, get_database_manager
)
from src.api.migrations import DatabaseMigrator, MigrationVersion
from src.api.database_init import initialize_database, reset_database, backup_database


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        config = APIConfig()
        config.database_url = f"sqlite:///{temp_db.name}"
        
        yield config
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
    
    @pytest.fixture
    def db_manager(self, temp_db_config):
        """Create DatabaseManager instance for testing."""
        return DatabaseManager(temp_db_config.database_url)
    
    @pytest.fixture
    def sample_prediction_data(self):
        """Create sample prediction data for testing."""
        return PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v1.0.0",
            model_stage="Production",
            input_features={
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23
            },
            prediction=4.526,
            confidence_lower=4.1,
            confidence_upper=4.9,
            confidence_score=0.85,
            processing_time_ms=12.5,
            status="success"
        )
    
    def test_database_initialization(self, temp_db_config):
        """Test database initialization and table creation."""
        db_manager = DatabaseManager(temp_db_config.database_url)
        
        # Check that tables were created
        with db_manager.get_session() as session:
            # Test that we can query each table
            predictions = session.query(PredictionLog).count()
            performance = session.query(ModelPerformance).count()
            metrics = session.query(SystemMetrics).count()
            
            assert predictions == 0
            assert performance == 0
            assert metrics == 0
    
    def test_health_check(self, db_manager):
        """Test database health check functionality."""
        assert db_manager.health_check() is True
    
    def test_log_prediction_success(self, db_manager, sample_prediction_data):
        """Test successful prediction logging."""
        result = db_manager.log_prediction(sample_prediction_data)
        assert result is True
        
        # Verify data was logged
        with db_manager.get_session() as session:
            logged_prediction = session.query(PredictionLog).filter(
                PredictionLog.request_id == sample_prediction_data.request_id
            ).first()
            
            assert logged_prediction is not None
            assert logged_prediction.model_version == sample_prediction_data.model_version
            assert logged_prediction.prediction == sample_prediction_data.prediction
            assert logged_prediction.status == "success"
    
    def test_log_prediction_with_batch_id(self, db_manager):
        """Test prediction logging with batch ID."""
        batch_id = str(uuid.uuid4())
        prediction_data = PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v1.0.0",
            model_stage="Production",
            input_features={"test": "data"},
            prediction=1.0,
            processing_time_ms=10.0,
            batch_id=batch_id,
            status="success"
        )
        
        result = db_manager.log_prediction(prediction_data)
        assert result is True
        
        # Verify batch ID was logged
        with db_manager.get_session() as session:
            logged_prediction = session.query(PredictionLog).filter(
                PredictionLog.batch_id == batch_id
            ).first()
            
            assert logged_prediction is not None
            assert logged_prediction.batch_id == batch_id
    
    def test_log_prediction_error(self, db_manager):
        """Test logging prediction with error status."""
        prediction_data = PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v1.0.0",
            model_stage="Production",
            input_features={"test": "data"},
            prediction=0.0,
            processing_time_ms=5.0,
            status="error",
            error_message="Model inference failed"
        )
        
        result = db_manager.log_prediction(prediction_data)
        assert result is True
        
        # Verify error was logged
        with db_manager.get_session() as session:
            logged_prediction = session.query(PredictionLog).filter(
                PredictionLog.request_id == prediction_data.request_id
            ).first()
            
            assert logged_prediction is not None
            assert logged_prediction.status == "error"
            assert logged_prediction.error_message == "Model inference failed"
    
    def test_log_model_performance(self, db_manager):
        """Test model performance logging."""
        metrics = {
            "rmse": 0.654,
            "mae": 0.523,
            "r2_score": 0.789
        }
        
        result = db_manager.log_model_performance(
            model_version="v1.0.0",
            model_stage="Production",
            metrics=metrics,
            dataset_version="test_v1"
        )
        
        assert result is True
        
        # Verify metrics were logged
        with db_manager.get_session() as session:
            logged_metrics = session.query(ModelPerformance).filter(
                ModelPerformance.model_version == "v1.0.0"
            ).all()
            
            assert len(logged_metrics) == 3
            
            metric_dict = {m.metric_name: m.metric_value for m in logged_metrics}
            assert metric_dict["rmse"] == 0.654
            assert metric_dict["mae"] == 0.523
            assert metric_dict["r2_score"] == 0.789
    
    def test_log_system_metric(self, db_manager):
        """Test system metric logging."""
        labels = {"device": "cuda:0", "unit": "percent"}
        
        result = db_manager.log_system_metric(
            metric_name="gpu_utilization",
            metric_value=75.5,
            labels=labels
        )
        
        assert result is True
        
        # Verify metric was logged
        with db_manager.get_session() as session:
            logged_metric = session.query(SystemMetrics).filter(
                SystemMetrics.metric_name == "gpu_utilization"
            ).first()
            
            assert logged_metric is not None
            assert logged_metric.metric_value == 75.5
            assert logged_metric.labels == labels
    
    def test_get_predictions_basic(self, db_manager, sample_prediction_data):
        """Test basic prediction retrieval."""
        # Log a prediction first
        db_manager.log_prediction(sample_prediction_data)
        
        # Retrieve predictions
        predictions = db_manager.get_predictions(limit=10)
        
        assert len(predictions) == 1
        assert predictions[0].request_id == sample_prediction_data.request_id
    
    def test_get_predictions_with_filters(self, db_manager):
        """Test prediction retrieval with filters."""
        batch_id = str(uuid.uuid4())
        
        # Log predictions with different batch IDs
        for i in range(3):
            prediction_data = PredictionLogData(
                request_id=str(uuid.uuid4()),
                model_version="v1.0.0",
                model_stage="Production",
                input_features={"test": f"data_{i}"},
                prediction=float(i),
                processing_time_ms=10.0,
                batch_id=batch_id if i < 2 else str(uuid.uuid4()),
                status="success"
            )
            db_manager.log_prediction(prediction_data)
        
        # Filter by batch ID
        predictions = db_manager.get_predictions(batch_id=batch_id)
        assert len(predictions) == 2
        
        # Test time filtering
        start_time = datetime.utcnow() - timedelta(minutes=1)
        end_time = datetime.utcnow() + timedelta(minutes=1)
        
        predictions = db_manager.get_predictions(
            start_time=start_time,
            end_time=end_time
        )
        assert len(predictions) == 3
    
    def test_get_prediction_stats(self, db_manager):
        """Test prediction statistics retrieval."""
        # Log some predictions with different statuses
        for i in range(5):
            status = "success" if i < 4 else "error"
            prediction_data = PredictionLogData(
                request_id=str(uuid.uuid4()),
                model_version="v1.0.0",
                model_stage="Production",
                input_features={"test": f"data_{i}"},
                prediction=float(i),
                processing_time_ms=10.0 + i,
                status=status
            )
            db_manager.log_prediction(prediction_data)
        
        # Get statistics
        stats = db_manager.get_prediction_stats()
        
        assert stats["total_predictions"] == 5
        assert stats["successful_predictions"] == 4
        assert stats["failed_predictions"] == 1
        assert stats["success_rate"] == 0.8
    
    def test_cleanup_old_records(self, db_manager):
        """Test cleanup of old database records."""
        # Create old prediction record
        old_prediction = PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v1.0.0",
            model_stage="Production",
            input_features={"test": "old_data"},
            prediction=1.0,
            processing_time_ms=10.0,
            status="success"
        )
        
        # Log the prediction
        db_manager.log_prediction(old_prediction)
        
        # Manually update timestamp to be old
        with db_manager.get_session() as session:
            old_record = session.query(PredictionLog).filter(
                PredictionLog.request_id == old_prediction.request_id
            ).first()
            old_record.timestamp = datetime.utcnow() - timedelta(days=35)
            session.commit()
        
        # Log old system metric
        db_manager.log_system_metric("test_metric", 1.0)
        with db_manager.get_session() as session:
            old_metric = session.query(SystemMetrics).filter(
                SystemMetrics.metric_name == "test_metric"
            ).first()
            old_metric.timestamp = datetime.utcnow() - timedelta(days=35)
            session.commit()
        
        # Run cleanup
        deleted_count = db_manager.cleanup_old_records(days_to_keep=30)
        
        assert deleted_count == 2  # 1 prediction + 1 system metric
        
        # Verify records were deleted
        with db_manager.get_session() as session:
            remaining_predictions = session.query(PredictionLog).count()
            remaining_metrics = session.query(SystemMetrics).count()
            
            assert remaining_predictions == 0
            assert remaining_metrics == 0


class TestDatabaseMigrator:
    """Test cases for DatabaseMigrator class."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        config = APIConfig()
        config.database_url = f"sqlite:///{temp_db.name}"
        
        yield config
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
    
    @pytest.fixture
    def migrator(self, temp_db_config):
        """Create DatabaseMigrator instance for testing."""
        return DatabaseMigrator(temp_db_config.database_url)
    
    def test_migration_table_creation(self, migrator):
        """Test that migration version table is created."""
        migrator._ensure_migration_table()
        
        with migrator.SessionLocal() as session:
            count = session.query(MigrationVersion).count()
            assert count == 0  # Table exists but is empty
    
    def test_get_current_version_empty(self, migrator):
        """Test getting current version when no migrations applied."""
        version = migrator.get_current_version()
        assert version is None
    
    def test_apply_migration(self, migrator):
        """Test applying a single migration."""
        migration = migrator.migrations[0]  # Initial schema migration
        
        result = migrator.apply_migration(migration)
        assert result is True
        
        # Check that migration was recorded
        version = migrator.get_current_version()
        assert version == migration["version"]
        
        applied = migrator.get_applied_migrations()
        assert migration["version"] in applied
    
    def test_migrate_to_latest(self, migrator):
        """Test migrating to latest version."""
        result = migrator.migrate_to_latest()
        assert result is True
        
        # Check that all migrations were applied
        applied = migrator.get_applied_migrations()
        expected_migrations = [m["version"] for m in migrator.migrations]
        
        for expected in expected_migrations:
            assert expected in applied
    
    def test_validate_schema_valid(self, migrator):
        """Test schema validation with valid schema."""
        # Apply all migrations first
        migrator.migrate_to_latest()
        
        validation = migrator.validate_schema()
        
        assert validation["valid"] is True
        assert len(validation["missing_tables"]) == 0
        assert len(validation["schema_issues"]) == 0
    
    def test_validate_schema_missing_tables(self, temp_db_config):
        """Test schema validation with missing tables."""
        # Create migrator but don't run migrations
        migrator = DatabaseMigrator(temp_db_config.database_url)
        
        validation = migrator.validate_schema()
        
        assert validation["valid"] is False
        assert "predictions" in validation["missing_tables"]
        assert "model_performance" in validation["missing_tables"]
        assert "system_metrics" in validation["missing_tables"]


class TestDatabaseInitialization:
    """Test cases for database initialization utilities."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        config = APIConfig()
        config.database_url = f"sqlite:///{temp_db.name}"
        
        yield config
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
    
    def test_initialize_database_success(self, temp_db_config):
        """Test successful database initialization."""
        result = initialize_database(
            config=temp_db_config,
            run_migrations=True,
            create_sample_data=False
        )
        
        assert result is True
        
        # Verify database was initialized
        db_manager = DatabaseManager(temp_db_config.database_url)
        assert db_manager.health_check() is True
    
    def test_initialize_database_with_sample_data(self, temp_db_config):
        """Test database initialization with sample data."""
        result = initialize_database(
            config=temp_db_config,
            run_migrations=True,
            create_sample_data=True
        )
        
        assert result is True
        
        # Verify sample data was created
        db_manager = DatabaseManager(temp_db_config.database_url)
        
        with db_manager.get_session() as session:
            predictions = session.query(PredictionLog).count()
            performance = session.query(ModelPerformance).count()
            metrics = session.query(SystemMetrics).count()
            
            assert predictions > 0
            assert performance > 0
            assert metrics > 0
    
    def test_reset_database_without_confirmation(self, temp_db_config):
        """Test that database reset requires confirmation."""
        result = reset_database(config=temp_db_config, confirm=False)
        assert result is False
    
    def test_reset_database_with_confirmation(self, temp_db_config):
        """Test database reset with confirmation."""
        # Initialize database first
        initialize_database(config=temp_db_config, create_sample_data=True)
        
        # Reset database
        result = reset_database(config=temp_db_config, confirm=True)
        assert result is True
        
        # Verify database was reset
        db_manager = DatabaseManager(temp_db_config.database_url)
        
        with db_manager.get_session() as session:
            predictions = session.query(PredictionLog).count()
            performance = session.query(ModelPerformance).count()
            metrics = session.query(SystemMetrics).count()
            
            assert predictions == 0
            assert performance == 0
            assert metrics == 0
    
    def test_backup_database_sqlite(self, temp_db_config):
        """Test database backup for SQLite."""
        # Initialize database with some data
        initialize_database(config=temp_db_config, create_sample_data=True)
        
        # Create backup
        backup_path = backup_database(config=temp_db_config)
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
        
        # Cleanup backup file
        os.unlink(backup_path)
    
    def test_backup_database_non_sqlite(self):
        """Test that backup fails for non-SQLite databases."""
        config = APIConfig()
        config.database_url = "postgresql://user:pass@localhost/db"
        
        backup_path = backup_database(config=config)
        assert backup_path is None


class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        config = APIConfig()
        config.database_url = f"sqlite:///{temp_db.name}"
        
        yield config
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
    
    def test_full_database_lifecycle(self, temp_db_config):
        """Test complete database lifecycle from initialization to cleanup."""
        # Initialize database
        result = initialize_database(
            config=temp_db_config,
            run_migrations=True,
            create_sample_data=True
        )
        assert result is True
        
        # Get database manager
        db_manager = DatabaseManager(temp_db_config.database_url)
        
        # Log additional data
        prediction_data = PredictionLogData(
            request_id=str(uuid.uuid4()),
            model_version="v2.0.0",
            model_stage="Staging",
            input_features={"test": "integration"},
            prediction=2.5,
            processing_time_ms=20.0,
            status="success"
        )
        
        assert db_manager.log_prediction(prediction_data) is True
        
        # Log performance metrics
        metrics = {"accuracy": 0.95, "precision": 0.92}
        assert db_manager.log_model_performance(
            "v2.0.0", "Staging", metrics
        ) is True
        
        # Log system metrics
        assert db_manager.log_system_metric(
            "cpu_usage", 45.2, {"host": "test"}
        ) is True
        
        # Retrieve and verify data
        predictions = db_manager.get_predictions(limit=100)
        assert len(predictions) > 2  # Sample data + our new prediction
        
        stats = db_manager.get_prediction_stats()
        assert stats["total_predictions"] > 2
        assert stats["success_rate"] > 0
        
        # Test cleanup
        deleted_count = db_manager.cleanup_old_records(days_to_keep=0)
        assert deleted_count > 0
        
        # Verify health check
        assert db_manager.health_check() is True


if __name__ == "__main__":
    pytest.main([__file__])