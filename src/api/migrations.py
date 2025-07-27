"""
Database Migration Scripts and Schema Management

This module provides database migration utilities for schema versioning,
upgrades, and data migration operations.
"""

import os
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .database import Base, PredictionLog, ModelPerformance, SystemMetrics
from .config import APIConfig

logger = logging.getLogger(__name__)


class MigrationVersion(Base):
    """Database model for tracking migration versions."""
    
    __tablename__ = "migration_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<MigrationVersion(version='{self.version}', applied_at='{self.applied_at}')>"


class DatabaseMigrator:
    """
    Database migration manager for handling schema changes and data migrations.
    
    This class provides methods for applying migrations, checking schema versions,
    and managing database schema evolution.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database migrator.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Migration definitions
        self.migrations = [
            {
                "version": "001_initial_schema",
                "description": "Create initial database schema with predictions, model_performance, and system_metrics tables",
                "upgrade": self._migration_001_initial_schema,
                "downgrade": self._migration_001_initial_schema_down
            },
            {
                "version": "002_add_batch_support",
                "description": "Add batch_id column to predictions table for batch prediction support",
                "upgrade": self._migration_002_add_batch_support,
                "downgrade": self._migration_002_add_batch_support_down
            },
            {
                "version": "003_add_confidence_metrics",
                "description": "Add confidence score and interval columns to predictions table",
                "upgrade": self._migration_003_add_confidence_metrics,
                "downgrade": self._migration_003_add_confidence_metrics_down
            },
            {
                "version": "004_add_error_tracking",
                "description": "Add status and error_message columns for better error tracking",
                "upgrade": self._migration_004_add_error_tracking,
                "downgrade": self._migration_004_add_error_tracking_down
            }
        ]
    
    def _ensure_migration_table(self) -> None:
        """Ensure migration version tracking table exists."""
        try:
            # Create migration_versions table if it doesn't exist
            MigrationVersion.metadata.create_all(bind=self.engine)
            logger.debug("Migration version table ensured")
        except Exception as e:
            logger.error(f"Failed to create migration version table: {e}")
            raise
    
    def get_current_version(self) -> Optional[str]:
        """
        Get current database schema version.
        
        Returns:
            Current version string or None if no migrations applied
        """
        try:
            self._ensure_migration_table()
            
            with self.SessionLocal() as session:
                latest_migration = session.query(MigrationVersion).order_by(
                    MigrationVersion.applied_at.desc()
                ).first()
                
                return latest_migration.version if latest_migration else None
                
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return None
    
    def get_applied_migrations(self) -> List[str]:
        """
        Get list of applied migration versions.
        
        Returns:
            List of applied migration version strings
        """
        try:
            self._ensure_migration_table()
            
            with self.SessionLocal() as session:
                migrations = session.query(MigrationVersion).order_by(
                    MigrationVersion.applied_at.asc()
                ).all()
                
                return [m.version for m in migrations]
                
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    def apply_migration(self, migration: Dict[str, Any]) -> bool:
        """
        Apply a single migration.
        
        Args:
            migration: Migration definition dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            version = migration["version"]
            description = migration["description"]
            upgrade_func = migration["upgrade"]
            
            logger.info(f"Applying migration {version}: {description}")
            
            # Execute migration
            upgrade_func()
            
            # Record migration in version table
            with self.SessionLocal() as session:
                migration_record = MigrationVersion(
                    version=version,
                    description=description
                )
                session.add(migration_record)
                session.commit()
            
            logger.info(f"Successfully applied migration {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration['version']}: {e}")
            return False
    
    def migrate_to_latest(self) -> bool:
        """
        Apply all pending migrations to bring database to latest version.
        
        Returns:
            True if all migrations successful, False otherwise
        """
        try:
            applied_migrations = set(self.get_applied_migrations())
            
            for migration in self.migrations:
                if migration["version"] not in applied_migrations:
                    if not self.apply_migration(migration):
                        return False
            
            logger.info("Database migrated to latest version")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to latest: {e}")
            return False
    
    def rollback_migration(self, version: str) -> bool:
        """
        Rollback a specific migration.
        
        Args:
            version: Migration version to rollback
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find migration definition
            migration = None
            for m in self.migrations:
                if m["version"] == version:
                    migration = m
                    break
            
            if not migration:
                logger.error(f"Migration {version} not found")
                return False
            
            logger.info(f"Rolling back migration {version}")
            
            # Execute downgrade
            downgrade_func = migration["downgrade"]
            downgrade_func()
            
            # Remove from version table
            with self.SessionLocal() as session:
                session.query(MigrationVersion).filter(
                    MigrationVersion.version == version
                ).delete()
                session.commit()
            
            logger.info(f"Successfully rolled back migration {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False
    
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate current database schema against expected schema.
        
        Returns:
            Dictionary with validation results
        """
        try:
            inspector = inspect(self.engine)
            
            # Get current tables
            current_tables = set(inspector.get_table_names())
            
            # Expected tables
            expected_tables = {
                "predictions",
                "model_performance", 
                "system_metrics",
                "migration_versions"
            }
            
            # Check for missing tables
            missing_tables = expected_tables - current_tables
            extra_tables = current_tables - expected_tables
            
            # Validate table schemas
            schema_issues = []
            
            for table_name in expected_tables.intersection(current_tables):
                columns = inspector.get_columns(table_name)
                column_names = {col['name'] for col in columns}
                
                # Define expected columns for each table
                expected_columns = {
                    "predictions": {
                        "id", "request_id", "model_version", "model_stage",
                        "input_features", "prediction", "confidence_lower",
                        "confidence_upper", "confidence_score", "processing_time_ms",
                        "timestamp", "user_agent", "ip_address", "batch_id",
                        "status", "error_message"
                    },
                    "model_performance": {
                        "id", "model_version", "model_stage", "metric_name",
                        "metric_value", "dataset_version", "timestamp"
                    },
                    "system_metrics": {
                        "id", "metric_name", "metric_value", "labels", "timestamp"
                    },
                    "migration_versions": {
                        "id", "version", "description", "applied_at"
                    }
                }
                
                if table_name in expected_columns:
                    expected_cols = expected_columns[table_name]
                    missing_cols = expected_cols - column_names
                    extra_cols = column_names - expected_cols
                    
                    if missing_cols or extra_cols:
                        schema_issues.append({
                            "table": table_name,
                            "missing_columns": list(missing_cols),
                            "extra_columns": list(extra_cols)
                        })
            
            return {
                "valid": len(missing_tables) == 0 and len(schema_issues) == 0,
                "missing_tables": list(missing_tables),
                "extra_tables": list(extra_tables),
                "schema_issues": schema_issues,
                "current_version": self.get_current_version()
            }
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    # Migration implementations
    
    def _migration_001_initial_schema(self) -> None:
        """Create initial database schema."""
        Base.metadata.create_all(bind=self.engine)
    
    def _migration_001_initial_schema_down(self) -> None:
        """Drop initial database schema."""
        Base.metadata.drop_all(bind=self.engine)
    
    def _migration_002_add_batch_support(self) -> None:
        """Add batch_id column to predictions table."""
        with self.engine.connect() as conn:
            # Check if column already exists
            inspector = inspect(self.engine)
            columns = [col['name'] for col in inspector.get_columns('predictions')]
            
            if 'batch_id' not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN batch_id VARCHAR(100)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_predictions_batch_id ON predictions (batch_id)"))
                conn.commit()
    
    def _migration_002_add_batch_support_down(self) -> None:
        """Remove batch_id column from predictions table."""
        # SQLite doesn't support DROP COLUMN, so we'd need to recreate the table
        # For simplicity, we'll leave the column but log the limitation
        logger.warning("SQLite doesn't support DROP COLUMN. batch_id column will remain.")
    
    def _migration_003_add_confidence_metrics(self) -> None:
        """Add confidence score and interval columns."""
        with self.engine.connect() as conn:
            inspector = inspect(self.engine)
            columns = [col['name'] for col in inspector.get_columns('predictions')]
            
            if 'confidence_score' not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN confidence_score FLOAT"))
            if 'confidence_lower' not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN confidence_lower FLOAT"))
            if 'confidence_upper' not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN confidence_upper FLOAT"))
            
            conn.commit()
    
    def _migration_003_add_confidence_metrics_down(self) -> None:
        """Remove confidence metrics columns."""
        logger.warning("SQLite doesn't support DROP COLUMN. Confidence columns will remain.")
    
    def _migration_004_add_error_tracking(self) -> None:
        """Add status and error_message columns."""
        with self.engine.connect() as conn:
            inspector = inspect(self.engine)
            columns = [col['name'] for col in inspector.get_columns('predictions')]
            
            if 'status' not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN status VARCHAR(20) DEFAULT 'success'"))
            if 'error_message' not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN error_message TEXT"))
            
            conn.commit()
    
    def _migration_004_add_error_tracking_down(self) -> None:
        """Remove error tracking columns."""
        logger.warning("SQLite doesn't support DROP COLUMN. Error tracking columns will remain.")


def create_migrator(config: Optional[APIConfig] = None) -> DatabaseMigrator:
    """
    Create database migrator instance.
    
    Args:
        config: Optional API configuration
        
    Returns:
        DatabaseMigrator instance
    """
    if config is None:
        from .config import get_api_config
        config = get_api_config()
    
    return DatabaseMigrator(config.database_url)


def migrate_database(config: Optional[APIConfig] = None) -> bool:
    """
    Apply all pending database migrations.
    
    Args:
        config: Optional API configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        migrator = create_migrator(config)
        return migrator.migrate_to_latest()
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        return False


def validate_database_schema(config: Optional[APIConfig] = None) -> Dict[str, Any]:
    """
    Validate database schema.
    
    Args:
        config: Optional API configuration
        
    Returns:
        Validation results dictionary
    """
    try:
        migrator = create_migrator(config)
        return migrator.validate_schema()
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        return {"valid": False, "error": str(e)}


if __name__ == "__main__":
    """Command-line interface for database migrations."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration utility")
    parser.add_argument("command", choices=["migrate", "validate", "version", "rollback"],
                       help="Migration command to execute")
    parser.add_argument("--version", help="Migration version (for rollback)")
    parser.add_argument("--database-url", help="Database URL override")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    migrator = create_migrator(config)
    
    if args.command == "migrate":
        success = migrator.migrate_to_latest()
        sys.exit(0 if success else 1)
    
    elif args.command == "validate":
        result = migrator.validate_schema()
        print(f"Schema valid: {result['valid']}")
        if not result['valid']:
            print(f"Issues: {result}")
        sys.exit(0 if result['valid'] else 1)
    
    elif args.command == "version":
        version = migrator.get_current_version()
        print(f"Current version: {version or 'No migrations applied'}")
        applied = migrator.get_applied_migrations()
        print(f"Applied migrations: {applied}")
    
    elif args.command == "rollback":
        if not args.version:
            print("--version required for rollback")
            sys.exit(1)
        success = migrator.rollback_migration(args.version)
        sys.exit(0 if success else 1)