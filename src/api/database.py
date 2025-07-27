"""
Database Models and Operations

This module provides SQLAlchemy models and database operations for
prediction logging and system metrics tracking.
"""

import os
import time
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

from .config import APIConfig

logger = logging.getLogger(__name__)

Base = declarative_base()


class PredictionLog(Base):
    """Database model for prediction logging."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_stage = Column(String(20), nullable=False)
    input_features = Column(JSON, nullable=False)
    prediction = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    batch_id = Column(String(100), nullable=True, index=True)
    status = Column(String(20), default="success", nullable=False)
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, request_id='{self.request_id}', prediction={self.prediction})>"


class ModelPerformance(Base):
    """Database model for model performance tracking."""
    
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, index=True)
    model_stage = Column(String(20), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    dataset_version = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<ModelPerformance(model_version='{self.model_version}', metric='{self.metric_name}', value={self.metric_value})>"


class SystemMetrics(Base):
    """Database model for system metrics logging."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(50), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    labels = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<SystemMetrics(metric='{self.metric_name}', value={self.metric_value})>"


class PredictionLogData(BaseModel):
    """Pydantic model for prediction log data."""
    
    request_id: str
    model_version: str
    model_stage: str
    input_features: Dict[str, Any]
    prediction: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    confidence_score: Optional[float] = None
    processing_time_ms: float
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    batch_id: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None


class DatabaseManager:
    """
    Database manager for handling database operations.
    
    This class provides methods for database initialization, connection management,
    and CRUD operations for prediction logging and metrics tracking.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database connection and create tables."""
        try:
            # Create engine
            self.engine = create_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,  # Verify connections before use
                connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            logger.info(f"Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def log_prediction(self, prediction_data: PredictionLogData) -> bool:
        """
        Log prediction to database.
        
        Args:
            prediction_data: Prediction data to log
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                prediction_log = PredictionLog(
                    request_id=prediction_data.request_id,
                    model_version=prediction_data.model_version,
                    model_stage=prediction_data.model_stage,
                    input_features=prediction_data.input_features,
                    prediction=prediction_data.prediction,
                    confidence_lower=prediction_data.confidence_lower,
                    confidence_upper=prediction_data.confidence_upper,
                    confidence_score=prediction_data.confidence_score,
                    processing_time_ms=prediction_data.processing_time_ms,
                    user_agent=prediction_data.user_agent,
                    ip_address=prediction_data.ip_address,
                    batch_id=prediction_data.batch_id,
                    status=prediction_data.status,
                    error_message=prediction_data.error_message
                )
                
                session.add(prediction_log)
                session.commit()
                
                logger.debug(f"Logged prediction: {prediction_data.request_id}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to log prediction {prediction_data.request_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error logging prediction {prediction_data.request_id}: {e}")
            return False
    
    def log_model_performance(self, model_version: str, model_stage: str, 
                            metrics: Dict[str, float], dataset_version: Optional[str] = None) -> bool:
        """
        Log model performance metrics.
        
        Args:
            model_version: Model version
            model_stage: Model stage
            metrics: Dictionary of metric name -> value
            dataset_version: Optional dataset version
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                for metric_name, metric_value in metrics.items():
                    performance_log = ModelPerformance(
                        model_version=model_version,
                        model_stage=model_stage,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        dataset_version=dataset_version
                    )
                    session.add(performance_log)
                
                session.commit()
                
                logger.debug(f"Logged performance metrics for model {model_version}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to log model performance: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error logging model performance: {e}")
            return False
    
    def log_system_metric(self, metric_name: str, metric_value: float, 
                         labels: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log system metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Metric value
            labels: Optional labels/tags for the metric
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                system_metric = SystemMetrics(
                    metric_name=metric_name,
                    metric_value=metric_value,
                    labels=labels
                )
                
                session.add(system_metric)
                session.commit()
                
                logger.debug(f"Logged system metric: {metric_name}={metric_value}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to log system metric {metric_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error logging system metric {metric_name}: {e}")
            return False
    
    def get_predictions(self, limit: int = 100, offset: int = 0, 
                       batch_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[PredictionLog]:
        """
        Get prediction logs with filtering and pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            batch_id: Optional batch ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of prediction logs
        """
        try:
            with self.get_session() as session:
                query = session.query(PredictionLog)
                
                # Apply filters
                if batch_id:
                    query = query.filter(PredictionLog.batch_id == batch_id)
                
                if start_time:
                    query = query.filter(PredictionLog.timestamp >= start_time)
                
                if end_time:
                    query = query.filter(PredictionLog.timestamp <= end_time)
                
                # Apply pagination and ordering
                predictions = query.order_by(PredictionLog.timestamp.desc()).offset(offset).limit(limit).all()
                
                # Detach objects from session to avoid DetachedInstanceError
                session.expunge_all()
                
                return predictions
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get predictions: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting predictions: {e}")
            return []
    
    def get_prediction_stats(self, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get prediction statistics.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary with prediction statistics
        """
        try:
            from sqlalchemy import func
            
            with self.get_session() as session:
                query = session.query(PredictionLog)
                
                # Apply time filters
                if start_time:
                    query = query.filter(PredictionLog.timestamp >= start_time)
                
                if end_time:
                    query = query.filter(PredictionLog.timestamp <= end_time)
                
                # Get basic counts
                total_predictions = query.count()
                successful_predictions = query.filter(PredictionLog.status == "success").count()
                failed_predictions = query.filter(PredictionLog.status == "error").count()
                
                # Get average processing time for successful predictions
                avg_processing_time_result = session.query(
                    func.avg(PredictionLog.processing_time_ms)
                ).filter(PredictionLog.status == "success")
                
                # Apply time filters to avg query as well
                if start_time:
                    avg_processing_time_result = avg_processing_time_result.filter(
                        PredictionLog.timestamp >= start_time
                    )
                if end_time:
                    avg_processing_time_result = avg_processing_time_result.filter(
                        PredictionLog.timestamp <= end_time
                    )
                
                avg_processing_time = avg_processing_time_result.scalar() or 0.0
                
                return {
                    "total_predictions": total_predictions,
                    "successful_predictions": successful_predictions,
                    "failed_predictions": failed_predictions,
                    "success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0.0,
                    "average_processing_time_ms": float(avg_processing_time)
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting prediction stats: {e}")
            return {}
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """
        Clean up old records from the database.
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            with self.get_session() as session:
                # Delete old predictions
                deleted_predictions = session.query(PredictionLog).filter(
                    PredictionLog.timestamp < cutoff_date
                ).delete()
                
                # Delete old system metrics
                deleted_metrics = session.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                
                total_deleted = deleted_predictions + deleted_metrics
                logger.info(f"Cleaned up {total_deleted} old records (older than {days_to_keep} days)")
                
                return total_deleted
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}")
            return 0
    
    def health_check(self) -> bool:
        """
        Perform database health check.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            from sqlalchemy import text
            with self.get_session() as session:
                # Simple query to test connection
                session.execute(text("SELECT 1"))
                return True
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[APIConfig] = None) -> DatabaseManager:
    """
    Get global database manager instance.
    
    Args:
        config: Optional API configuration
        
    Returns:
        DatabaseManager instance
    """
    global _database_manager
    
    if _database_manager is None:
        if config is None:
            from .config import get_api_config
            config = get_api_config()
        
        _database_manager = DatabaseManager(config.database_url)
    
    return _database_manager


def initialize_database_manager(config: APIConfig) -> DatabaseManager:
    """
    Initialize global database manager instance.
    
    Args:
        config: API configuration
        
    Returns:
        DatabaseManager instance
    """
    global _database_manager
    
    _database_manager = DatabaseManager(config.database_url)
    
    return _database_manager