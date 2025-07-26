"""
FastAPI Configuration Management

This module provides configuration management for the FastAPI service,
including environment-based settings, model configuration, and service parameters.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class APIConfig(BaseModel):
    """Configuration class for FastAPI service."""
    
    # Server Configuration
    host: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"),
        description="API server host"
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000")),
        description="API server port"
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("API_DEBUG", "false").lower() == "true",
        description="Enable debug mode"
    )
    reload: bool = Field(
        default_factory=lambda: os.getenv("API_RELOAD", "false").lower() == "true",
        description="Enable auto-reload in development"
    )
    
    # API Metadata
    title: str = Field(
        default="MLOps California Housing Prediction API",
        description="API title"
    )
    description: str = Field(
        default="GPU-accelerated machine learning API for California Housing price prediction with MLflow integration",
        description="API description"
    )
    version: str = Field(
        default_factory=lambda: os.getenv("API_VERSION", "1.0.0"),
        description="API version"
    )
    
    # Model Configuration
    model_name: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "california-housing-model"),
        description="MLflow registered model name"
    )
    model_stage: str = Field(
        default_factory=lambda: os.getenv("MODEL_STAGE", "Production"),
        description="MLflow model stage to load"
    )
    model_fallback_stage: str = Field(
        default_factory=lambda: os.getenv("MODEL_FALLBACK_STAGE", "Staging"),
        description="Fallback model stage if primary stage is not available"
    )
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        description="MLflow tracking server URI"
    )
    mlflow_registry_uri: Optional[str] = Field(
        default_factory=lambda: os.getenv("MLFLOW_REGISTRY_URI"),
        description="MLflow model registry URI"
    )
    
    # Database Configuration
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./predictions.db"),
        description="Database connection URL"
    )
    
    # Monitoring Configuration
    enable_prometheus: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
        description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(
        default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "8001")),
        description="Prometheus metrics server port"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    enable_structured_logging: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true",
        description="Enable structured JSON logging"
    )
    
    # Performance Configuration
    max_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "100")),
        description="Maximum batch size for batch predictions"
    )
    request_timeout: float = Field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "30.0")),
        description="Request timeout in seconds"
    )
    
    # Security Configuration
    cors_origins: List[str] = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","),
        description="CORS allowed origins"
    )
    api_key_header: str = Field(
        default_factory=lambda: os.getenv("API_KEY_HEADER", "X-API-Key"),
        description="API key header name"
    )
    
    @field_validator('port', 'prometheus_port')
    @classmethod
    def validate_port(cls, v):
        """Validate port numbers."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('max_batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        if v > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        return v


class ModelConfig(BaseModel):
    """Configuration for model-specific settings."""
    
    # Feature names for California Housing dataset
    feature_names: List[str] = Field(
        default=[
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"
        ],
        description="Expected feature names for the model"
    )
    
    # Feature validation ranges
    feature_ranges: Dict[str, Dict[str, float]] = Field(
        default={
            "MedInc": {"min": 0.0, "max": 15.0},
            "HouseAge": {"min": 1.0, "max": 52.0},
            "AveRooms": {"min": 1.0, "max": 20.0},
            "AveBedrms": {"min": 0.0, "max": 5.0},
            "Population": {"min": 3.0, "max": 35682.0},
            "AveOccup": {"min": 0.5, "max": 1243.0},
            "Latitude": {"min": 32.54, "max": 41.95},
            "Longitude": {"min": -124.35, "max": -114.31}
        },
        description="Valid ranges for each feature"
    )
    
    # Model performance thresholds
    performance_thresholds: Dict[str, float] = Field(
        default={
            "min_r2_score": 0.6,
            "max_rmse": 1.0,
            "max_mae": 0.8
        },
        description="Performance thresholds for model validation"
    )


def get_api_config() -> APIConfig:
    """
    Get API configuration with environment variable overrides.
    
    Returns:
        APIConfig instance
    """
    return APIConfig()


def get_model_config() -> ModelConfig:
    """
    Get model configuration.
    
    Returns:
        ModelConfig instance
    """
    return ModelConfig()


def setup_logging(config: APIConfig) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: API configuration object
    """
    import json
    import sys
    from datetime import datetime
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up structured logging if enabled
    if config.enable_structured_logging:
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                 'pathname', 'filename', 'module', 'lineno', 
                                 'funcName', 'created', 'msecs', 'relativeCreated', 
                                 'thread', 'threadName', 'processName', 'process',
                                 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                        log_entry[key] = value
                
                return json.dumps(log_entry)
        
        # Apply structured formatter to all handlers
        for handler in logging.root.handlers:
            handler.setFormatter(StructuredFormatter())
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured with level: {config.log_level}")