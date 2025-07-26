"""
Pydantic Validation Models

This module provides comprehensive Pydantic models for request/response validation
with advanced validation logic for California Housing data edge cases and constraints.
"""

import re
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import PositiveFloat, PositiveInt
from enum import Enum


class ValidationErrorType(str, Enum):
    """Types of validation errors."""
    FIELD_REQUIRED = "field_required"
    FIELD_TYPE = "field_type"
    FIELD_RANGE = "field_range"
    FIELD_FORMAT = "field_format"
    FIELD_CONSTRAINT = "field_constraint"
    MODEL_CONSTRAINT = "model_constraint"
    BUSINESS_LOGIC = "business_logic"


class ModelStage(str, Enum):
    """MLflow model stages."""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    NONE = "None"


class PredictionStatus(str, Enum):
    """Prediction request status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL_SUCCESS = "partial_success"


class HousingPredictionRequest(BaseModel):
    """
    Comprehensive request model for California Housing price prediction.
    
    Includes advanced validation for all housing features with custom validators
    for edge cases and business logic constraints.
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
    )
    
    # Median income in block group (in tens of thousands of dollars)
    MedInc: float = Field(
        ...,
        ge=0.4999,  # Minimum observed value with small buffer
        le=15.0001,  # Maximum observed value with small buffer
        description="Median income in block group (in tens of thousands of dollars)",
        json_schema_extra={
            "example": 8.3252,
            "unit": "tens_of_thousands_usd",
            "typical_range": "2.0 - 12.0"
        }
    )
    
    # Median house age in block group (in years)
    HouseAge: float = Field(
        ...,
        ge=1.0,
        le=52.0,
        description="Median house age in block group (in years)",
        json_schema_extra={
            "example": 41.0,
            "unit": "years",
            "typical_range": "5.0 - 45.0"
        }
    )
    
    # Average number of rooms per household
    AveRooms: float = Field(
        ...,
        ge=0.8461,  # Minimum observed value with small buffer
        le=141.9091,  # Maximum observed value with small buffer
        description="Average number of rooms per household",
        json_schema_extra={
            "example": 6.984127,
            "unit": "rooms_per_household",
            "typical_range": "3.0 - 10.0"
        }
    )
    
    # Average number of bedrooms per household
    AveBedrms: float = Field(
        ...,
        ge=0.3333,  # Minimum observed value with small buffer
        le=34.0667,  # Maximum observed value with small buffer
        description="Average number of bedrooms per household",
        json_schema_extra={
            "example": 1.023810,
            "unit": "bedrooms_per_household",
            "typical_range": "0.8 - 2.0"
        }
    )
    
    # Block group population
    Population: float = Field(
        ...,
        ge=3.0,
        le=35682.0,
        description="Block group population",
        json_schema_extra={
            "example": 322.0,
            "unit": "people",
            "typical_range": "500.0 - 5000.0"
        }
    )
    
    # Average number of household members
    AveOccup: float = Field(
        ...,
        ge=0.6923,  # Minimum observed value with small buffer
        le=1243.3333,  # Maximum observed value with small buffer
        description="Average number of household members",
        json_schema_extra={
            "example": 2.555556,
            "unit": "people_per_household",
            "typical_range": "2.0 - 5.0"
        }
    )
    
    # Block group latitude
    Latitude: float = Field(
        ...,
        ge=32.54,
        le=41.95,
        description="Block group latitude (California bounds)",
        json_schema_extra={
            "example": 37.88,
            "unit": "degrees",
            "typical_range": "33.0 - 40.0"
        }
    )
    
    # Block group longitude
    Longitude: float = Field(
        ...,
        ge=-124.35,
        le=-114.31,
        description="Block group longitude (California bounds)",
        json_schema_extra={
            "example": -122.23,
            "unit": "degrees",
            "typical_range": "-124.0 - -115.0"
        }
    )
    
    # Optional model version specification
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use for prediction",
        pattern=r"^[a-zA-Z0-9\-_.]+$",
        max_length=50,
        json_schema_extra={
            "example": "v1.2.3"
        }
    )
    
    # Optional request metadata
    request_id: Optional[str] = Field(
        None,
        description="Optional request identifier for tracking",
        max_length=100,
        json_schema_extra={
            "example": "req_12345"
        }
    )
    
    @field_validator('MedInc')
    @classmethod
    def validate_median_income(cls, v: float) -> float:
        """Validate median income with business logic."""
        if v < 0.5:
            raise ValueError("Median income cannot be less than $5,000 annually")
        if v > 15.0:
            raise ValueError("Median income exceeds reasonable maximum for California housing data")
        
        # Warning for unusual values (but don't reject)
        if v < 1.0:
            # This would be logged in a real application
            pass  # Very low income area
        elif v > 12.0:
            # This would be logged in a real application
            pass  # Very high income area
        
        return v
    
    @field_validator('HouseAge')
    @classmethod
    def validate_house_age(cls, v: float) -> float:
        """Validate house age with business logic."""
        if v < 1.0:
            raise ValueError("House age must be at least 1 year")
        if v > 52.0:
            raise ValueError("House age exceeds maximum in California housing dataset")
        
        return v
    
    @field_validator('AveRooms')
    @classmethod
    def validate_average_rooms(cls, v: float) -> float:
        """Validate average rooms per household."""
        if v < 1.0:
            raise ValueError("Average rooms per household must be at least 1.0")
        if v > 50.0:
            raise ValueError("Average rooms per household seems unreasonably high")
        
        return v
    
    @field_validator('AveBedrms')
    @classmethod
    def validate_average_bedrooms(cls, v: float) -> float:
        """Validate average bedrooms per household."""
        if v < 0.1:
            raise ValueError("Average bedrooms per household must be at least 0.1")
        if v > 10.0:
            raise ValueError("Average bedrooms per household seems unreasonably high")
        
        return v
    
    @field_validator('Population')
    @classmethod
    def validate_population(cls, v: float) -> float:
        """Validate block group population."""
        if v < 1.0:
            raise ValueError("Population must be at least 1 person")
        if v > 50000.0:
            raise ValueError("Population exceeds reasonable maximum for a block group")
        
        return v
    
    @field_validator('AveOccup')
    @classmethod
    def validate_average_occupancy(cls, v: float) -> float:
        """Validate average occupancy per household."""
        if v < 0.5:
            raise ValueError("Average occupancy must be at least 0.5 people per household")
        if v > 20.0:
            raise ValueError("Average occupancy seems unreasonably high")
        
        return v
    
    @field_validator('Latitude')
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude within California bounds."""
        if not (32.0 <= v <= 42.0):
            raise ValueError("Latitude must be within California bounds (32.0 to 42.0 degrees)")
        
        return v
    
    @field_validator('Longitude')
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude within California bounds."""
        if not (-125.0 <= v <= -114.0):
            raise ValueError("Longitude must be within California bounds (-125.0 to -114.0 degrees)")
        
        return v
    
    @model_validator(mode='after')
    def validate_housing_relationships(self) -> 'HousingPredictionRequest':
        """Validate relationships between housing features."""
        
        # Bedrooms should generally be less than total rooms
        if self.AveBedrms > self.AveRooms:
            raise ValueError(
                f"Average bedrooms ({self.AveBedrms}) cannot exceed average rooms ({self.AveRooms})"
            )
        
        # Reasonable bedroom to room ratio
        if self.AveRooms > 0 and (self.AveBedrms / self.AveRooms) > 0.8:
            raise ValueError(
                "Bedroom to room ratio seems unreasonably high (>80%)"
            )
        
        # Population density check
        if self.Population > 0 and self.AveOccup > 0:
            estimated_households = self.Population / self.AveOccup
            if estimated_households < 1.0:
                raise ValueError(
                    "Population and average occupancy values are inconsistent"
                )
        
        # Geographic consistency checks for known California regions
        # Northern California (San Francisco Bay Area)
        if 37.0 <= self.Latitude <= 38.5 and -123.0 <= self.Longitude <= -121.0:
            if self.MedInc < 3.0:
                # This would be a warning in production, not an error
                pass  # Unusually low income for Bay Area
        
        # Southern California (Los Angeles area)
        elif 33.5 <= self.Latitude <= 34.5 and -119.0 <= self.Longitude <= -117.0:
            if self.MedInc < 2.0:
                # This would be a warning in production, not an error
                pass  # Unusually low income for LA area
        
        return self


class PredictionResponse(BaseModel):
    """Response model for single housing price prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 4.526,
                "model_version": "v1.2.3",
                "model_stage": "Production",
                "confidence_interval": [4.1, 4.9],
                "processing_time_ms": 15.2,
                "request_id": "req_12345",
                "timestamp": "2024-01-15T10:30:00Z",
                "features_used": 8,
                "model_info": {
                    "algorithm": "XGBoost",
                    "training_date": "2024-01-10T00:00:00Z",
                    "performance_metrics": {
                        "r2_score": 0.85,
                        "rmse": 0.65,
                        "mae": 0.48
                    }
                }
            }
        }
    )
    
    # Predicted house value (in hundreds of thousands of dollars)
    prediction: float = Field(
        ...,
        description="Predicted median house value (in hundreds of thousands of dollars)",
        json_schema_extra={
            "unit": "hundreds_of_thousands_usd",
            "typical_range": "0.5 - 5.0"
        }
    )
    
    # Model metadata
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction"
    )
    
    model_stage: ModelStage = Field(
        ...,
        description="MLflow stage of the model used"
    )
    
    # Confidence information
    confidence_interval: Optional[Tuple[float, float]] = Field(
        None,
        description="95% confidence interval for the prediction [lower, upper]"
    )
    
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0.0 to 1.0)"
    )
    
    # Performance metadata
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )
    
    # Request tracking
    request_id: str = Field(
        ...,
        description="Unique identifier for this prediction request"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when prediction was made"
    )
    
    # Feature information
    features_used: PositiveInt = Field(
        ...,
        description="Number of features used in prediction"
    )
    
    # Optional model information
    model_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional model information and performance metrics"
    )
    
    # Validation warnings (non-blocking issues)
    warnings: Optional[List[str]] = Field(
        None,
        description="Non-critical validation warnings"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch housing price predictions."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "MedInc": 8.3252,
                        "HouseAge": 41.0,
                        "AveRooms": 6.984127,
                        "AveBedrms": 1.023810,
                        "Population": 322.0,
                        "AveOccup": 2.555556,
                        "Latitude": 37.88,
                        "Longitude": -122.23
                    }
                ],
                "model_version": "v1.2.3",
                "return_confidence": True,
                "batch_id": "batch_12345"
            }
        }
    )
    
    predictions: List[HousingPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,  # Configurable batch size limit
        description="List of housing prediction requests"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Model version to use for all predictions in batch"
    )
    
    return_confidence: bool = Field(
        False,
        description="Whether to return confidence intervals for predictions"
    )
    
    batch_id: Optional[str] = Field(
        None,
        description="Optional batch identifier for tracking"
    )
    
    @field_validator('predictions')
    @classmethod
    def validate_batch_size(cls, v: List[HousingPredictionRequest]) -> List[HousingPredictionRequest]:
        """Validate batch size constraints."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 predictions")
        if len(v) == 0:
            raise ValueError("Batch must contain at least one prediction request")
        
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch housing price predictions."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "prediction": 4.526,
                        "model_version": "v1.2.3",
                        "model_stage": "Production",
                        "processing_time_ms": 15.2,
                        "request_id": "req_12345_0",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "features_used": 8
                    }
                ],
                "batch_id": "batch_12345",
                "total_predictions": 1,
                "successful_predictions": 1,
                "failed_predictions": 0,
                "total_processing_time_ms": 15.2,
                "status": "success",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    predictions: List[Union[PredictionResponse, 'PredictionError']] = Field(
        ...,
        description="List of prediction results or errors"
    )
    
    batch_id: Optional[str] = Field(
        None,
        description="Batch identifier if provided in request"
    )
    
    # Batch statistics
    total_predictions: PositiveInt = Field(
        ...,
        description="Total number of predictions requested"
    )
    
    successful_predictions: int = Field(
        ...,
        ge=0,
        description="Number of successful predictions"
    )
    
    failed_predictions: int = Field(
        ...,
        ge=0,
        description="Number of failed predictions"
    )
    
    # Performance metrics
    total_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time for entire batch"
    )
    
    average_processing_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Average processing time per prediction"
    )
    
    # Status and metadata
    status: PredictionStatus = Field(
        ...,
        description="Overall batch processing status"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when batch processing completed"
    )
    
    # Optional warnings and errors summary
    warnings: Optional[List[str]] = Field(
        None,
        description="Non-critical warnings from batch processing"
    )
    
    errors_summary: Optional[Dict[str, int]] = Field(
        None,
        description="Summary of error types encountered"
    )


class ModelInfo(BaseModel):
    """Model information and metadata."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "california-housing-model",
                "version": "v1.2.3",
                "stage": "Production",
                "algorithm": "XGBoost",
                "framework": "xgboost",
                "training_date": "2024-01-10T00:00:00Z",
                "features": [
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"
                ],
                "performance_metrics": {
                    "r2_score": 0.85,
                    "rmse": 0.65,
                    "mae": 0.48,
                    "training_samples": 16512,
                    "validation_samples": 4128
                },
                "model_size_mb": 2.5,
                "gpu_accelerated": True,
                "last_updated": "2024-01-10T00:00:00Z"
            }
        }
    )
    
    # Basic model information
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: ModelStage = Field(..., description="Model stage in MLflow")
    
    # Algorithm information
    algorithm: str = Field(..., description="Machine learning algorithm used")
    framework: str = Field(..., description="ML framework (e.g., xgboost, pytorch)")
    
    # Training information
    training_date: datetime = Field(..., description="When the model was trained")
    features: List[str] = Field(..., description="List of feature names")
    
    # Performance metrics
    performance_metrics: Dict[str, Union[float, int]] = Field(
        ...,
        description="Model performance metrics"
    )
    
    # Technical details
    model_size_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="Model size in megabytes"
    )
    
    gpu_accelerated: bool = Field(
        False,
        description="Whether model uses GPU acceleration"
    )
    
    # Metadata
    last_updated: datetime = Field(
        ...,
        description="When model information was last updated"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Model description"
    )
    
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Model tags and metadata"
    )


class PredictionError(BaseModel):
    """Error information for failed predictions."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_type": "validation_error",
                "error_code": "FIELD_RANGE",
                "message": "Latitude must be within California bounds",
                "field": "Latitude",
                "value": 50.0,
                "request_id": "req_12345",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    error_type: ValidationErrorType = Field(
        ...,
        description="Type of validation error"
    )
    
    error_code: str = Field(
        ...,
        description="Specific error code"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    field: Optional[str] = Field(
        None,
        description="Field that caused the error"
    )
    
    value: Optional[Any] = Field(
        None,
        description="Value that caused the error"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request ID associated with the error"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )


class ValidationErrorResponse(BaseModel):
    """Detailed validation error response."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "errors": [
                    {
                        "error_type": "field_range",
                        "error_code": "FIELD_RANGE",
                        "message": "Latitude must be within California bounds",
                        "field": "Latitude",
                        "value": 50.0
                    }
                ],
                "request_id": "req_12345",
                "timestamp": "2024-01-15T10:30:00Z",
                "path": "/predict",
                "method": "POST"
            }
        }
    )
    
    error: str = Field(
        default="validation_error",
        description="Error type identifier"
    )
    
    message: str = Field(
        ...,
        description="General error message"
    )
    
    errors: List[PredictionError] = Field(
        ...,
        description="Detailed list of validation errors"
    )
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    
    path: Optional[str] = Field(
        None,
        description="API endpoint path"
    )
    
    method: Optional[str] = Field(
        None,
        description="HTTP method"
    )


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds")
    
    # Optional detailed health information
    model_status: Optional[str] = Field(None, description="Model loading status")
    gpu_available: Optional[bool] = Field(None, description="GPU availability")
    memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Memory usage in MB")
    
    # Service dependencies
    dependencies: Optional[Dict[str, str]] = Field(
        None,
        description="Status of external dependencies"
    )


# Export all models for easy importing
__all__ = [
    "HousingPredictionRequest",
    "PredictionResponse", 
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ModelInfo",
    "PredictionError",
    "ValidationErrorResponse",
    "HealthCheckResponse",
    "ValidationErrorType",
    "ModelStage",
    "PredictionStatus"
]