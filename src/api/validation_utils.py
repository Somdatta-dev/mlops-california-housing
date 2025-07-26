"""
Validation Utilities

This module provides utility functions for handling Pydantic validation errors
and converting them to standardized error responses.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import ValidationError
from fastapi import Request

from .models import (
    PredictionError,
    ValidationErrorResponse,
    ValidationErrorType
)


def convert_pydantic_error_to_prediction_error(
    error: Dict[str, Any],
    request_id: Optional[str] = None
) -> PredictionError:
    """
    Convert a single Pydantic validation error to a PredictionError.
    
    Args:
        error: Pydantic error dictionary
        request_id: Optional request identifier
        
    Returns:
        PredictionError instance
    """
    error_type = error.get("type", "unknown")
    field_path = ".".join(str(loc) for loc in error.get("loc", []))
    
    # Map Pydantic error types to our ValidationErrorType enum
    error_type_mapping = {
        "missing": ValidationErrorType.FIELD_REQUIRED,
        "type_error": ValidationErrorType.FIELD_TYPE,
        "value_error": ValidationErrorType.FIELD_CONSTRAINT,
        "greater_than_equal": ValidationErrorType.FIELD_RANGE,
        "less_than_equal": ValidationErrorType.FIELD_RANGE,
        "string_pattern_mismatch": ValidationErrorType.FIELD_FORMAT,
        "string_too_long": ValidationErrorType.FIELD_CONSTRAINT,
        "string_too_short": ValidationErrorType.FIELD_CONSTRAINT,
        "too_long": ValidationErrorType.FIELD_CONSTRAINT,
        "too_short": ValidationErrorType.FIELD_CONSTRAINT,
        "extra_forbidden": ValidationErrorType.FIELD_CONSTRAINT
    }
    
    mapped_error_type = error_type_mapping.get(error_type, ValidationErrorType.FIELD_CONSTRAINT)
    
    # Generate error code
    error_code = f"{error_type.upper()}_{field_path.upper().replace('.', '_')}" if field_path else error_type.upper()
    
    # Extract field and value information
    field = field_path if field_path else None
    input_value = error.get("input")
    
    # Create human-readable message
    message = error.get("msg", "Validation error occurred")
    
    return PredictionError(
        error_type=mapped_error_type,
        error_code=error_code,
        message=message,
        field=field,
        value=input_value,
        request_id=request_id,
        details={
            "pydantic_error_type": error_type,
            "location": error.get("loc", []),
            "context": error.get("ctx", {})
        }
    )


def convert_validation_error_to_response(
    validation_error: ValidationError,
    request: Optional[Request] = None,
    request_id: Optional[str] = None
) -> ValidationErrorResponse:
    """
    Convert a Pydantic ValidationError to a ValidationErrorResponse.
    
    Args:
        validation_error: Pydantic ValidationError
        request: Optional FastAPI Request object
        request_id: Optional request identifier
        
    Returns:
        ValidationErrorResponse instance
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # Convert each Pydantic error to a PredictionError
    prediction_errors = []
    for error in validation_error.errors():
        prediction_error = convert_pydantic_error_to_prediction_error(error, request_id)
        prediction_errors.append(prediction_error)
    
    # Extract request information if available
    path = None
    method = None
    if request:
        path = str(request.url.path)
        method = request.method
    
    return ValidationErrorResponse(
        message="Request validation failed",
        errors=prediction_errors,
        request_id=request_id,
        path=path,
        method=method
    )


def create_business_logic_error(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> PredictionError:
    """
    Create a business logic validation error.
    
    Args:
        message: Error message
        field: Field that caused the error
        value: Value that caused the error
        request_id: Optional request identifier
        details: Additional error details
        
    Returns:
        PredictionError instance
    """
    error_code = f"BUSINESS_LOGIC_{field.upper()}" if field else "BUSINESS_LOGIC_ERROR"
    
    return PredictionError(
        error_type=ValidationErrorType.BUSINESS_LOGIC,
        error_code=error_code,
        message=message,
        field=field,
        value=value,
        request_id=request_id,
        details=details or {}
    )


def validate_housing_data_business_rules(
    data: Dict[str, Any],
    request_id: Optional[str] = None
) -> List[PredictionError]:
    """
    Validate business rules for housing data that go beyond basic field validation.
    
    Args:
        data: Housing data dictionary
        request_id: Optional request identifier
        
    Returns:
        List of PredictionError instances (empty if no errors)
    """
    errors = []
    
    # Extract values with defaults
    med_inc = data.get("MedInc", 0)
    house_age = data.get("HouseAge", 0)
    ave_rooms = data.get("AveRooms", 0)
    ave_bedrms = data.get("AveBedrms", 0)
    population = data.get("Population", 0)
    ave_occup = data.get("AveOccup", 0)
    latitude = data.get("Latitude", 0)
    longitude = data.get("Longitude", 0)
    
    # Business rule: Very low income areas should have reasonable housing characteristics
    if med_inc < 1.0 and ave_rooms > 8.0:
        errors.append(create_business_logic_error(
            message="Very low income areas typically don't have large average room counts",
            field="MedInc",
            value=med_inc,
            request_id=request_id,
            details={"ave_rooms": ave_rooms, "threshold": 8.0}
        ))
    
    # Business rule: Very old houses should have reasonable room configurations
    if house_age > 45.0 and ave_rooms > 10.0:
        errors.append(create_business_logic_error(
            message="Very old houses typically don't have extremely large room counts",
            field="HouseAge",
            value=house_age,
            request_id=request_id,
            details={"ave_rooms": ave_rooms, "age_threshold": 45.0, "room_threshold": 10.0}
        ))
    
    # Business rule: Population density consistency
    if population > 0 and ave_occup > 0:
        estimated_households = population / ave_occup
        if estimated_households < 10 and population > 1000:
            errors.append(create_business_logic_error(
                message="Population and occupancy values suggest unrealistic household density",
                field="Population",
                value=population,
                request_id=request_id,
                details={
                    "ave_occup": ave_occup,
                    "estimated_households": estimated_households,
                    "min_expected_households": 10
                }
            ))
    
    # Business rule: Geographic income consistency (basic checks)
    # San Francisco Bay Area (high cost area)
    if (37.0 <= latitude <= 38.5 and -123.0 <= longitude <= -121.0):
        if med_inc < 2.0:
            errors.append(create_business_logic_error(
                message="Income level seems unusually low for San Francisco Bay Area",
                field="MedInc",
                value=med_inc,
                request_id=request_id,
                details={"region": "SF_Bay_Area", "expected_min_income": 2.0}
            ))
    
    # Los Angeles Area
    elif (33.5 <= latitude <= 34.5 and -119.0 <= longitude <= -117.0):
        if med_inc < 1.5:
            errors.append(create_business_logic_error(
                message="Income level seems unusually low for Los Angeles area",
                field="MedInc",
                value=med_inc,
                request_id=request_id,
                details={"region": "LA_Area", "expected_min_income": 1.5}
            ))
    
    return errors


def get_validation_summary(errors: List[PredictionError]) -> Dict[str, Any]:
    """
    Generate a summary of validation errors.
    
    Args:
        errors: List of PredictionError instances
        
    Returns:
        Dictionary with error summary statistics
    """
    if not errors:
        return {"total_errors": 0, "error_types": {}, "fields_with_errors": []}
    
    error_types = {}
    fields_with_errors = set()
    
    for error in errors:
        # Count error types
        error_type = error.error_type.value
        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Track fields with errors
        if error.field:
            fields_with_errors.add(error.field)
    
    return {
        "total_errors": len(errors),
        "error_types": error_types,
        "fields_with_errors": list(fields_with_errors),
        "most_common_error_type": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
    }


def format_validation_error_for_logging(
    validation_error: ValidationError,
    request_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a validation error for structured logging.
    
    Args:
        validation_error: Pydantic ValidationError
        request_data: Optional request data that caused the error
        
    Returns:
        Dictionary formatted for logging
    """
    errors = []
    for error in validation_error.errors():
        errors.append({
            "type": error.get("type"),
            "location": error.get("loc"),
            "message": error.get("msg"),
            "input": error.get("input")
        })
    
    log_data = {
        "validation_error": True,
        "error_count": len(errors),
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if request_data:
        # Don't log sensitive data, just structure info
        log_data["request_structure"] = {
            "fields": list(request_data.keys()) if isinstance(request_data, dict) else None,
            "data_type": type(request_data).__name__
        }
    
    return log_data


# Validation decorators for common use cases
def validate_housing_prediction_request(func):
    """
    Decorator to validate housing prediction requests with enhanced error handling.
    
    This decorator can be applied to FastAPI endpoint functions to provide
    standardized validation error handling.
    """
    from functools import wraps
    from fastapi import HTTPException
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            # Convert to standardized error response
            error_response = convert_validation_error_to_response(e)
            raise HTTPException(
                status_code=422,
                detail=error_response.model_dump()
            )
    
    return wrapper


# Export all utility functions
__all__ = [
    "convert_pydantic_error_to_prediction_error",
    "convert_validation_error_to_response",
    "create_business_logic_error",
    "validate_housing_data_business_rules",
    "get_validation_summary",
    "format_validation_error_for_logging",
    "validate_housing_prediction_request"
]