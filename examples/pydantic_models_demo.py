"""
Pydantic Models Demo

This script demonstrates the comprehensive Pydantic validation models
for the California Housing prediction API.
"""

import json
from datetime import datetime
from pydantic import ValidationError

from src.api.models import (
    HousingPredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    PredictionError,
    ValidationErrorResponse,
    HealthCheckResponse,
    ModelStage,
    PredictionStatus,
    ValidationErrorType
)
from src.api.validation_utils import (
    convert_validation_error_to_response,
    validate_housing_data_business_rules,
    get_validation_summary
)


def demo_valid_housing_request():
    """Demonstrate valid housing prediction request."""
    print("=== Valid Housing Prediction Request ===")
    
    valid_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
        "model_version": "v1.2.3",
        "request_id": "demo_request_001"
    }
    
    try:
        request = HousingPredictionRequest(**valid_data)
        print(f"✅ Valid request created successfully")
        print(f"   Median Income: ${request.MedInc * 10000:,.0f}")
        print(f"   Location: ({request.Latitude}, {request.Longitude})")
        print(f"   Model Version: {request.model_version}")
        print(f"   Request ID: {request.request_id}")
        
        # Demonstrate JSON serialization
        json_data = request.model_dump_json(indent=2)
        print(f"\n📄 JSON Representation:")
        print(json_data[:200] + "..." if len(json_data) > 200 else json_data)
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    print()


def demo_validation_errors():
    """Demonstrate validation error handling."""
    print("=== Validation Error Handling ===")
    
    # Invalid data with multiple errors
    invalid_data = {
        "MedInc": -1.0,  # Below minimum
        "HouseAge": 100.0,  # Above maximum
        "AveRooms": 0.5,  # Below minimum
        "AveBedrms": 15.0,  # Above rooms (will fail model validation)
        "Population": 0.0,  # Below minimum
        "AveOccup": 0.1,  # Below minimum
        "Latitude": 50.0,  # Outside California
        "Longitude": -100.0,  # Outside California
        "model_version": "invalid@version!",  # Invalid pattern
        "extra_field": "not_allowed"  # Extra field
    }
    
    try:
        request = HousingPredictionRequest(**invalid_data)
        print("❌ This should not succeed!")
    except ValidationError as e:
        print(f"✅ Validation correctly failed with {len(e.errors())} errors:")
        
        # Convert to standardized error response
        error_response = convert_validation_error_to_response(e)
        
        for i, error in enumerate(error_response.errors[:3], 1):  # Show first 3 errors
            print(f"   {i}. {error.error_type.value}: {error.message}")
            if error.field:
                print(f"      Field: {error.field}, Value: {error.value}")
        
        if len(error_response.errors) > 3:
            print(f"   ... and {len(error_response.errors) - 3} more errors")
        
        # Show validation summary
        summary = get_validation_summary(error_response.errors)
        print(f"\n📊 Error Summary:")
        print(f"   Total Errors: {summary['total_errors']}")
        print(f"   Error Types: {summary['error_types']}")
        print(f"   Fields with Errors: {summary['fields_with_errors']}")
    
    print()


def demo_business_logic_validation():
    """Demonstrate business logic validation."""
    print("=== Business Logic Validation ===")
    
    # Data that passes field validation but fails business rules
    suspicious_data = {
        "MedInc": 0.8,  # Very low income
        "HouseAge": 41.0,
        "AveRooms": 12.0,  # Very high rooms for low income area
        "AveBedrms": 1.0,
        "Population": 2000.0,
        "AveOccup": 50.0,  # Unrealistic occupancy
        "Latitude": 37.88,  # SF Bay Area
        "Longitude": -122.23
    }
    
    try:
        # This will pass Pydantic validation
        request = HousingPredictionRequest(**suspicious_data)
        print("✅ Pydantic validation passed")
        
        # But business logic validation will flag issues
        business_errors = validate_housing_data_business_rules(
            suspicious_data, 
            request_id="demo_business_001"
        )
        
        if business_errors:
            print(f"⚠️  Business logic validation found {len(business_errors)} warnings:")
            for error in business_errors:
                print(f"   - {error.message}")
                if error.details:
                    print(f"     Details: {error.details}")
        else:
            print("✅ Business logic validation passed")
            
    except ValidationError as e:
        print(f"❌ Pydantic validation failed: {e}")
    
    print()


def demo_prediction_response():
    """Demonstrate prediction response model."""
    print("=== Prediction Response Model ===")
    
    response_data = {
        "prediction": 4.526,
        "model_version": "v1.2.3",
        "model_stage": "Production",
        "confidence_interval": [4.1, 4.9],
        "confidence_score": 0.85,
        "processing_time_ms": 15.2,
        "request_id": "demo_request_001",
        "features_used": 8,
        "model_info": {
            "algorithm": "XGBoost",
            "training_date": "2024-01-10T00:00:00Z",
            "performance_metrics": {
                "r2_score": 0.85,
                "rmse": 0.65,
                "mae": 0.48
            }
        },
        "warnings": ["Prediction confidence is moderate"]
    }
    
    try:
        response = PredictionResponse(**response_data)
        print("✅ Prediction response created successfully")
        print(f"   Predicted Value: ${response.prediction * 100000:,.0f}")
        print(f"   Confidence: {response.confidence_score:.1%}")
        print(f"   Processing Time: {response.processing_time_ms:.1f}ms")
        print(f"   Model: {response.model_info['algorithm']} {response.model_version}")
        
        if response.warnings:
            print(f"   Warnings: {', '.join(response.warnings)}")
            
    except ValidationError as e:
        print(f"❌ Response validation failed: {e}")
    
    print()


def demo_batch_prediction():
    """Demonstrate batch prediction models."""
    print("=== Batch Prediction Models ===")
    
    # Create batch request
    batch_data = {
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
            },
            {
                "MedInc": 7.2574,
                "HouseAge": 21.0,
                "AveRooms": 6.238137,
                "AveBedrms": 0.971880,
                "Population": 2401.0,
                "AveOccup": 2.109842,
                "Latitude": 37.86,
                "Longitude": -122.22
            }
        ],
        "model_version": "v1.2.3",
        "return_confidence": True,
        "batch_id": "demo_batch_001"
    }
    
    try:
        batch_request = BatchPredictionRequest(**batch_data)
        print(f"✅ Batch request created with {len(batch_request.predictions)} predictions")
        print(f"   Batch ID: {batch_request.batch_id}")
        print(f"   Return Confidence: {batch_request.return_confidence}")
        
        # Create corresponding batch response
        batch_response_data = {
            "predictions": [
                {
                    "prediction": 4.526,
                    "model_version": "v1.2.3",
                    "model_stage": "Production",
                    "processing_time_ms": 15.2,
                    "request_id": "demo_batch_001_0",
                    "features_used": 8
                },
                {
                    "prediction": 3.847,
                    "model_version": "v1.2.3",
                    "model_stage": "Production",
                    "processing_time_ms": 12.8,
                    "request_id": "demo_batch_001_1",
                    "features_used": 8
                }
            ],
            "batch_id": "demo_batch_001",
            "total_predictions": 2,
            "successful_predictions": 2,
            "failed_predictions": 0,
            "total_processing_time_ms": 28.0,
            "status": "success"
        }
        
        batch_response = BatchPredictionResponse(**batch_response_data)
        print(f"✅ Batch response created successfully")
        print(f"   Status: {batch_response.status}")
        print(f"   Success Rate: {batch_response.successful_predictions}/{batch_response.total_predictions}")
        print(f"   Total Processing Time: {batch_response.total_processing_time_ms:.1f}ms")
        
    except ValidationError as e:
        print(f"❌ Batch validation failed: {e}")
    
    print()


def demo_model_info():
    """Demonstrate model info model."""
    print("=== Model Info Model ===")
    
    model_info_data = {
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
        "last_updated": "2024-01-10T00:00:00Z",
        "description": "GPU-accelerated XGBoost model for California housing price prediction",
        "tags": {
            "environment": "production",
            "version": "1.2.3",
            "gpu": "enabled"
        }
    }
    
    try:
        model_info = ModelInfo(**model_info_data)
        print("✅ Model info created successfully")
        print(f"   Model: {model_info.name} {model_info.version}")
        print(f"   Algorithm: {model_info.algorithm} ({model_info.framework})")
        print(f"   Stage: {model_info.stage}")
        print(f"   Performance: R² = {model_info.performance_metrics['r2_score']:.3f}")
        print(f"   GPU Accelerated: {model_info.gpu_accelerated}")
        print(f"   Size: {model_info.model_size_mb:.1f} MB")
        print(f"   Features: {len(model_info.features)} features")
        
    except ValidationError as e:
        print(f"❌ Model info validation failed: {e}")
    
    print()


def demo_health_check():
    """Demonstrate health check response."""
    print("=== Health Check Response ===")
    
    health_data = {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": 3600.5,
        "model_status": "loaded",
        "gpu_available": True,
        "memory_usage_mb": 512.0,
        "dependencies": {
            "mlflow": "healthy",
            "database": "healthy",
            "gpu": "available"
        }
    }
    
    try:
        health_response = HealthCheckResponse(**health_data)
        print("✅ Health check response created successfully")
        print(f"   Status: {health_response.status}")
        print(f"   Uptime: {health_response.uptime_seconds / 3600:.1f} hours")
        print(f"   Model: {health_response.model_status}")
        print(f"   GPU: {'Available' if health_response.gpu_available else 'Not Available'}")
        print(f"   Memory Usage: {health_response.memory_usage_mb:.0f} MB")
        print(f"   Dependencies: {health_response.dependencies}")
        
    except ValidationError as e:
        print(f"❌ Health check validation failed: {e}")
    
    print()


def main():
    """Run all demonstrations."""
    print("🏠 California Housing Prediction API - Pydantic Models Demo")
    print("=" * 60)
    print()
    
    demo_valid_housing_request()
    demo_validation_errors()
    demo_business_logic_validation()
    demo_prediction_response()
    demo_batch_prediction()
    demo_model_info()
    demo_health_check()
    
    print("✅ All demonstrations completed successfully!")
    print("\nThe Pydantic models provide:")
    print("• Comprehensive field validation with custom validators")
    print("• Advanced business logic validation")
    print("• Detailed error reporting and handling")
    print("• Type safety and automatic documentation")
    print("• JSON serialization/deserialization")
    print("• Edge case handling for California Housing data")


if __name__ == "__main__":
    main()