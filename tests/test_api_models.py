"""
Comprehensive Tests for Pydantic API Models

This module provides extensive testing for all Pydantic models including
validation scenarios, edge cases, and error handling.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any
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
    ValidationErrorType,
    ModelStage,
    PredictionStatus
)


class TestHousingPredictionRequest:
    """Test cases for HousingPredictionRequest model."""
    
    @pytest.fixture
    def valid_housing_data(self) -> Dict[str, Any]:
        """Valid housing data for testing."""
        return {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    
    def test_valid_housing_request(self, valid_housing_data):
        """Test creation of valid housing prediction request."""
        request = HousingPredictionRequest(**valid_housing_data)
        
        assert request.MedInc == 8.3252
        assert request.HouseAge == 41.0
        assert request.AveRooms == 6.984127
        assert request.AveBedrms == 1.023810
        assert request.Population == 322.0
        assert request.AveOccup == 2.555556
        assert request.Latitude == 37.88
        assert request.Longitude == -122.23
        assert request.model_version is None
        assert request.request_id is None
    
    def test_valid_housing_request_with_optional_fields(self, valid_housing_data):
        """Test housing request with optional fields."""
        valid_housing_data.update({
            "model_version": "v1.2.3",
            "request_id": "test_request_123"
        })
        
        request = HousingPredictionRequest(**valid_housing_data)
        
        assert request.model_version == "v1.2.3"
        assert request.request_id == "test_request_123"
    
    def test_median_income_validation(self, valid_housing_data):
        """Test median income field validation."""
        # Test minimum boundary
        valid_housing_data["MedInc"] = 0.5
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.MedInc == 0.5
        
        # Test maximum boundary
        valid_housing_data["MedInc"] = 15.0
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.MedInc == 15.0
        
        # Test below minimum (Field constraint)
        valid_housing_data["MedInc"] = 0.4
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test above maximum
        valid_housing_data["MedInc"] = 15.1
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "less than or equal to" in str(exc_info.value)
    
    def test_house_age_validation(self, valid_housing_data):
        """Test house age field validation."""
        # Test minimum boundary
        valid_housing_data["HouseAge"] = 1.0
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.HouseAge == 1.0
        
        # Test maximum boundary
        valid_housing_data["HouseAge"] = 52.0
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.HouseAge == 52.0
        
        # Test below minimum (Field constraint)
        valid_housing_data["HouseAge"] = 0.5
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test above maximum
        valid_housing_data["HouseAge"] = 53.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "less than or equal to" in str(exc_info.value)
    
    def test_average_rooms_validation(self, valid_housing_data):
        """Test average rooms field validation."""
        # Test reasonable value
        valid_housing_data["AveRooms"] = 5.5
        valid_housing_data["AveBedrms"] = 1.0  # Ensure bedrooms < rooms
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.AveRooms == 5.5
        
        # Test minimum boundary
        valid_housing_data["AveRooms"] = 2.0
        valid_housing_data["AveBedrms"] = 1.0  # Ensure bedrooms < rooms
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.AveRooms == 2.0
        
        # Test below minimum (Field constraint)
        valid_housing_data["AveRooms"] = 0.5
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test unreasonably high value
        valid_housing_data["AveRooms"] = 51.0
        valid_housing_data["AveBedrms"] = 1.0  # Keep bedrooms reasonable
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "unreasonably high" in str(exc_info.value)
    
    def test_average_bedrooms_validation(self, valid_housing_data):
        """Test average bedrooms field validation."""
        # Test reasonable value
        valid_housing_data["AveBedrms"] = 1.2
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.AveBedrms == 1.2
        
        # Test minimum boundary (Field constraint minimum is 0.3333)
        valid_housing_data["AveBedrms"] = 0.4
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.AveBedrms == 0.4
        
        # Test below minimum (Field constraint)
        valid_housing_data["AveBedrms"] = 0.1
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test unreasonably high value
        valid_housing_data["AveBedrms"] = 11.0
        valid_housing_data["AveRooms"] = 15.0  # Ensure rooms > bedrooms
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "unreasonably high" in str(exc_info.value)
    
    def test_population_validation(self, valid_housing_data):
        """Test population field validation."""
        # Test reasonable value
        valid_housing_data["Population"] = 1000.0
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.Population == 1000.0
        
        # Test minimum boundary (Field constraint minimum is 3.0)
        valid_housing_data["Population"] = 3.0
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.Population == 3.0
        
        # Test below minimum (Field constraint)
        valid_housing_data["Population"] = 1.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test unreasonably high value
        valid_housing_data["Population"] = 50001.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "less than or equal to" in str(exc_info.value)
    
    def test_average_occupancy_validation(self, valid_housing_data):
        """Test average occupancy field validation."""
        # Test reasonable value
        valid_housing_data["AveOccup"] = 3.0
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.AveOccup == 3.0
        
        # Test minimum boundary (Field constraint minimum is 0.6923)
        valid_housing_data["AveOccup"] = 0.7
        request = HousingPredictionRequest(**valid_housing_data)
        assert request.AveOccup == 0.7
        
        # Test below minimum (Field constraint)
        valid_housing_data["AveOccup"] = 0.5
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test unreasonably high value
        valid_housing_data["AveOccup"] = 21.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "unreasonably high" in str(exc_info.value)
    
    def test_latitude_validation(self, valid_housing_data):
        """Test latitude field validation."""
        # Test valid California latitudes (within field constraints)
        for lat in [32.6, 37.0, 41.9]:
            valid_housing_data["Latitude"] = lat
            request = HousingPredictionRequest(**valid_housing_data)
            assert request.Latitude == lat
        
        # Test below California bounds (Field constraint)
        valid_housing_data["Latitude"] = 31.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test above California bounds
        valid_housing_data["Latitude"] = 43.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "less than or equal to" in str(exc_info.value)
    
    def test_longitude_validation(self, valid_housing_data):
        """Test longitude field validation."""
        # Test valid California longitudes
        for lon in [-124.0, -120.0, -115.0]:
            valid_housing_data["Longitude"] = lon
            request = HousingPredictionRequest(**valid_housing_data)
            assert request.Longitude == lon
        
        # Test below California bounds (too far west) - Field constraint
        valid_housing_data["Longitude"] = -126.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "greater than or equal to" in str(exc_info.value)
        
        # Test above California bounds (too far east)
        valid_housing_data["Longitude"] = -113.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "less than or equal to" in str(exc_info.value)
    
    def test_model_relationships_validation(self, valid_housing_data):
        """Test validation of relationships between housing features."""
        # Test bedrooms exceeding rooms
        valid_housing_data["AveRooms"] = 3.0
        valid_housing_data["AveBedrms"] = 4.0
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "cannot exceed average rooms" in str(exc_info.value)
        
        # Test unreasonable bedroom to room ratio
        valid_housing_data["AveRooms"] = 4.0
        valid_housing_data["AveBedrms"] = 3.5  # 87.5% ratio
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "ratio seems unreasonably high" in str(exc_info.value)
        
        # Test inconsistent population and occupancy
        valid_housing_data["AveRooms"] = 5.0
        valid_housing_data["AveBedrms"] = 1.0
        valid_housing_data["Population"] = 10.0
        valid_housing_data["AveOccup"] = 20.0  # Would result in 0.5 households
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "inconsistent" in str(exc_info.value)
    
    def test_model_version_validation(self, valid_housing_data):
        """Test model version field validation."""
        # Test valid model versions
        valid_versions = ["v1.0.0", "model-v2", "latest", "prod_model_123"]
        for version in valid_versions:
            valid_housing_data["model_version"] = version
            request = HousingPredictionRequest(**valid_housing_data)
            assert request.model_version == version
        
        # Test invalid model version (special characters)
        valid_housing_data["model_version"] = "v1.0@invalid"
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "String should match pattern" in str(exc_info.value)
        
        # Test too long model version
        valid_housing_data["model_version"] = "a" * 51
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "String should have at most 50 characters" in str(exc_info.value)
    
    def test_extra_fields_forbidden(self, valid_housing_data):
        """Test that extra fields are forbidden."""
        valid_housing_data["extra_field"] = "not_allowed"
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**valid_housing_data)
        assert "Extra inputs are not permitted" in str(exc_info.value)
    
    def test_missing_required_fields(self):
        """Test validation when required fields are missing."""
        incomplete_data = {"MedInc": 5.0}  # Missing other required fields
        
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionRequest(**incomplete_data)
        
        errors = exc_info.value.errors()
        required_fields = ["HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        
        for field in required_fields:
            assert any(error["loc"] == (field,) and error["type"] == "missing" for error in errors)
    
    def test_json_serialization(self, valid_housing_data):
        """Test JSON serialization and deserialization."""
        request = HousingPredictionRequest(**valid_housing_data)
        
        # Test serialization
        json_data = request.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["MedInc"] == 8.3252
        
        # Test JSON string serialization
        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        new_request = HousingPredictionRequest(**parsed_data)
        assert new_request.MedInc == request.MedInc


class TestPredictionResponse:
    """Test cases for PredictionResponse model."""
    
    @pytest.fixture
    def valid_response_data(self) -> Dict[str, Any]:
        """Valid prediction response data."""
        return {
            "prediction": 4.526,
            "model_version": "v1.2.3",
            "model_stage": "Production",
            "processing_time_ms": 15.2,
            "request_id": "req_12345",
            "features_used": 8
        }
    
    def test_valid_prediction_response(self, valid_response_data):
        """Test creation of valid prediction response."""
        response = PredictionResponse(**valid_response_data)
        
        assert response.prediction == 4.526
        assert response.model_version == "v1.2.3"
        assert response.model_stage == ModelStage.PRODUCTION
        assert response.processing_time_ms == 15.2
        assert response.request_id == "req_12345"
        assert response.features_used == 8
        assert isinstance(response.timestamp, datetime)
    
    def test_prediction_response_with_optional_fields(self, valid_response_data):
        """Test prediction response with optional fields."""
        valid_response_data.update({
            "confidence_interval": [4.1, 4.9],
            "confidence_score": 0.85,
            "model_info": {
                "algorithm": "XGBoost",
                "r2_score": 0.85
            },
            "warnings": ["Low confidence in prediction"]
        })
        
        response = PredictionResponse(**valid_response_data)
        
        assert response.confidence_interval == (4.1, 4.9)
        assert response.confidence_score == 0.85
        assert response.model_info["algorithm"] == "XGBoost"
        assert response.warnings == ["Low confidence in prediction"]
    
    def test_invalid_confidence_score(self, valid_response_data):
        """Test validation of confidence score bounds."""
        # Test below minimum
        valid_response_data["confidence_score"] = -0.1
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**valid_response_data)
        assert "greater than or equal to 0" in str(exc_info.value)
        
        # Test above maximum
        valid_response_data["confidence_score"] = 1.1
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**valid_response_data)
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_invalid_processing_time(self, valid_response_data):
        """Test validation of processing time."""
        valid_response_data["processing_time_ms"] = -1.0
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**valid_response_data)
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_features_used(self, valid_response_data):
        """Test validation of features used count."""
        valid_response_data["features_used"] = 0
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**valid_response_data)
        assert "greater than 0" in str(exc_info.value)


class TestBatchPredictionRequest:
    """Test cases for BatchPredictionRequest model."""
    
    @pytest.fixture
    def valid_batch_data(self) -> Dict[str, Any]:
        """Valid batch prediction request data."""
        return {
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
            ]
        }
    
    def test_valid_batch_request(self, valid_batch_data):
        """Test creation of valid batch prediction request."""
        request = BatchPredictionRequest(**valid_batch_data)
        
        assert len(request.predictions) == 2
        assert request.predictions[0].MedInc == 8.3252
        assert request.predictions[1].MedInc == 7.2574
        assert request.model_version is None
        assert request.return_confidence is False
        assert request.batch_id is None
    
    def test_batch_request_with_optional_fields(self, valid_batch_data):
        """Test batch request with optional fields."""
        valid_batch_data.update({
            "model_version": "v1.2.3",
            "return_confidence": True,
            "batch_id": "batch_12345"
        })
        
        request = BatchPredictionRequest(**valid_batch_data)
        
        assert request.model_version == "v1.2.3"
        assert request.return_confidence is True
        assert request.batch_id == "batch_12345"
    
    def test_empty_batch_validation(self):
        """Test validation of empty batch."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(predictions=[])
        assert "List should have at least 1 item" in str(exc_info.value)
    
    def test_oversized_batch_validation(self):
        """Test validation of oversized batch."""
        # Create a batch with 101 predictions (exceeds limit)
        large_batch = {
            "predictions": [
                {
                    "MedInc": 8.0,
                    "HouseAge": 40.0,
                    "AveRooms": 6.0,
                    "AveBedrms": 1.0,
                    "Population": 300.0,
                    "AveOccup": 2.5,
                    "Latitude": 37.0,
                    "Longitude": -122.0
                }
            ] * 101
        }
        
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(**large_batch)
        assert "at most 100 items" in str(exc_info.value)


class TestBatchPredictionResponse:
    """Test cases for BatchPredictionResponse model."""
    
    @pytest.fixture
    def valid_batch_response_data(self) -> Dict[str, Any]:
        """Valid batch prediction response data."""
        return {
            "predictions": [
                {
                    "prediction": 4.526,
                    "model_version": "v1.2.3",
                    "model_stage": "Production",
                    "processing_time_ms": 15.2,
                    "request_id": "req_12345_0",
                    "features_used": 8
                },
                {
                    "prediction": 3.847,
                    "model_version": "v1.2.3",
                    "model_stage": "Production",
                    "processing_time_ms": 12.8,
                    "request_id": "req_12345_1",
                    "features_used": 8
                }
            ],
            "total_predictions": 2,
            "successful_predictions": 2,
            "failed_predictions": 0,
            "total_processing_time_ms": 28.0,
            "status": "success"
        }
    
    def test_valid_batch_response(self, valid_batch_response_data):
        """Test creation of valid batch prediction response."""
        response = BatchPredictionResponse(**valid_batch_response_data)
        
        assert len(response.predictions) == 2
        assert response.total_predictions == 2
        assert response.successful_predictions == 2
        assert response.failed_predictions == 0
        assert response.total_processing_time_ms == 28.0
        assert response.status == PredictionStatus.SUCCESS
        assert isinstance(response.timestamp, datetime)
    
    def test_batch_response_with_optional_fields(self, valid_batch_response_data):
        """Test batch response with optional fields."""
        valid_batch_response_data.update({
            "batch_id": "batch_12345",
            "average_processing_time_ms": 14.0,
            "warnings": ["Some predictions had low confidence"],
            "errors_summary": {"validation_error": 1}
        })
        
        response = BatchPredictionResponse(**valid_batch_response_data)
        
        assert response.batch_id == "batch_12345"
        assert response.average_processing_time_ms == 14.0
        assert response.warnings == ["Some predictions had low confidence"]
        assert response.errors_summary == {"validation_error": 1}


class TestModelInfo:
    """Test cases for ModelInfo model."""
    
    @pytest.fixture
    def valid_model_info_data(self) -> Dict[str, Any]:
        """Valid model info data."""
        return {
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
            "last_updated": "2024-01-10T00:00:00Z"
        }
    
    def test_valid_model_info(self, valid_model_info_data):
        """Test creation of valid model info."""
        model_info = ModelInfo(**valid_model_info_data)
        
        assert model_info.name == "california-housing-model"
        assert model_info.version == "v1.2.3"
        assert model_info.stage == ModelStage.PRODUCTION
        assert model_info.algorithm == "XGBoost"
        assert model_info.framework == "xgboost"
        assert len(model_info.features) == 8
        assert model_info.performance_metrics["r2_score"] == 0.85
        assert model_info.gpu_accelerated is False  # Default value
    
    def test_model_info_with_optional_fields(self, valid_model_info_data):
        """Test model info with optional fields."""
        valid_model_info_data.update({
            "model_size_mb": 2.5,
            "gpu_accelerated": True,
            "description": "GPU-accelerated XGBoost model for California housing prediction",
            "tags": {"environment": "production", "version": "1.2.3"}
        })
        
        model_info = ModelInfo(**valid_model_info_data)
        
        assert model_info.model_size_mb == 2.5
        assert model_info.gpu_accelerated is True
        assert "GPU-accelerated" in model_info.description
        assert model_info.tags["environment"] == "production"
    
    def test_invalid_model_size(self, valid_model_info_data):
        """Test validation of model size."""
        valid_model_info_data["model_size_mb"] = -1.0
        with pytest.raises(ValidationError) as exc_info:
            ModelInfo(**valid_model_info_data)
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_description_length_validation(self, valid_model_info_data):
        """Test validation of description length."""
        valid_model_info_data["description"] = "a" * 501  # Exceeds 500 character limit
        with pytest.raises(ValidationError) as exc_info:
            ModelInfo(**valid_model_info_data)
        assert "String should have at most 500 characters" in str(exc_info.value)


class TestPredictionError:
    """Test cases for PredictionError model."""
    
    def test_valid_prediction_error(self):
        """Test creation of valid prediction error."""
        error = PredictionError(
            error_type=ValidationErrorType.FIELD_RANGE,
            error_code="LATITUDE_OUT_OF_BOUNDS",
            message="Latitude must be within California bounds",
            field="Latitude",
            value=50.0,
            request_id="req_12345"
        )
        
        assert error.error_type == ValidationErrorType.FIELD_RANGE
        assert error.error_code == "LATITUDE_OUT_OF_BOUNDS"
        assert error.message == "Latitude must be within California bounds"
        assert error.field == "Latitude"
        assert error.value == 50.0
        assert error.request_id == "req_12345"
        assert isinstance(error.timestamp, datetime)
    
    def test_prediction_error_with_optional_fields(self):
        """Test prediction error with optional fields."""
        error = PredictionError(
            error_type=ValidationErrorType.MODEL_CONSTRAINT,
            error_code="BEDROOM_ROOM_RATIO",
            message="Bedroom to room ratio is unreasonable",
            details={
                "bedrooms": 4.0,
                "rooms": 3.0,
                "ratio": 1.33
            }
        )
        
        assert error.error_type == ValidationErrorType.MODEL_CONSTRAINT
        assert error.details["ratio"] == 1.33
        assert error.field is None
        assert error.value is None


class TestValidationErrorResponse:
    """Test cases for ValidationErrorResponse model."""
    
    def test_valid_validation_error_response(self):
        """Test creation of valid validation error response."""
        errors = [
            PredictionError(
                error_type=ValidationErrorType.FIELD_RANGE,
                error_code="LATITUDE_OUT_OF_BOUNDS",
                message="Latitude out of bounds",
                field="Latitude",
                value=50.0
            )
        ]
        
        response = ValidationErrorResponse(
            message="Request validation failed",
            errors=errors,
            path="/predict",
            method="POST"
        )
        
        assert response.error == "validation_error"
        assert response.message == "Request validation failed"
        assert len(response.errors) == 1
        assert response.errors[0].error_type == ValidationErrorType.FIELD_RANGE
        assert response.path == "/predict"
        assert response.method == "POST"
        assert isinstance(response.timestamp, datetime)
        # request_id should be auto-generated
        assert response.request_id is not None


class TestHealthCheckResponse:
    """Test cases for HealthCheckResponse model."""
    
    def test_valid_health_check_response(self):
        """Test creation of valid health check response."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.uptime_seconds == 3600.5
        assert isinstance(response.timestamp, datetime)
    
    def test_health_check_with_optional_fields(self):
        """Test health check response with optional fields."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5,
            model_status="loaded",
            gpu_available=True,
            memory_usage_mb=512.0,
            dependencies={
                "mlflow": "healthy",
                "database": "healthy",
                "gpu": "available"
            }
        )
        
        assert response.model_status == "loaded"
        assert response.gpu_available is True
        assert response.memory_usage_mb == 512.0
        assert response.dependencies["mlflow"] == "healthy"
    
    def test_invalid_uptime(self):
        """Test validation of uptime."""
        with pytest.raises(ValidationError) as exc_info:
            HealthCheckResponse(
                status="healthy",
                version="1.0.0",
                uptime_seconds=-1.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_memory_usage(self):
        """Test validation of memory usage."""
        with pytest.raises(ValidationError) as exc_info:
            HealthCheckResponse(
                status="healthy",
                version="1.0.0",
                uptime_seconds=100.0,
                memory_usage_mb=-1.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)


class TestEnumValidation:
    """Test cases for enum validation."""
    
    def test_validation_error_type_enum(self):
        """Test ValidationErrorType enum values."""
        assert ValidationErrorType.FIELD_REQUIRED == "field_required"
        assert ValidationErrorType.FIELD_TYPE == "field_type"
        assert ValidationErrorType.FIELD_RANGE == "field_range"
        assert ValidationErrorType.FIELD_FORMAT == "field_format"
        assert ValidationErrorType.FIELD_CONSTRAINT == "field_constraint"
        assert ValidationErrorType.MODEL_CONSTRAINT == "model_constraint"
        assert ValidationErrorType.BUSINESS_LOGIC == "business_logic"
    
    def test_model_stage_enum(self):
        """Test ModelStage enum values."""
        assert ModelStage.STAGING == "Staging"
        assert ModelStage.PRODUCTION == "Production"
        assert ModelStage.ARCHIVED == "Archived"
        assert ModelStage.NONE == "None"
    
    def test_prediction_status_enum(self):
        """Test PredictionStatus enum values."""
        assert PredictionStatus.SUCCESS == "success"
        assert PredictionStatus.ERROR == "error"
        assert PredictionStatus.PARTIAL_SUCCESS == "partial_success"


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    def test_extreme_california_coordinates(self):
        """Test extreme but valid California coordinates."""
        # Extreme north California
        data = {
            "MedInc": 5.0,
            "HouseAge": 25.0,
            "AveRooms": 5.0,
            "AveBedrms": 1.0,
            "Population": 1000.0,
            "AveOccup": 2.5,
            "Latitude": 41.95,  # Northern border
            "Longitude": -124.35  # Western border
        }
        request = HousingPredictionRequest(**data)
        assert request.Latitude == 41.95
        assert request.Longitude == -124.35
        
        # Extreme south California
        data.update({
            "Latitude": 32.54,  # Southern border
            "Longitude": -114.31  # Eastern border
        })
        request = HousingPredictionRequest(**data)
        assert request.Latitude == 32.54
        assert request.Longitude == -114.31
    
    def test_minimum_valid_values(self):
        """Test minimum valid values for all fields."""
        data = {
            "MedInc": 0.5,  # Just above minimum
            "HouseAge": 1.0,
            "AveRooms": 1.0,
            "AveBedrms": 0.4,  # Above field constraint minimum of 0.3333
            "Population": 3.0,  # Above field constraint minimum of 3.0
            "AveOccup": 0.7,  # Above field constraint minimum of 0.6923
            "Latitude": 32.54,
            "Longitude": -124.35
        }
        request = HousingPredictionRequest(**data)
        assert request.MedInc == 0.5
        assert request.Population == 3.0
    
    def test_maximum_valid_values(self):
        """Test maximum valid values for all fields."""
        data = {
            "MedInc": 15.0,
            "HouseAge": 52.0,
            "AveRooms": 20.0,  # Reasonable maximum
            "AveBedrms": 5.0,  # Reasonable maximum
            "Population": 35682.0,
            "AveOccup": 10.0,  # Reasonable maximum
            "Latitude": 41.95,
            "Longitude": -114.31
        }
        request = HousingPredictionRequest(**data)
        assert request.MedInc == 15.0
        assert request.HouseAge == 52.0
    
    def test_floating_point_precision(self):
        """Test handling of floating point precision."""
        data = {
            "MedInc": 8.325234567890123,  # High precision
            "HouseAge": 41.0,
            "AveRooms": 6.984127456789,
            "AveBedrms": 1.023810123456,
            "Population": 322.0,
            "AveOccup": 2.555556789012,
            "Latitude": 37.88123456789,
            "Longitude": -122.23987654321
        }
        request = HousingPredictionRequest(**data)
        # Values should be preserved with reasonable precision
        assert abs(request.MedInc - 8.325234567890123) < 1e-10
        assert abs(request.Longitude - (-122.23987654321)) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])