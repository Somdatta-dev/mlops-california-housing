"""
Tests for Prediction API Endpoints

This module provides comprehensive tests for the prediction API endpoints
including single predictions, batch predictions, and model info.
"""

import pytest
import json
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import (
    HousingPredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, ModelInfo, PredictionStatus, ModelStage
)
from src.api.database import PredictionLogData
from src.api.model_loader import ModelInfo as LoaderModelInfo


@pytest.fixture
def client():
    """Create test client."""
    from unittest.mock import Mock
    
    # Create a mock app with state
    test_app = TestClient(app)
    
    # Mock the app state
    mock_state = Mock()
    mock_state.model_loader = None
    mock_state.database_manager = None
    mock_state.metrics = None
    mock_state.api_config = None
    
    test_app.app.state = mock_state
    
    return test_app


@pytest.fixture
def sample_housing_data():
    """Sample housing data for testing."""
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


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    import numpy as np
    model = Mock()
    model.predict.return_value = np.array([4.526])  # Mock prediction value as numpy array
    return model


@pytest.fixture
def mock_model_info():
    """Mock model info for testing."""
    return LoaderModelInfo(
        name="california-housing-model",
        version="v1.2.3",
        stage="Production",
        model_type="XGBoost",
        features=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
        performance_metrics={"r2_score": 0.85, "rmse": 0.65, "mae": 0.48},
        load_time=datetime.utcnow(),
        model_uri="models:/california-housing-model/Production",
        run_id="test_run_id"
    )


class TestSinglePrediction:
    """Test cases for single prediction endpoint."""
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_single_prediction_success(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                     client, sample_housing_data, mock_model, mock_model_info):
        """Test successful single prediction."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_db.log_prediction.return_value = True
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.post("/predict/", json=sample_housing_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        assert data["prediction"] == 4.526
        assert data["model_version"] == "v1.2.3"
        assert data["model_stage"] == "Production"
        assert data["features_used"] == 8
        
        # Verify mocks were called
        mock_loader.get_current_model.assert_called_once()
        mock_model.predict.assert_called_once()
        mock_db.log_prediction.assert_called_once()
        mock_metrics.record_prediction.assert_called_once()
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_single_prediction_no_model(self, mock_get_metrics, mock_get_db, mock_get_loader, client, sample_housing_data):
        """Test single prediction when no model is available."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (None, None)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.post("/predict/", json=sample_housing_data)
        
        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "Model service is currently unavailable" in data["message"]
        
        # Verify error was recorded
        mock_metrics.record_error.assert_called_once_with("model_not_available", "predict_single")
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_single_prediction_model_error(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                         client, sample_housing_data, mock_model_info):
        """Test single prediction when model prediction fails."""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_db.log_prediction.return_value = True
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.post("/predict/", json=sample_housing_data)
        
        # Assertions
        assert response.status_code == 422
        data = response.json()
        assert "Prediction failed" in data["message"]
        
        # Verify error was recorded
        mock_metrics.record_error.assert_called_once_with("prediction_failed", "predict_single")
    
    def test_single_prediction_validation_error(self, client):
        """Test single prediction with invalid input data."""
        invalid_data = {
            "MedInc": -1.0,  # Invalid negative income
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
        
        response = client.post("/predict/", json=invalid_data)
        
        # Should get validation error
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "validation_error"
        assert "details" in data
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_single_prediction_with_request_id(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                              client, sample_housing_data, mock_model, mock_model_info):
        """Test single prediction with custom request ID."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_db.log_prediction.return_value = True
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Add custom request ID
        request_data = sample_housing_data.copy()
        request_data["request_id"] = "custom_request_123"
        
        # Make request
        response = client.post("/predict/", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "custom_request_123"


class TestBatchPrediction:
    """Test cases for batch prediction endpoint."""
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_batch_prediction_success(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                    client, sample_housing_data, mock_model, mock_model_info):
        """Test successful batch prediction."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_db.log_prediction.return_value = True
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Create batch request
        batch_data = {
            "predictions": [sample_housing_data, sample_housing_data],
            "batch_id": "test_batch_123",
            "return_confidence": False
        }
        
        # Make request
        response = client.post("/predict/batch", json=batch_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["batch_id"] == "test_batch_123"
        assert data["total_predictions"] == 2
        assert data["successful_predictions"] == 2
        assert data["failed_predictions"] == 0
        assert data["status"] == "success"
        assert len(data["predictions"]) == 2
        
        # Verify all predictions succeeded
        for prediction in data["predictions"]:
            assert "prediction" in prediction
            assert prediction["prediction"] == 4.526
        
        # Verify mocks were called
        assert mock_model.predict.call_count == 2
        assert mock_db.log_prediction.call_count == 2
        mock_metrics.record_batch_prediction.assert_called_once()
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_batch_prediction_partial_failure(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                             client, sample_housing_data, mock_model_info):
        """Test batch prediction with some failures."""
        # Setup mocks - model fails on second prediction
        import numpy as np
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([4.526]),  # First prediction succeeds
            Exception("Model prediction failed")  # Second prediction fails
        ]
        
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_db.log_prediction.return_value = True
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Create batch request
        batch_data = {
            "predictions": [sample_housing_data, sample_housing_data],
            "batch_id": "test_batch_partial"
        }
        
        # Make request
        response = client.post("/predict/batch", json=batch_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_predictions"] == 2
        assert data["successful_predictions"] == 1
        assert data["failed_predictions"] == 1
        assert data["status"] == "partial_success"
        assert len(data["predictions"]) == 2
        
        # First prediction should succeed, second should be error
        assert "prediction" in data["predictions"][0]
        assert "error_type" in data["predictions"][1]
    
    def test_batch_prediction_validation_error(self, client):
        """Test batch prediction with invalid batch size."""
        # Empty batch should fail validation
        batch_data = {
            "predictions": []
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "validation_error"
    
    def test_batch_prediction_too_large(self, client, sample_housing_data):
        """Test batch prediction with batch size exceeding limit."""
        # Create batch larger than allowed limit (100)
        batch_data = {
            "predictions": [sample_housing_data] * 101
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "validation_error"


class TestModelInfo:
    """Test cases for model info endpoint."""
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_metrics')
    def test_get_model_info_success(self, mock_get_metrics, mock_get_loader, 
                                   client, mock_model, mock_model_info):
        """Test successful model info retrieval."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.get("/predict/model/info")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "california-housing-model"
        assert data["version"] == "v1.2.3"
        assert data["stage"] == "Production"
        assert data["algorithm"] == "XGBoost"
        assert data["framework"] == "mlflow"
        assert data["gpu_accelerated"] == True
        assert len(data["features"]) == 8
        assert "performance_metrics" in data
        
        # Verify mock was called
        mock_loader.get_current_model.assert_called_once()
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_metrics')
    def test_get_model_info_no_model(self, mock_get_metrics, mock_get_loader, client):
        """Test model info when no model is available."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (None, None)
        mock_get_loader.return_value = mock_loader
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.get("/predict/model/info")
        
        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "No model is currently loaded" in data["message"]
        
        # Verify error was recorded
        mock_metrics.record_error.assert_called_once_with("model_not_available", "get_model_info")


class TestPredictionLogging:
    """Test cases for prediction logging functionality."""
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_prediction_logging_success(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                       client, sample_housing_data, mock_model, mock_model_info):
        """Test that predictions are properly logged to database."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        mock_db = Mock()
        mock_db.log_prediction.return_value = True
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.post("/predict/", json=sample_housing_data)
        
        # Assertions
        assert response.status_code == 200
        
        # Verify database logging was called
        mock_db.log_prediction.assert_called_once()
        
        # Check the logged data
        logged_data = mock_db.log_prediction.call_args[0][0]
        assert isinstance(logged_data, PredictionLogData)
        assert logged_data.model_version == "v1.2.3"
        assert logged_data.model_stage == "Production"
        assert logged_data.prediction == 4.526
        assert logged_data.status == "success"
        assert logged_data.input_features == sample_housing_data
    
    @patch('src.api.predictions.get_model_loader')
    @patch('src.api.predictions.get_database_manager')
    @patch('src.api.predictions.get_metrics')
    def test_prediction_logging_failure_handling(self, mock_get_metrics, mock_get_db, mock_get_loader, 
                                                client, sample_housing_data, mock_model, mock_model_info):
        """Test that prediction succeeds even if database logging fails."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.get_current_model.return_value = (mock_model, mock_model_info)
        mock_get_loader.return_value = mock_loader
        
        # Database logging fails
        mock_db = Mock()
        mock_db.log_prediction.side_effect = Exception("Database error")
        mock_get_db.return_value = mock_db
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Make request
        response = client.post("/predict/", json=sample_housing_data)
        
        # Prediction should still succeed even if logging fails
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 4.526
        
        # Verify database logging was attempted
        mock_db.log_prediction.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])