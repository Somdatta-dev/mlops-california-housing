"""
Tests for FastAPI Service Foundation

This module tests the core FastAPI service components including
configuration, metrics, model loading, and health checks.
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Import the modules to test
from src.api.config import APIConfig, ModelConfig, get_api_config, get_model_config
from src.api.metrics import PrometheusMetrics
from src.api.model_loader import ModelLoader, ModelCache, ModelInfo
from src.api.health import get_system_info, get_gpu_info
from src.api.main import create_app


class TestAPIConfig:
    """Test API configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = APIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.title == "MLOps California Housing Prediction API"
        assert config.model_name == "california-housing-model"
        assert config.model_stage == "Production"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid port
        with pytest.raises(ValueError):
            APIConfig(port=0)
        
        with pytest.raises(ValueError):
            APIConfig(port=70000)
        
        # Test invalid log level
        with pytest.raises(ValueError):
            APIConfig(log_level="INVALID")
        
        # Test invalid batch size
        with pytest.raises(ValueError):
            APIConfig(max_batch_size=0)
        
        with pytest.raises(ValueError):
            APIConfig(max_batch_size=2000)
    
    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            'API_HOST': '127.0.0.1',
            'API_PORT': '9000',
            'API_DEBUG': 'true',
            'MODEL_NAME': 'test-model'
        }):
            config = APIConfig()
            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.debug is True
            assert config.model_name == "test-model"


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert len(config.feature_names) == 8
        assert "MedInc" in config.feature_names
        assert "Latitude" in config.feature_names
        
        assert "MedInc" in config.feature_ranges
        assert config.feature_ranges["MedInc"]["min"] == 0.0
        assert config.feature_ranges["MedInc"]["max"] == 15.0
        
        assert "min_r2_score" in config.performance_thresholds
        assert config.performance_thresholds["min_r2_score"] == 0.6


class TestPrometheusMetrics:
    """Test Prometheus metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PrometheusMetrics()
        
        assert metrics.requests_total is not None
        assert metrics.prediction_duration is not None
        assert metrics.model_info is not None
    
    def test_record_request(self):
        """Test request recording."""
        metrics = PrometheusMetrics()
        
        # Record a request
        metrics.record_request("GET", "/health", 200, 0.1)
        
        # Check that metric was recorded (basic test)
        assert metrics.requests_total._value._value > 0
    
    def test_record_prediction(self):
        """Test prediction recording."""
        metrics = PrometheusMetrics()
        
        # Record a prediction
        metrics.record_prediction("v1.0", "single", 0.05, 2.5)
        
        # Check that metrics were recorded
        assert metrics.predictions_total._value._value > 0
    
    def test_set_model_info(self):
        """Test setting model information."""
        metrics = PrometheusMetrics()
        
        features = ["feature1", "feature2"]
        metrics.set_model_info("test-model", "1.0", "Production", "xgboost", features)
        
        # Check that model info was set
        info = metrics.model_info._value
        assert info["name"] == "test-model"
        assert info["version"] == "1.0"
        assert info["stage"] == "Production"
    
    def test_gpu_metrics_unavailable(self):
        """Test GPU metrics when nvidia-ml-py is unavailable."""
        with patch('src.api.metrics.NVIDIA_ML_AVAILABLE', False):
            metrics = PrometheusMetrics()
            
            # Should not have GPU metrics
            assert not hasattr(metrics, 'gpu_utilization')
            
            # Update should not fail
            metrics.update_gpu_metrics()


class TestModelCache:
    """Test model cache."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = ModelCache(ttl_seconds=1)
        
        # Test put and get
        test_model = "test_model"
        cache.put("test_key", test_model)
        
        retrieved = cache.get("test_key")
        assert retrieved == test_model
        
        # Test cache size
        assert cache.size() == 1
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = ModelCache(ttl_seconds=0.1)  # Very short TTL
        
        # Put item in cache
        cache.put("test_key", "test_model")
        
        # Should be available immediately
        assert cache.get("test_key") is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("test_key") is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ModelCache()
        
        cache.put("key1", "model1")
        cache.put("key2", "model2")
        
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0


class TestModelLoader:
    """Test model loader."""
    
    @pytest.fixture
    def mock_api_config(self):
        """Mock API configuration."""
        return APIConfig(
            model_name="test-model",
            model_stage="Production",
            mlflow_tracking_uri="sqlite:///test.db"
        )
    
    @pytest.fixture
    def mock_model_config(self):
        """Mock model configuration."""
        return ModelConfig()
    
    def test_model_loader_initialization(self, mock_api_config, mock_model_config):
        """Test model loader initialization."""
        with patch('src.api.model_loader.MLflowExperimentManager'):
            loader = ModelLoader(mock_api_config, mock_model_config)
            
            assert loader.api_config == mock_api_config
            assert loader.model_config == mock_model_config
            assert loader.cache is not None
    
    def test_cache_key_generation(self, mock_api_config, mock_model_config):
        """Test cache key generation."""
        with patch('src.api.model_loader.MLflowExperimentManager'):
            loader = ModelLoader(mock_api_config, mock_model_config)
            
            key = loader._get_cache_key("test-model", "Production")
            assert key == "test-model:Production"
    
    def test_current_model_operations(self, mock_api_config, mock_model_config):
        """Test current model get/set operations."""
        with patch('src.api.model_loader.MLflowExperimentManager'):
            loader = ModelLoader(mock_api_config, mock_model_config)
            
            # Initially no model
            model, info = loader.get_current_model()
            assert model is None
            assert info is None
            
            # Set a model
            mock_model = Mock()
            mock_info = ModelInfo(
                name="test-model",
                version="1.0",
                stage="Production",
                model_type="xgboost",
                features=["f1", "f2"],
                performance_metrics={"rmse": 0.5},
                load_time=time.time(),
                model_uri="models:/test-model/Production"
            )
            
            loader.set_current_model(mock_model, mock_info)
            
            # Should now have model
            model, info = loader.get_current_model()
            assert model == mock_model
            assert info == mock_info


class TestHealthChecks:
    """Test health check functions."""
    
    def test_get_system_info(self):
        """Test system information retrieval."""
        system_info = get_system_info()
        
        assert system_info.platform is not None
        assert system_info.python_version is not None
        assert system_info.cpu_count > 0
        assert system_info.memory_total_gb > 0
        assert 0 <= system_info.memory_usage_percent <= 100
    
    def test_get_gpu_info_unavailable(self):
        """Test GPU info when not available."""
        with patch('src.api.health.NVIDIA_ML_AVAILABLE', False):
            gpu_info = get_gpu_info()
            assert gpu_info is None


class TestFastAPIApp:
    """Test FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "MLOps California Housing Prediction API" in response.text
    
    def test_info_endpoint(self, client):
        """Test info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    def test_basic_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data
    
    def test_system_health_check(self, client):
        """Test system health check."""
        response = client.get("/health/system")
        assert response.status_code == 200
        
        data = response.json()
        assert "platform" in data
        assert "python_version" in data
        assert "cpu_count" in data
        assert "memory_total_gb" in data
    
    def test_gpu_health_check(self, client):
        """Test GPU health check."""
        response = client.get("/health/gpu")
        assert response.status_code == 200
        
        # Should return None or list of GPU info
        data = response.json()
        assert data is None or isinstance(data, list)
    
    def test_error_handling(self, client):
        """Test error handling."""
        # Test 404
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])