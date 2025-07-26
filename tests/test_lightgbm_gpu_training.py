#!/usr/bin/env python3
"""
Unit tests for LightGBM GPU training implementation.

This module tests the comprehensive LightGBM training functionality including:
- GPU acceleration and optimized parameters
- Configuration validation and serialization
- Model evaluation and performance comparison utilities
- MLflow integration for experiment tracking
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.gpu_model_trainer import (
    GPUModelTrainer, ModelConfig, LightGBMConfig, GPUMonitor
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig


class TestLightGBMGPUTraining:
    """Test suite for LightGBM GPU training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data for testing."""
        X, y = make_regression(
            n_samples=1000,
            n_features=8,
            noise=0.1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    @pytest.fixture
    def lightgbm_config(self):
        """Create LightGBM configuration for testing."""
        return LightGBMConfig(
            device_type='cpu',  # Use CPU for testing
            n_estimators=100,   # Minimum allowed value
            num_leaves=31,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=10,
            random_state=42
        )
    
    @pytest.fixture
    def model_config(self, lightgbm_config):
        """Create model configuration for testing."""
        return ModelConfig(lightgbm=lightgbm_config)
    
    @pytest.fixture
    def mock_mlflow_manager(self):
        """Create mock MLflow manager for testing."""
        mock_manager = Mock(spec=MLflowExperimentManager)
        mock_manager.client = Mock()
        mock_manager.client.active_run.return_value.info.run_id = "test_run_id"
        mock_manager.log_parameters = Mock()
        mock_manager.log_metrics = Mock()
        mock_manager.log_model = Mock()
        mock_manager.log_artifacts = Mock()
        return mock_manager
    
    @pytest.fixture
    def temp_plots_dir(self):
        """Create temporary directory for plots."""
        temp_dir = tempfile.mkdtemp()
        plots_dir = Path(temp_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        yield plots_dir
        shutil.rmtree(temp_dir)
    
    def test_lightgbm_config_validation(self):
        """Test LightGBM configuration validation."""
        # Valid configuration
        config = LightGBMConfig(
            device_type='gpu',
            n_estimators=1000,
            num_leaves=255,
            learning_rate=0.01
        )
        
        assert config.device_type == 'gpu'
        assert config.n_estimators == 1000
        assert config.num_leaves == 255
        assert config.learning_rate == 0.01
        
        # Invalid device type
        with pytest.raises(ValueError, match="device_type must be one of"):
            LightGBMConfig(device_type='invalid_device')
        
        # Test parameter bounds
        config = LightGBMConfig(
            num_leaves=500,
            max_depth=15,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.9
        )
        
        assert config.num_leaves == 500
        assert config.max_depth == 15
        assert config.learning_rate == 0.05
    
    def test_lightgbm_config_serialization(self):
        """Test LightGBM configuration serialization."""
        config = LightGBMConfig(
            device_type='gpu',
            n_estimators=2000,
            num_leaves=255,
            learning_rate=0.01,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        # Test model_dump
        config_dict = config.model_dump()
        
        assert config_dict['device_type'] == 'gpu'
        assert config_dict['n_estimators'] == 2000
        assert config_dict['num_leaves'] == 255
        assert config_dict['learning_rate'] == 0.01
        assert config_dict['reg_alpha'] == 0.1
        assert config_dict['reg_lambda'] == 1.0
        
        # Test JSON serialization
        config_json = json.dumps(config_dict)
        loaded_config = json.loads(config_json)
        
        assert loaded_config['device_type'] == 'gpu'
        assert loaded_config['n_estimators'] == 2000
    
    def test_lightgbm_gpu_parameters(self):
        """Test LightGBM GPU-specific parameters."""
        # Create GPU configuration
        gpu_config = LightGBMConfig(
            device_type='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_estimators=100
        )
        
        assert gpu_config.device_type == 'gpu'
        assert gpu_config.gpu_platform_id == 0
        assert gpu_config.gpu_device_id == 0
        assert gpu_config.n_estimators == 100
        
        # Test CPU configuration
        cpu_config = LightGBMConfig(
            device_type='cpu',
            n_estimators=200
        )
        
        assert cpu_config.device_type == 'cpu'
        assert cpu_config.n_estimators == 200
    
    def test_model_config_integration(self, lightgbm_config):
        """Test LightGBM integration with ModelConfig."""
        model_config = ModelConfig(lightgbm=lightgbm_config)
        
        assert isinstance(model_config.lightgbm, LightGBMConfig)
        assert model_config.lightgbm.device_type == 'cpu'
        assert model_config.lightgbm.n_estimators == 100
        
        # Test serialization
        config_dict = model_config.model_dump()
        assert 'lightgbm' in config_dict
        assert config_dict['lightgbm']['device_type'] == 'cpu'
    
    def test_gpu_monitor_initialization(self):
        """Test GPU monitor initialization."""
        monitor = GPUMonitor()
        
        # Test device info retrieval
        device_info = monitor.get_device_info()
        assert isinstance(device_info, dict)
        assert 'available' in device_info
        
        # Test metrics retrieval (may return None if no GPU)
        metrics = monitor.get_metrics()
        if metrics is not None:
            assert hasattr(metrics, 'utilization_percent')
            assert hasattr(metrics, 'memory_used_mb')
            assert hasattr(metrics, 'temperature_celsius')
    
    def test_trainer_initialization(self, model_config, mock_mlflow_manager):
        """Test GPUModelTrainer initialization with LightGBM config."""
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        assert trainer.config == model_config
        assert trainer.mlflow_manager == mock_mlflow_manager
        assert isinstance(trainer.gpu_monitor, GPUMonitor)
        
        # Test device info
        device_info = trainer.get_device_info()
        assert isinstance(device_info, dict)
        
        # Test GPU availability check
        gpu_available = trainer.is_gpu_available()
        assert isinstance(gpu_available, bool)
    
    def test_lightgbm_prediction_compatibility(self):
        """Test prediction compatibility with both old and new model structures."""
        # Create mock trainer
        mock_config = ModelConfig()
        mock_mlflow = Mock()
        trainer = GPUModelTrainer(mock_config, mock_mlflow)
        
        # Test with old structure (direct model)
        mock_old_model = Mock()
        mock_old_model.predict.return_value = np.array([1.0, 2.0, 3.0])
        
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        predictions_old = trainer._predict_model(mock_old_model, 'lightgbm', X_test)
        assert len(predictions_old) == 3
        
        # Test with new structure (dict with metadata)
        mock_new_model = Mock()
        mock_new_model.predict.return_value = np.array([1.5, 2.5, 3.5])
        
        model_dict = {
            'model': mock_new_model,
            'feature_names': ['feature_0', 'feature_1'],
            'training_time': 10.5
        }
        
        predictions_new = trainer._predict_model(model_dict, 'lightgbm', X_test)
        assert len(predictions_new) == 3
    
    def test_lightgbm_error_handling(self, model_config, mock_mlflow_manager):
        """Test error handling in LightGBM training."""
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        # Create sample data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)
        
        # Test import error by mocking the import to fail
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'lightgbm':
                raise ImportError("LightGBM not installed")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="LightGBM not installed"):
                trainer._train_lightgbm(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])