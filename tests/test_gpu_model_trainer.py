"""
Tests for GPU-Accelerated Model Training Infrastructure

This module contains comprehensive tests for the GPU model training infrastructure,
including configuration validation, GPU monitoring, and training functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.gpu_model_trainer import (
    GPUMetrics, TrainingProgress, XGBoostConfig, LightGBMConfig, 
    PyTorchConfig, CuMLConfig, ModelConfig, GPUMonitor, 
    BaseModelTrainer, GPUModelTrainer
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig


class TestModelConfigurations:
    """Test model configuration classes."""
    
    def test_xgboost_config_defaults(self):
        """Test XGBoost configuration with default values."""
        config = XGBoostConfig()
        
        assert config.tree_method == 'gpu_hist'
        assert config.gpu_id == 0
        assert config.max_depth == 12
        assert config.n_estimators == 5000
        assert config.learning_rate == 0.01
        assert config.random_state == 42
    
    def test_xgboost_config_validation(self):
        """Test XGBoost configuration validation."""
        # Valid configuration
        config = XGBoostConfig(tree_method='hist', max_depth=10)
        assert config.tree_method == 'hist'
        assert config.max_depth == 10
        
        # Invalid tree method
        with pytest.raises(ValueError, match="tree_method must be one of"):
            XGBoostConfig(tree_method='invalid_method')
    
    def test_lightgbm_config_defaults(self):
        """Test LightGBM configuration with default values."""
        config = LightGBMConfig()
        
        assert config.device_type == 'gpu'
        assert config.gpu_platform_id == 0
        assert config.gpu_device_id == 0
        assert config.objective == 'regression'
        assert config.metric == 'rmse'
        assert config.num_leaves == 255
    
    def test_lightgbm_config_validation(self):
        """Test LightGBM configuration validation."""
        # Valid configuration
        config = LightGBMConfig(device_type='cpu', num_leaves=100)
        assert config.device_type == 'cpu'
        assert config.num_leaves == 100
        
        # Invalid device type
        with pytest.raises(ValueError, match="device_type must be one of"):
            LightGBMConfig(device_type='invalid_device')
    
    def test_pytorch_config_defaults(self):
        """Test PyTorch configuration with default values."""
        config = PyTorchConfig()
        
        assert config.hidden_layers == [512, 256, 128, 64]
        assert config.activation == 'relu'
        assert config.dropout_rate == 0.2
        assert config.batch_size == 2048
        assert config.epochs == 500
        assert config.learning_rate == 0.001
        assert config.mixed_precision == True
    
    def test_pytorch_config_validation(self):
        """Test PyTorch configuration validation."""
        # Valid configuration
        config = PyTorchConfig(activation='gelu', lr_scheduler='step')
        assert config.activation == 'gelu'
        assert config.lr_scheduler == 'step'
        
        # Invalid activation
        with pytest.raises(ValueError, match="activation must be one of"):
            PyTorchConfig(activation='invalid_activation')
        
        # Invalid scheduler
        with pytest.raises(ValueError, match="lr_scheduler must be one of"):
            PyTorchConfig(lr_scheduler='invalid_scheduler')
    
    def test_cuml_config_defaults(self):
        """Test cuML configuration with default values."""
        config = CuMLConfig()
        
        assert isinstance(config.linear_regression, dict)
        assert isinstance(config.random_forest, dict)
        assert config.linear_regression['fit_intercept'] == True
        assert config.random_forest['n_estimators'] == 1000
    
    def test_model_config_composition(self):
        """Test complete model configuration composition."""
        config = ModelConfig()
        
        assert isinstance(config.xgboost, XGBoostConfig)
        assert isinstance(config.lightgbm, LightGBMConfig)
        assert isinstance(config.pytorch, PyTorchConfig)
        assert isinstance(config.cuml, CuMLConfig)


class TestGPUMetrics:
    """Test GPU metrics data class."""
    
    def test_gpu_metrics_creation(self):
        """Test GPU metrics creation and conversion."""
        timestamp = datetime.now()
        metrics = GPUMetrics(
            utilization_percent=85.5,
            memory_used_mb=8192.0,
            memory_total_mb=12288.0,
            memory_free_mb=4096.0,
            temperature_celsius=72.0,
            power_usage_watts=250.0,
            timestamp=timestamp
        )
        
        assert metrics.utilization_percent == 85.5
        assert metrics.memory_used_mb == 8192.0
        assert metrics.temperature_celsius == 72.0
        
        # Test dictionary conversion
        metrics_dict = metrics.to_dict()
        assert metrics_dict['gpu_utilization'] == 85.5
        assert metrics_dict['gpu_memory_used_mb'] == 8192.0
        assert metrics_dict['gpu_temperature_c'] == 72.0
        assert 'timestamp' in metrics_dict


class TestTrainingProgress:
    """Test training progress tracking."""
    
    def test_training_progress_creation(self):
        """Test training progress creation and conversion."""
        gpu_metrics = GPUMetrics(
            utilization_percent=80.0,
            memory_used_mb=6144.0,
            memory_total_mb=12288.0,
            memory_free_mb=6144.0,
            temperature_celsius=70.0,
            power_usage_watts=200.0,
            timestamp=datetime.now()
        )
        
        progress = TrainingProgress(
            epoch=10,
            total_epochs=100,
            train_loss=0.25,
            val_loss=0.30,
            train_metrics={'rmse': 0.5, 'mae': 0.4},
            val_metrics={'rmse': 0.55, 'mae': 0.45},
            gpu_metrics=gpu_metrics,
            elapsed_time=120.5,
            eta_seconds=1080.0
        )
        
        assert progress.epoch == 10
        assert progress.total_epochs == 100
        assert progress.train_loss == 0.25
        assert progress.val_loss == 0.30
        
        # Test dictionary conversion
        progress_dict = progress.to_dict()
        assert progress_dict['epoch'] == 10
        assert progress_dict['train_loss'] == 0.25
        assert 'gpu_metrics' in progress_dict
        assert progress_dict['gpu_metrics']['gpu_utilization'] == 80.0


class TestGPUMonitor:
    """Test GPU monitoring functionality."""
    
    @patch('src.gpu_model_trainer.NVML_AVAILABLE', False)
    def test_gpu_monitor_unavailable(self):
        """Test GPU monitor when NVML is not available."""
        monitor = GPUMonitor()
        
        assert monitor.available == False
        assert monitor.get_metrics() is None
        
        device_info = monitor.get_device_info()
        assert device_info['available'] == False
    
    @patch('src.gpu_model_trainer.NVML_AVAILABLE', True)
    @patch('src.gpu_model_trainer.nvml')
    def test_gpu_monitor_available(self, mock_nvml):
        """Test GPU monitor when NVML is available."""
        # Mock NVML functions
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        
        # Mock utilization
        mock_util = Mock()
        mock_util.gpu = 85
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        
        # Mock memory info
        mock_mem = Mock()
        mock_mem.used = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_mem.total = 12 * 1024 * 1024 * 1024  # 12GB in bytes
        mock_mem.free = 4 * 1024 * 1024 * 1024   # 4GB in bytes
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        
        # Mock temperature and power
        mock_nvml.nvmlDeviceGetTemperature.return_value = 72
        mock_nvml.nvmlDeviceGetPowerUsage.return_value = 250000  # 250W in milliwatts
        
        # Mock device info
        mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"
        mock_nvml.nvmlSystemGetDriverVersion.return_value = b"535.86.10"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020  # CUDA 12.2
        mock_nvml.NVML_TEMPERATURE_GPU = 0
        
        monitor = GPUMonitor()
        
        assert monitor.available == True
        
        # Test metrics
        metrics = monitor.get_metrics()
        assert metrics is not None
        assert metrics.utilization_percent == 85
        assert metrics.memory_used_mb == 8192.0  # 8GB in MB
        assert metrics.temperature_celsius == 72
        assert metrics.power_usage_watts == 250.0
        
        # Test device info
        device_info = monitor.get_device_info()
        assert device_info['available'] == True
        assert device_info['name'] == "NVIDIA RTX 4090"
        assert device_info['driver_version'] == "535.86.10"
        assert device_info['cuda_version'] == "12.2"
    
    @patch('src.gpu_model_trainer.NVML_AVAILABLE', True)
    @patch('src.gpu_model_trainer.nvml')
    def test_gpu_monitor_error_handling(self, mock_nvml):
        """Test GPU monitor error handling."""
        # Mock initialization failure
        mock_nvml.nvmlInit.side_effect = Exception("NVML init failed")
        
        monitor = GPUMonitor()
        assert monitor.available == False
        
        # Test metrics error handling
        mock_nvml.nvmlInit.side_effect = None
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_nvml.nvmlDeviceGetUtilizationRates.side_effect = Exception("GPU error")
        
        monitor = GPUMonitor()
        monitor.available = True
        monitor.handle = "mock_handle"
        
        metrics = monitor.get_metrics()
        assert metrics is None


class TestGPUModelTrainer:
    """Test main GPU model trainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(1000, 8)
        y = np.random.randn(1000)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val
    
    @pytest.fixture
    def mock_mlflow_manager(self):
        """Create mock MLflow manager."""
        mock_manager = Mock()
        mock_manager.start_run.return_value = "test_run_id"
        mock_manager.log_parameters.return_value = None
        mock_manager.log_metrics.return_value = None
        mock_manager.log_model.return_value = None
        mock_manager.log_artifacts.return_value = None
        mock_manager.end_run.return_value = None
        
        # Mock client and active run
        mock_client = Mock()
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_client.active_run.return_value = mock_run
        mock_client.log_metric.return_value = None
        mock_manager.client = mock_client
        
        return mock_manager
    
    def test_gpu_model_trainer_initialization(self, mock_mlflow_manager):
        """Test GPU model trainer initialization."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        assert trainer.config == config
        assert trainer.mlflow_manager == mock_mlflow_manager
        assert isinstance(trainer.gpu_monitor, GPUMonitor)
        assert trainer.current_training is None
    
    def test_device_info_retrieval(self, mock_mlflow_manager):
        """Test device information retrieval."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        device_info = trainer.get_device_info()
        assert isinstance(device_info, dict)
        assert 'available' in device_info
    
    def test_gpu_availability_check(self, mock_mlflow_manager):
        """Test GPU availability check."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # This will depend on the actual system, but should not raise an error
        is_available = trainer.is_gpu_available()
        assert isinstance(is_available, bool)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_fallback(self, mock_cuda, mock_mlflow_manager):
        """Test CPU fallback when CUDA is not available."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        assert not trainer.is_gpu_available()
    
    def test_metrics_calculation(self, mock_mlflow_manager):
        """Test metrics calculation functionality."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # Create sample predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2_score'] <= 1
    
    def test_pytorch_model_creation(self, mock_mlflow_manager):
        """Test PyTorch model creation."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        input_size = 8
        pytorch_config = config.pytorch
        
        model = trainer._create_pytorch_model(input_size, pytorch_config)
        
        assert isinstance(model, torch.nn.Module)
        
        # Test model forward pass
        test_input = torch.randn(1, input_size)
        output = model(test_input)
        assert output.shape == (1, 1)
    
    def test_activation_functions(self, mock_mlflow_manager):
        """Test activation function selection."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # Test different activation functions
        activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish']
        
        for activation in activations:
            activation_layer = trainer._get_activation(activation)
            assert isinstance(activation_layer, torch.nn.Module)
        
        # Test default activation
        default_activation = trainer._get_activation('unknown')
        assert isinstance(default_activation, torch.nn.ReLU)
    
    @patch('src.gpu_model_trainer.Path.mkdir')
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    def test_plots_directory_creation(self, mock_close, mock_savefig, mock_mkdir, mock_mlflow_manager):
        """Test plots directory creation."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        plots_dir = trainer._create_plots_directory()
        
        assert isinstance(plots_dir, Path)
        assert plots_dir.name == "plots"
    
    def test_training_state_management(self, mock_mlflow_manager):
        """Test training state management."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # Initial state
        assert trainer.current_training is None
        assert not trainer.stop_training.is_set()
        assert not trainer.pause_training.is_set()
        
        # Test direct event manipulation (since async methods require active thread)
        trainer.stop_training.set()
        assert trainer.stop_training.is_set()
        
        trainer.pause_training.set()
        assert trainer.pause_training.is_set()
        
        trainer.pause_training.clear()
        assert not trainer.pause_training.is_set()
    
    def test_progress_queue_handling(self, mock_mlflow_manager):
        """Test training progress queue handling."""
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # Initially empty
        progress = trainer.get_training_progress()
        assert progress is None
        
        # Add progress to queue
        test_progress = TrainingProgress(
            epoch=1,
            total_epochs=10,
            train_loss=0.5
        )
        
        trainer.progress_queue.put(test_progress)
        
        # Retrieve progress
        retrieved_progress = trainer.get_training_progress()
        assert retrieved_progress is not None
        assert retrieved_progress.epoch == 1
        assert retrieved_progress.train_loss == 0.5


class TestIntegration:
    """Integration tests for the complete training infrastructure."""
    
    @pytest.fixture
    def temp_mlflow_config(self):
        """Create temporary MLflow configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MLflowConfig(
                tracking_uri=f"file://{temp_dir}/mlruns",
                experiment_name="test_experiment"
            )
            yield config
    
    def test_end_to_end_configuration(self, temp_mlflow_config):
        """Test end-to-end configuration setup."""
        # Create model config
        model_config = ModelConfig()
        
        # Verify all configurations are properly initialized
        assert isinstance(model_config.xgboost, XGBoostConfig)
        assert isinstance(model_config.lightgbm, LightGBMConfig)
        assert isinstance(model_config.pytorch, PyTorchConfig)
        assert isinstance(model_config.cuml, CuMLConfig)
        
        # Test configuration serialization
        config_dict = model_config.model_dump()
        assert 'xgboost' in config_dict
        assert 'lightgbm' in config_dict
        assert 'pytorch' in config_dict
        assert 'cuml' in config_dict
    
    @patch('src.gpu_model_trainer.NVML_AVAILABLE', False)
    def test_training_without_gpu_monitoring(self, temp_mlflow_config):
        """Test training infrastructure without GPU monitoring."""
        try:
            mlflow_manager = MLflowExperimentManager(temp_mlflow_config)
            model_config = ModelConfig()
            trainer = GPUModelTrainer(model_config, mlflow_manager)
            
            # Should initialize successfully even without GPU monitoring
            assert trainer.gpu_monitor.available == False
            assert trainer.get_gpu_metrics() is None
            
        except Exception as e:
            # If MLflow setup fails, that's expected in test environment
            assert "MLflow" in str(e) or "tracking" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])