"""
GPU-Accelerated Model Training Infrastructure

This module provides comprehensive GPU-accelerated model training capabilities
with CUDA device detection, configuration management, GPU metrics collection,
and training progress tracking for the MLOps platform.
"""

import os
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# GPU monitoring
try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    nvml = None

# MLflow integration
from .mlflow_config import MLflowExperimentManager, ExperimentMetrics, ModelArtifacts

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Data class for GPU metrics."""
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    temperature_celsius: float
    power_usage_watts: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'gpu_utilization': self.utilization_percent,
            'gpu_memory_used_mb': self.memory_used_mb,
            'gpu_memory_total_mb': self.memory_total_mb,
            'gpu_memory_free_mb': self.memory_free_mb,
            'gpu_temperature_c': self.temperature_celsius,
            'gpu_power_watts': self.power_usage_watts,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TrainingProgress:
    """Data class for training progress tracking."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    gpu_metrics: Optional[GPUMetrics] = None
    elapsed_time: float = 0.0
    eta_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'elapsed_time': self.elapsed_time,
            'eta_seconds': self.eta_seconds
        }
        
        if self.train_metrics:
            result['train_metrics'] = self.train_metrics
        if self.val_metrics:
            result['val_metrics'] = self.val_metrics
        if self.gpu_metrics:
            result['gpu_metrics'] = self.gpu_metrics.to_dict()
            
        return result


class XGBoostConfig(BaseModel):
    """Configuration for XGBoost GPU training."""
    tree_method: str = Field(default='gpu_hist', description="Tree construction algorithm")
    gpu_id: int = Field(default=0, description="GPU device ID")
    max_depth: int = Field(default=12, ge=1, le=20, description="Maximum tree depth")
    n_estimators: int = Field(default=5000, ge=100, le=10000, description="Number of boosting rounds")
    learning_rate: float = Field(default=0.01, ge=0.001, le=1.0, description="Learning rate")
    subsample: float = Field(default=0.8, ge=0.1, le=1.0, description="Subsample ratio")
    colsample_bytree: float = Field(default=0.8, ge=0.1, le=1.0, description="Feature sampling ratio")
    reg_alpha: float = Field(default=0.1, ge=0.0, le=10.0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, ge=0.0, le=10.0, description="L2 regularization")
    random_state: int = Field(default=42, description="Random seed")
    early_stopping_rounds: int = Field(default=100, ge=10, le=500, description="Early stopping patience")
    
    @field_validator('tree_method')
    @classmethod
    def validate_tree_method(cls, v):
        valid_methods = ['gpu_hist', 'hist', 'exact', 'approx']
        if v not in valid_methods:
            raise ValueError(f"tree_method must be one of {valid_methods}")
        return v


class LightGBMConfig(BaseModel):
    """Configuration for LightGBM GPU training."""
    device_type: str = Field(default='gpu', description="Device type for training")
    gpu_platform_id: int = Field(default=0, description="OpenCL platform ID")
    gpu_device_id: int = Field(default=0, description="OpenCL device ID")
    objective: str = Field(default='regression', description="Learning objective")
    metric: str = Field(default='rmse', description="Evaluation metric")
    boosting_type: str = Field(default='gbdt', description="Boosting type")
    num_leaves: int = Field(default=255, ge=10, le=1000, description="Maximum number of leaves")
    max_depth: int = Field(default=12, ge=1, le=20, description="Maximum tree depth")
    n_estimators: int = Field(default=5000, ge=100, le=10000, description="Number of boosting rounds")
    learning_rate: float = Field(default=0.01, ge=0.001, le=1.0, description="Learning rate")
    feature_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="Feature sampling ratio")
    bagging_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="Data sampling ratio")
    bagging_freq: int = Field(default=5, ge=1, le=10, description="Bagging frequency")
    reg_alpha: float = Field(default=0.1, ge=0.0, le=10.0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, ge=0.0, le=10.0, description="L2 regularization")
    random_state: int = Field(default=42, description="Random seed")
    early_stopping_rounds: int = Field(default=100, ge=10, le=500, description="Early stopping patience")
    
    @field_validator('device_type')
    @classmethod
    def validate_device_type(cls, v):
        valid_devices = ['gpu', 'cpu']
        if v not in valid_devices:
            raise ValueError(f"device_type must be one of {valid_devices}")
        return v


class PyTorchConfig(BaseModel):
    """Configuration for PyTorch neural network training."""
    hidden_layers: List[int] = Field(default=[512, 256, 128, 64], description="Hidden layer sizes")
    activation: str = Field(default='relu', description="Activation function")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.8, description="Dropout rate")
    batch_size: int = Field(default=2048, ge=32, le=8192, description="Training batch size")
    epochs: int = Field(default=500, ge=10, le=2000, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, ge=1e-5, le=0.1, description="Learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1e-2, description="Weight decay (L2 regularization)")
    device: str = Field(default='cuda', description="Training device")
    mixed_precision: bool = Field(default=True, description="Use mixed precision training")
    early_stopping_patience: int = Field(default=50, ge=5, le=200, description="Early stopping patience")
    lr_scheduler: str = Field(default='cosine', description="Learning rate scheduler")
    warmup_epochs: int = Field(default=10, ge=0, le=50, description="Warmup epochs")
    
    @field_validator('activation')
    @classmethod
    def validate_activation(cls, v):
        valid_activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish']
        if v not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
        return v
    
    @field_validator('lr_scheduler')
    @classmethod
    def validate_lr_scheduler(cls, v):
        valid_schedulers = ['cosine', 'step', 'exponential', 'plateau']
        if v not in valid_schedulers:
            raise ValueError(f"lr_scheduler must be one of {valid_schedulers}")
        return v


class CuMLConfig(BaseModel):
    """Configuration for cuML GPU-accelerated models."""
    linear_regression: Dict[str, Any] = Field(
        default_factory=lambda: {
            'fit_intercept': True,
            'normalize': False,
            'algorithm': 'eig'
        },
        description="Linear regression parameters"
    )
    random_forest: Dict[str, Any] = Field(
        default_factory=lambda: {
            'n_estimators': 1000,
            'max_depth': 16,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'random_state': 42,
            'n_streams': 4
        },
        description="Random forest parameters"
    )


class ModelConfig(BaseModel):
    """Comprehensive model configuration for all supported algorithms."""
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    pytorch: PyTorchConfig = Field(default_factory=PyTorchConfig)
    cuml: CuMLConfig = Field(default_factory=CuMLConfig)


class GPUMonitor:
    """GPU monitoring utility using nvidia-ml-py."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU monitor.
        
        Args:
            device_id: GPU device ID to monitor
        """
        self.device_id = device_id
        self.available = False
        self.handle = None
        
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                self.available = True
                logger.info(f"GPU monitoring initialized for device {device_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
        else:
            logger.warning("nvidia-ml-py not available, GPU monitoring disabled")
    
    def get_metrics(self) -> Optional[GPUMetrics]:
        """
        Get current GPU metrics.
        
        Returns:
            GPUMetrics object or None if monitoring unavailable
        """
        if not self.available or not self.handle:
            return None
        
        try:
            # Get utilization
            util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # Get memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            # Get temperature
            temp = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
            
            # Get power usage
            power = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
            
            return GPUMetrics(
                utilization_percent=util.gpu,
                memory_used_mb=mem_info.used / (1024 * 1024),
                memory_total_mb=mem_info.total / (1024 * 1024),
                memory_free_mb=mem_info.free / (1024 * 1024),
                temperature_celsius=temp,
                power_usage_watts=power,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return None
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get GPU device information.
        
        Returns:
            Dictionary with device information
        """
        if not self.available or not self.handle:
            return {"available": False}
        
        try:
            name = nvml.nvmlDeviceGetName(self.handle).decode('utf-8')
            driver_version = nvml.nvmlSystemGetDriverVersion().decode('utf-8')
            cuda_version = nvml.nvmlSystemGetCudaDriverVersion()
            
            return {
                "available": True,
                "name": name,
                "driver_version": driver_version,
                "cuda_version": f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}",
                "device_id": self.device_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {"available": False, "error": str(e)}


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers with common GPU optimization patterns."""
    
    def __init__(self, config: ModelConfig, mlflow_manager: MLflowExperimentManager):
        """
        Initialize base trainer.
        
        Args:
            config: Model configuration
            mlflow_manager: MLflow experiment manager
        """
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.gpu_monitor = GPUMonitor()
        self.device_info = self.gpu_monitor.get_device_info()
        self.training_history = []
        self.progress_callbacks = []
        
        # Setup device
        self.device = self._setup_device()
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup and validate CUDA device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """Add a callback function to receive training progress updates."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, progress: TrainingProgress):
        """Notify all registered callbacks about training progress."""
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def _log_gpu_metrics(self, step: Optional[int] = None):
        """Log current GPU metrics to MLflow."""
        metrics = self.gpu_monitor.get_metrics()
        if metrics:
            gpu_metrics_dict = {
                'gpu_utilization': metrics.utilization_percent,
                'gpu_memory_used_mb': metrics.memory_used_mb,
                'gpu_temperature_c': metrics.temperature_celsius,
                'gpu_power_watts': metrics.power_usage_watts
            }
            
            for key, value in gpu_metrics_dict.items():
                try:
                    self.mlflow_manager.client.log_metric(
                        run_id=self.mlflow_manager.client.active_run().info.run_id,
                        key=key,
                        value=value,
                        step=step
                    )
                except Exception as e:
                    logger.debug(f"Failed to log GPU metric {key}: {e}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def _create_plots_directory(self) -> Path:
        """Create directory for saving plots."""
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        return plots_dir
    
    def _save_training_curves(self, model_name: str) -> str:
        """Save training curves plot."""
        if not self.training_history:
            return None
        
        plots_dir = self._create_plots_directory()
        plot_path = plots_dir / f"{model_name}_training_curves.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training Progress', fontsize=16)
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h.get('val_loss') for h in self.training_history if h.get('val_loss')]
        
        # Training loss
        axes[0, 0].plot(epochs, train_losses, label='Training Loss')
        if val_losses:
            axes[0, 0].plot(epochs[:len(val_losses)], val_losses, label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # GPU utilization
        gpu_utils = [h.get('gpu_metrics', {}).get('gpu_utilization', 0) for h in self.training_history]
        if any(gpu_utils):
            axes[0, 1].plot(epochs, gpu_utils, color='green')
            axes[0, 1].set_title('GPU Utilization')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Utilization (%)')
            axes[0, 1].grid(True)
        
        # GPU memory usage
        gpu_memory = [h.get('gpu_metrics', {}).get('gpu_memory_used_mb', 0) for h in self.training_history]
        if any(gpu_memory):
            axes[1, 0].plot(epochs, gpu_memory, color='red')
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid(True)
        
        # GPU temperature
        gpu_temps = [h.get('gpu_metrics', {}).get('gpu_temperature_c', 0) for h in self.training_history]
        if any(gpu_temps):
            axes[1, 1].plot(epochs, gpu_temps, color='orange')
            axes[1, 1].set_title('GPU Temperature')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions
        """
        pass


class GPUModelTrainer:
    """
    Main GPU-accelerated model trainer with comprehensive training infrastructure.
    
    This class orchestrates training of multiple GPU-accelerated models with
    comprehensive experiment tracking, GPU monitoring, and progress reporting.
    """
    
    def __init__(self, config: ModelConfig, mlflow_manager: MLflowExperimentManager):
        """
        Initialize GPU model trainer.
        
        Args:
            config: Model configuration
            mlflow_manager: MLflow experiment manager
        """
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.gpu_monitor = GPUMonitor()
        self.device_info = self.gpu_monitor.get_device_info()
        
        # Training state
        self.current_training = None
        self.training_thread = None
        self.stop_training = threading.Event()
        self.pause_training = threading.Event()
        self.progress_queue = queue.Queue()
        
        logger.info("GPUModelTrainer initialized")
        logger.info(f"GPU Info: {self.device_info}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        return self.device_info
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for training."""
        return torch.cuda.is_available() and self.device_info.get('available', False)
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        return self.gpu_monitor.get_metrics()
    
    def start_training_async(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                           models_to_train: Optional[List[str]] = None) -> str:
        """
        Start asynchronous training of multiple models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            models_to_train: List of model names to train
            
        Returns:
            Training session ID
        """
        if self.training_thread and self.training_thread.is_alive():
            raise RuntimeError("Training already in progress")
        
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.stop_training.clear()
        self.pause_training.clear()
        
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(session_id, X_train, y_train, X_val, y_val, models_to_train),
            daemon=True
        )
        self.training_thread.start()
        
        logger.info(f"Started async training session: {session_id}")
        return session_id
    
    def stop_training_async(self):
        """Stop the current training session."""
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training.set()
            logger.info("Training stop requested")
    
    def pause_training_async(self):
        """Pause the current training session."""
        if self.training_thread and self.training_thread.is_alive():
            self.pause_training.set()
            logger.info("Training pause requested")
    
    def resume_training_async(self):
        """Resume the paused training session."""
        if self.training_thread and self.training_thread.is_alive():
            self.pause_training.clear()
            logger.info("Training resumed")
    
    def get_training_progress(self) -> Optional[TrainingProgress]:
        """Get the latest training progress."""
        try:
            return self.progress_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _training_worker(self, session_id: str, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                        models_to_train: Optional[List[str]]):
        """Worker thread for training models."""
        try:
            self.current_training = session_id
            
            if models_to_train is None:
                models_to_train = ['xgboost', 'lightgbm', 'pytorch']
            
            total_models = len(models_to_train)
            
            for i, model_name in enumerate(models_to_train):
                if self.stop_training.is_set():
                    logger.info("Training stopped by user")
                    break
                
                # Wait if paused
                while self.pause_training.is_set() and not self.stop_training.is_set():
                    time.sleep(1)
                
                if self.stop_training.is_set():
                    break
                
                logger.info(f"Training model {i+1}/{total_models}: {model_name}")
                
                try:
                    self._train_single_model(model_name, X_train, y_train, X_val, y_val)
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    continue
            
            logger.info(f"Training session {session_id} completed")
            
        except Exception as e:
            logger.error(f"Training worker failed: {e}")
        finally:
            self.current_training = None
    
    def _train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """Train a single model with comprehensive tracking."""
        run_name = f"{model_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start MLflow run
        run_id = self.mlflow_manager.start_run(
            run_name=run_name,
            tags={
                'model_type': model_name,
                'gpu_enabled': str(self.is_gpu_available()),
                'device_info': json.dumps(self.device_info)
            }
        )
        
        try:
            start_time = time.time()
            
            # Log model configuration
            if model_name == 'xgboost':
                config_dict = self.config.xgboost.model_dump()
            elif model_name == 'lightgbm':
                config_dict = self.config.lightgbm.model_dump()
            elif model_name == 'pytorch':
                config_dict = self.config.pytorch.model_dump()
            else:
                config_dict = {}
            
            self.mlflow_manager.log_parameters(config_dict)
            
            # Train the model
            if model_name == 'xgboost':
                model = self._train_xgboost(X_train, y_train, X_val, y_val)
            elif model_name == 'lightgbm':
                model = self._train_lightgbm(X_train, y_train, X_val, y_val)
            elif model_name == 'pytorch':
                model = self._train_pytorch(X_train, y_train, X_val, y_val)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Calculate final metrics
            train_pred = self._predict_model(model, model_name, X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred)
            
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_pred = self._predict_model(model, model_name, X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)
            
            # Get final GPU metrics
            gpu_metrics = self.gpu_monitor.get_metrics()
            
            # Create experiment metrics
            experiment_metrics = ExperimentMetrics(
                rmse=val_metrics.get('rmse', train_metrics['rmse']),
                mae=val_metrics.get('mae', train_metrics['mae']),
                r2_score=val_metrics.get('r2_score', train_metrics['r2_score']),
                training_time=time.time() - start_time,
                gpu_utilization=gpu_metrics.utilization_percent if gpu_metrics else None,
                gpu_memory_used=gpu_metrics.memory_used_mb if gpu_metrics else None
            )
            
            # Log metrics to MLflow
            self.mlflow_manager.log_metrics(experiment_metrics)
            
            # Log model
            self.mlflow_manager.log_model(model, model_name)
            
            # Create and log artifacts
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            artifacts = ModelArtifacts(
                model_path=f"models/{model_name}_model.pkl",
                training_curves_plot=str(plots_dir / f"{model_name}_training_curves.png")
            )
            
            self.mlflow_manager.log_artifacts(artifacts)
            
            # End run successfully
            self.mlflow_manager.end_run("FINISHED")
            
            logger.info(f"Successfully trained {model_name} model")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            self.mlflow_manager.end_run("FAILED")
            raise
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """
        Train XGBoost model with GPU acceleration, advanced hyperparameters,
        feature importance extraction, cross-validation, and comprehensive MLflow logging.
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        config = self.config.xgboost
        start_time = time.time()
        
        logger.info("Starting XGBoost GPU training with advanced configuration...")
        
        # Advanced XGBoost parameters optimized for deep trees and high estimator counts
        params = {
            # GPU acceleration (modern XGBoost 3.x API)
            'tree_method': config.tree_method,
            'device': 'cuda' if config.tree_method == 'gpu_hist' else 'cpu',
            # Note: gpu_id is deprecated in XGBoost 3.x, use device instead
            
            # Advanced tree parameters for deep learning
            'max_depth': config.max_depth,
            'max_leaves': 2**config.max_depth,  # Exponential leaf growth for deep trees
            'grow_policy': 'lossguide',  # Loss-guided growth for better performance
            
            # Learning parameters
            'learning_rate': config.learning_rate,
            'n_estimators': config.n_estimators,
            
            # Advanced sampling parameters
            'subsample': config.subsample,
            'colsample_bytree': config.colsample_bytree,
            'colsample_bylevel': 0.8,  # Column sampling by tree level
            'colsample_bynode': 0.8,   # Column sampling by node
            
            # Regularization for high complexity models
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'gamma': 0.1,  # Minimum loss reduction for split
            'min_child_weight': 3,  # Minimum sum of instance weight in child
            
            # Advanced optimization
            'max_delta_step': 1,  # Maximum delta step for weight estimation
            'scale_pos_weight': 1,  # Balance of positive and negative weights
            
            # Objective and evaluation
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            
            # Reproducibility and performance
            'random_state': config.random_state,
            'n_jobs': -1,  # Use all available cores
            'verbosity': 1  # Moderate verbosity for monitoring
        }
        
        # Log advanced hyperparameters to MLflow
        advanced_params = {
            'xgb_max_leaves': params['max_leaves'],
            'xgb_grow_policy': params['grow_policy'],
            'xgb_colsample_bylevel': params['colsample_bylevel'],
            'xgb_colsample_bynode': params['colsample_bynode'],
            'xgb_gamma': params['gamma'],
            'xgb_min_child_weight': params['min_child_weight'],
            'xgb_max_delta_step': params['max_delta_step']
        }
        self.mlflow_manager.log_parameters(advanced_params)
        
        # Create DMatrix with feature names for better interpretability
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        
        # Setup evaluation sets
        eval_set = [(dtrain, 'train')]
        dval = None
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            eval_set.append((dval, 'val'))
        
        # Perform cross-validation before training
        cv_rounds = min(config.n_estimators, 1000)  # Limit CV rounds for efficiency
        logger.info(f"Performing {5}-fold cross-validation with {cv_rounds} rounds...")
        print(f"[CV] Starting cross-validation: {5} folds, {cv_rounds} rounds")
        cv_start_time = time.time()
        
        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=cv_rounds,
            nfold=5,
            stratified=False,  # Regression task
            shuffle=True,
            seed=config.random_state,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=50  # Show progress every 50 rounds
        )
        
        cv_time = time.time() - cv_start_time
        
        # Extract CV results
        cv_train_rmse_mean = cv_results['train-rmse-mean'].iloc[-1]
        cv_train_rmse_std = cv_results['train-rmse-std'].iloc[-1]
        cv_test_rmse_mean = cv_results['test-rmse-mean'].iloc[-1]
        cv_test_rmse_std = cv_results['test-rmse-std'].iloc[-1]
        cv_best_iteration = len(cv_results)
        
        # Log cross-validation results
        cv_metrics = {
            'cv_train_rmse_mean': cv_train_rmse_mean,
            'cv_train_rmse_std': cv_train_rmse_std,
            'cv_test_rmse_mean': cv_test_rmse_mean,
            'cv_test_rmse_std': cv_test_rmse_std,
            'cv_best_iteration': cv_best_iteration,
            'cv_time_seconds': cv_time
        }
        
        # Log metrics individually to MLflow
        for key, value in cv_metrics.items():
            try:
                import mlflow
                mlflow.log_metric(key, value)
            except Exception as e:
                logger.debug(f"Failed to log CV metric {key}: {e}")
        
        logger.info(f"Cross-validation completed in {cv_time:.2f}s")
        logger.info(f"CV RMSE: {cv_test_rmse_mean:.4f} ± {cv_test_rmse_std:.4f}")
        print(f"[CV] Cross-validation completed: RMSE={cv_test_rmse_mean:.4f} ± {cv_test_rmse_std:.4f}")
        
        # Train final model with optimal number of estimators from CV
        optimal_estimators = min(cv_best_iteration + config.early_stopping_rounds, config.n_estimators)
        logger.info(f"Training final model with {optimal_estimators} estimators...")
        print(f"[TRAIN] Starting final training with {optimal_estimators} estimators...")
        
        # Setup training history tracking
        training_history = []
        
        # Train the model with early stopping
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=optimal_estimators,
            evals=eval_set,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=100  # Print evaluation every 100 rounds
        )
        
        # Collect training history after training (simplified for compatibility)
        for i in range(0, model.num_boosted_rounds(), 100):
            gpu_metrics = self.gpu_monitor.get_metrics()
            progress_entry = {
                'iteration': i,
                'elapsed_time': time.time() - start_time
            }
            
            if gpu_metrics:
                progress_entry.update({
                    'gpu_utilization': gpu_metrics.utilization_percent,
                    'gpu_memory_used_mb': gpu_metrics.memory_used_mb,
                    'gpu_temperature_c': gpu_metrics.temperature_celsius,
                    'gpu_power_watts': gpu_metrics.power_usage_watts
                })
            
            training_history.append(progress_entry)
        
        training_time = time.time() - start_time
        logger.info(f"XGBoost training completed in {training_time:.2f}s")
        
        # Extract and log feature importance
        logger.info("Extracting feature importance...")
        feature_importance = model.get_score(importance_type='gain')  # Use gain for feature importance
        feature_importance_weight = model.get_score(importance_type='weight')  # Frequency of feature usage
        feature_importance_cover = model.get_score(importance_type='cover')  # Coverage of feature
        
        # Create feature importance visualization
        plots_dir = self._create_plots_directory()
        
        # Plot feature importance (gain)
        if feature_importance:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v, 'type': 'gain'} 
                for k, v in feature_importance.items()
            ])
            importance_df = importance_df.sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance (Gain)')
            plt.title('XGBoost Feature Importance - Top 20 Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            importance_plot_path = plots_dir / 'XGBoost_feature_importance.png'
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log feature importance metrics
            top_features = importance_df.head(10)
            for idx, row in top_features.iterrows():
                try:
                    import mlflow
                    mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
                except Exception as e:
                    logger.debug(f"Failed to log feature importance {row['feature']}: {e}")
        
        # Create comprehensive training curves plot
        if training_history:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('XGBoost Training Progress', fontsize=16)
            
            iterations = [h['iteration'] for h in training_history]
            
            # Training progress (simplified - show iteration progress)
            axes[0, 0].plot(iterations, [i for i in iterations], label='Training Progress', color='blue')
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Completed Iterations')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # GPU utilization
            gpu_utils = [h.get('gpu_utilization', 0) for h in training_history]
            if any(gpu_utils):
                axes[0, 1].plot(iterations, gpu_utils, color='green')
                axes[0, 1].set_title('GPU Utilization')
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Utilization (%)')
                axes[0, 1].grid(True)
            else:
                axes[0, 1].text(0.5, 0.5, 'GPU Metrics\nNot Available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('GPU Utilization')
            
            # GPU memory
            gpu_memory = [h.get('gpu_memory_used_mb', 0) for h in training_history]
            if any(gpu_memory):
                axes[0, 2].plot(iterations, gpu_memory, color='red')
                axes[0, 2].set_title('GPU Memory Usage')
                axes[0, 2].set_xlabel('Iteration')
                axes[0, 2].set_ylabel('Memory (MB)')
                axes[0, 2].grid(True)
            else:
                axes[0, 2].text(0.5, 0.5, 'GPU Memory\nNot Available', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('GPU Memory Usage')
            
            # GPU temperature
            gpu_temps = [h.get('gpu_temperature_c', 0) for h in training_history]
            if any(gpu_temps):
                axes[1, 0].plot(iterations, gpu_temps, color='orange')
                axes[1, 0].set_title('GPU Temperature')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Temperature (°C)')
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'GPU Temperature\nNot Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('GPU Temperature')
            
            # GPU power
            gpu_power = [h.get('gpu_power_watts', 0) for h in training_history]
            if any(gpu_power):
                axes[1, 1].plot(iterations, gpu_power, color='purple')
                axes[1, 1].set_title('GPU Power Usage')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Power (W)')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'GPU Power\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('GPU Power Usage')
            
            # Training time progression
            elapsed_times = [h.get('elapsed_time', 0) for h in training_history]
            if elapsed_times:
                axes[1, 2].plot(iterations, elapsed_times, color='brown')
                axes[1, 2].set_title('Cumulative Training Time')
                axes[1, 2].set_xlabel('Iteration')
                axes[1, 2].set_ylabel('Time (seconds)')
                axes[1, 2].grid(True)
            
            plt.tight_layout()
            training_curves_path = plots_dir / 'XGBoost_training_curves.png'
            plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Log comprehensive training metrics
        final_metrics = {
            'xgb_training_time': training_time,
            'xgb_best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else optimal_estimators,
            'xgb_num_features': X_train.shape[1],
            'xgb_num_trees': model.num_boosted_rounds(),
            'xgb_feature_importance_count': len(feature_importance) if feature_importance else 0
        }
        
        # Add final evaluation metrics
        if eval_set:
            train_pred = model.predict(dtrain)
            train_metrics = self._calculate_metrics(y_train, train_pred)
            final_metrics.update({
                'xgb_final_train_rmse': train_metrics['rmse'],
                'xgb_final_train_mae': train_metrics['mae'],
                'xgb_final_train_r2': train_metrics['r2_score']
            })
            
            if dval is not None:
                val_pred = model.predict(dval)
                val_metrics = self._calculate_metrics(y_val, val_pred)
                final_metrics.update({
                    'xgb_final_val_rmse': val_metrics['rmse'],
                    'xgb_final_val_mae': val_metrics['mae'],
                    'xgb_final_val_r2': val_metrics['r2_score']
                })
        
        # Log final metrics individually to MLflow
        for key, value in final_metrics.items():
            try:
                import mlflow
                mlflow.log_metric(key, value)
            except Exception as e:
                logger.debug(f"Failed to log final metric {key}: {e}")
        
        # Save model artifacts
        model_path = plots_dir / 'xgboost_model.json'
        model.save_model(str(model_path))
        
        # Save feature importance data
        if feature_importance:
            importance_data = {
                'gain': feature_importance,
                'weight': feature_importance_weight,
                'cover': feature_importance_cover
            }
            
            importance_json_path = plots_dir / 'xgboost_feature_importance.json'
            with open(importance_json_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
        
        # Save training history
        if training_history:
            history_path = plots_dir / 'xgboost_training_history.json'
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
        
        logger.info("XGBoost training completed successfully with comprehensive logging")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Feature importance plot saved to: {importance_plot_path}")
        logger.info(f"Training curves saved to: {training_curves_path}")
        
        return model
    
    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """Train LightGBM model with GPU acceleration."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        config = self.config.lightgbm
        
        # Prepare parameters
        params = {
            'device_type': config.device_type,
            'objective': config.objective,
            'metric': config.metric,
            'boosting_type': config.boosting_type,
            'num_leaves': config.num_leaves,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'feature_fraction': config.feature_fraction,
            'bagging_fraction': config.bagging_fraction,
            'bagging_freq': config.bagging_freq,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'random_state': config.random_state,
            'verbose': -1
        }
        
        if config.device_type == 'gpu':
            params.update({
                'gpu_platform_id': config.gpu_platform_id,
                'gpu_device_id': config.gpu_device_id
            })
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(config.early_stopping_rounds)]
        )
        
        return model
    
    def _train_pytorch(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """Train PyTorch neural network with GPU acceleration and mixed precision."""
        config = self.config.pytorch
        
        # Setup device
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Prepare data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Create model
        input_size = X_train.shape[1]
        model = self._create_pytorch_model(input_size, config).to(device)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        if config.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        elif config.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        elif config.lr_scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Setup mixed precision
        scaler_amp = torch.cuda.amp.GradScaler() if config.mixed_precision and device.type == 'cuda' else None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            if self.stop_training.is_set():
                break
            
            # Wait if paused
            while self.pause_training.is_set() and not self.stop_training.is_set():
                time.sleep(1)
            
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if config.mixed_precision and scaler_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        loss = nn.MSELoss()(outputs, batch_y)
                    
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    outputs = model(batch_X)
                    loss = nn.MSELoss()(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = None
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        if config.mixed_precision and scaler_amp:
                            with torch.cuda.amp.autocast():
                                outputs = model(batch_X)
                                loss = nn.MSELoss()(outputs, batch_y)
                        else:
                            outputs = model(batch_X)
                            loss = nn.MSELoss()(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Update scheduler
            if config.lr_scheduler == 'plateau' and val_loss:
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log progress
            gpu_metrics = self.gpu_monitor.get_metrics()
            progress = TrainingProgress(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                gpu_metrics=gpu_metrics,
                elapsed_time=time.time()
            )
            
            try:
                self.progress_queue.put_nowait(progress)
            except queue.Full:
                pass  # Skip if queue is full
        
        # Return model with scaler for preprocessing
        return {'model': model, 'scaler': scaler, 'device': device}
    
    def _create_pytorch_model(self, input_size: int, config: PyTorchConfig) -> nn.Module:
        """Create PyTorch neural network model."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, config.hidden_layers[0]))
        layers.append(self._get_activation(config.activation))
        layers.append(nn.Dropout(config.dropout_rate))
        
        # Hidden layers
        for i in range(len(config.hidden_layers) - 1):
            layers.append(nn.Linear(config.hidden_layers[i], config.hidden_layers[i + 1]))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(config.hidden_layers[-1], 1))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _predict_model(self, model: Any, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if model_name == 'xgboost':
            import xgboost as xgb
            # Create feature names to match training
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            dtest = xgb.DMatrix(X, feature_names=feature_names)
            return model.predict(dtest)
        
        elif model_name == 'lightgbm':
            return model.predict(X)
        
        elif model_name == 'pytorch':
            model_dict = model
            pytorch_model = model_dict['model']
            scaler = model_dict['scaler']
            device = model_dict['device']
            
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            
            pytorch_model.eval()
            with torch.no_grad():
                predictions = pytorch_model(X_tensor).cpu().numpy().flatten()
            
            return predictions
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def _create_plots_directory(self) -> Path:
        """Create directory for saving plots."""
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        return plots_dir