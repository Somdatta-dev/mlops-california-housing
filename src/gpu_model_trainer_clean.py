"""
GPU-Accelerated Model Training Infrastructure with VRAM Cleanup

This module provides comprehensive GPU-accelerated model training capabilities
with CUDA device detection, configuration management, GPU metrics collection,
VRAM cleanup, and training progress tracking for the MLOps platform.
"""

import os
import logging
import time
import json
import gc
import weakref
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
from pathlib import Path
import contextlib

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
from .pytorch_neural_network import PyTorchNeuralNetworkTrainer, HousingNeuralNetwork

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """GPU memory management utilities for VRAM cleanup and monitoring."""
    
    @staticmethod
    def clear_gpu_memory():
        """Comprehensive GPU memory cleanup."""
        if torch.cuda.is_available():
            try:
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Synchronize CUDA operations
                torch.cuda.synchronize()
                
                logger.debug("GPU memory cleared successfully")
                
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            # Get memory info from PyTorch
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            max_reserved = torch.cuda.max_memory_reserved() / (1024**3)    # GB
            
            return {
                "available": True,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "max_reserved_gb": max_reserved,
                "free_gb": reserved - allocated
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {"available": False, "error": str(e)}
    
    @staticmethod
    def reset_peak_memory_stats():
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                logger.debug("GPU peak memory stats reset")
            except Exception as e:
                logger.warning(f"Failed to reset peak memory stats: {e}")
    
    @staticmethod
    @contextlib.contextmanager
    def gpu_memory_context():
        """Context manager for automatic GPU memory cleanup."""
        initial_memory = GPUMemoryManager.get_gpu_memory_info()
        logger.debug(f"Initial GPU memory: {initial_memory}")
        
        try:
            yield
        finally:
            # Cleanup GPU memory
            GPUMemoryManager.clear_gpu_memory()
            
            final_memory = GPUMemoryManager.get_gpu_memory_info()
            logger.debug(f"Final GPU memory: {final_memory}")
    
    @staticmethod
    def cleanup_model_references(*models):
        """Clean up model references and associated GPU memory."""
        for model in models:
            if model is not None:
                try:
                    # Move model to CPU if it's a PyTorch model
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    
                    # Delete model reference
                    del model
                    
                except Exception as e:
                    logger.warning(f"Failed to cleanup model reference: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        GPUMemoryManager.clear_gpu_memory()


# Import the rest from the original file
from .gpu_model_trainer import (
    GPUMetrics, TrainingProgress, XGBoostConfig, LightGBMConfig, 
    PyTorchConfig, CuMLConfig, ModelConfig, GPUMonitor, BaseModelTrainer
)


class GPUModelTrainerWithCleanup:
    """
    Enhanced GPU model trainer with comprehensive VRAM cleanup capabilities.
    """
    
    def __init__(self, config: ModelConfig, mlflow_manager: MLflowExperimentManager):
        """Initialize GPU model trainer with cleanup capabilities."""
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
        
        logger.info("GPUModelTrainerWithCleanup initialized")
        logger.info(f"GPU Info: {self.device_info}")
    
    def cleanup_gpu_memory(self) -> Dict[str, Any]:
        """
        Comprehensive GPU memory cleanup and reporting.
        
        Returns:
            Dictionary with cleanup results and memory info
        """
        logger.info("Starting comprehensive GPU memory cleanup...")
        
        # Get initial memory state
        initial_memory = GPUMemoryManager.get_gpu_memory_info()
        
        try:
            # Stop any ongoing training
            if self.training_thread and self.training_thread.is_alive():
                logger.info("Stopping ongoing training for memory cleanup...")
                self.stop_training.set()
                
                # Wait for training to stop (with timeout)
                self.training_thread.join(timeout=30)
                if self.training_thread.is_alive():
                    logger.warning("Training thread did not stop within timeout")
            
            # Comprehensive cleanup
            GPUMemoryManager.clear_gpu_memory()
            
            # Additional cleanup steps
            if torch.cuda.is_available():
                # Clear all cached memory
                torch.cuda.empty_cache()
                
                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Force synchronization
                torch.cuda.synchronize()
                
                # Multiple cleanup passes
                for i in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(0.1)  # Small delay between cleanup passes
            
            # Get final memory state
            final_memory = GPUMemoryManager.get_gpu_memory_info()
            
            # Calculate cleanup results
            cleanup_results = {
                "success": True,
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "cleanup_timestamp": datetime.now().isoformat()
            }
            
            if initial_memory.get('available') and final_memory.get('available'):
                memory_freed = initial_memory.get('allocated_gb', 0) - final_memory.get('allocated_gb', 0)
                cleanup_results.update({
                    "memory_freed_gb": memory_freed,
                    "memory_freed_mb": memory_freed * 1024,
                    "cleanup_effective": memory_freed > 0.01  # More than 10MB freed
                })
                
                logger.info(f"GPU memory cleanup completed. Freed: {memory_freed:.3f} GB")
            else:
                logger.info("GPU memory cleanup completed (GPU not available)")
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "initial_memory": initial_memory,
                "cleanup_timestamp": datetime.now().isoformat()
            }
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU memory usage report.
        
        Returns:
            Dictionary with detailed memory usage information
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "pytorch_memory": GPUMemoryManager.get_gpu_memory_info(),
            "nvidia_metrics": None,
            "recommendations": []
        }
        
        # Get NVIDIA metrics if available
        gpu_metrics = self.gpu_monitor.get_metrics()
        if gpu_metrics:
            report["nvidia_metrics"] = {
                "memory_used_mb": gpu_metrics.memory_used_mb,
                "memory_total_mb": gpu_metrics.memory_total_mb,
                "memory_free_mb": gpu_metrics.memory_free_mb,
                "memory_utilization_percent": (gpu_metrics.memory_used_mb / gpu_metrics.memory_total_mb) * 100,
                "gpu_utilization_percent": gpu_metrics.utilization_percent,
                "temperature_celsius": gpu_metrics.temperature_celsius
            }
        
        # Add recommendations based on memory usage
        pytorch_memory = report["pytorch_memory"]
        if pytorch_memory.get('available'):
            allocated_gb = pytorch_memory.get('allocated_gb', 0)
            reserved_gb = pytorch_memory.get('reserved_gb', 0)
            
            if allocated_gb > 8:  # More than 8GB allocated
                report["recommendations"].append("High GPU memory usage detected. Consider reducing batch size or model complexity.")
            
            if reserved_gb - allocated_gb > 2:  # More than 2GB reserved but not allocated
                report["recommendations"].append("Significant reserved but unused GPU memory. Consider calling cleanup_gpu_memory().")
            
            if gpu_metrics and gpu_metrics.memory_used_mb > gpu_metrics.memory_total_mb * 0.9:
                report["recommendations"].append("GPU memory usage > 90%. Risk of out-of-memory errors.")
        
        return report
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        return self.device_info
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for training."""
        return torch.cuda.is_available() and self.device_info.get('available', False)
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        return self.gpu_monitor.get_metrics()
    
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
    
    def train_pytorch_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: Optional[np.ndarray] = None, 
                                   y_val: Optional[np.ndarray] = None) -> Tuple[HousingNeuralNetwork, Dict[str, Any]]:
        """
        Train PyTorch neural network with mixed precision, early stopping, and comprehensive logging.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Tuple of (trained_model, training_results)
        """
        logger.info("Starting PyTorch neural network training with mixed precision...")
        
        start_time = time.time()
        
        # Use GPU memory context for automatic cleanup
        with GPUMemoryManager.gpu_memory_context():
            try:
                # Convert PyTorchConfig to dictionary for trainer
                pytorch_config_dict = self.config.pytorch.model_dump()
                
                # Create PyTorch trainer
                pytorch_trainer = PyTorchNeuralNetworkTrainer(
                    config=pytorch_config_dict,
                    mlflow_manager=self.mlflow_manager
                )
                
                # Train the model
                trained_model = pytorch_trainer.train(X_train, y_train, X_val, y_val)
                
                # Calculate final metrics
                train_predictions = pytorch_trainer.predict(trained_model, X_train)
                train_metrics = self._calculate_metrics(y_train, train_predictions)
                
                val_metrics = {}
                if X_val is not None and y_val is not None:
                    val_predictions = pytorch_trainer.predict(trained_model, X_val)
                    val_metrics = self._calculate_metrics(y_val, val_predictions)
                
                # Get model info
                model_info = trained_model.get_model_info()
                
                # Save training curves
                plots_dir = self._create_plots_directory()
                curves_path = plots_dir / "PyTorch_training_curves.png"
                pytorch_trainer.save_training_curves(str(curves_path))
                
                # Save model checkpoint
                checkpoint_path = plots_dir / "pytorch_model_checkpoint.pth"
                pytorch_trainer.save_model_checkpoint(trained_model, str(checkpoint_path))
                
                # Get final GPU metrics
                gpu_metrics = self.get_gpu_metrics()
                
                # Prepare training results
                training_results = {
                    'model_info': model_info,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'training_time': time.time() - start_time,
                    'training_history': [m.to_dict() for m in pytorch_trainer.training_history],
                    'curves_plot_path': str(curves_path),
                    'checkpoint_path': str(checkpoint_path),
                    'gpu_metrics': gpu_metrics.to_dict() if gpu_metrics else None,
                    'mixed_precision_used': pytorch_trainer.use_mixed_precision,
                    'device_used': str(pytorch_trainer.device)
                }
                
                logger.info(f"PyTorch training completed successfully in {training_results['training_time']:.2f} seconds")
                logger.info(f"Final train RMSE: {train_metrics['rmse']:.4f}")
                if val_metrics:
                    logger.info(f"Final validation RMSE: {val_metrics['rmse']:.4f}")
                logger.info(f"Model parameters: {model_info['total_parameters']:,}")
                logger.info(f"Mixed precision: {pytorch_trainer.use_mixed_precision}")
                
                return trained_model, training_results
                
            except Exception as e:
                logger.error(f"PyTorch training failed: {e}")
                raise
    
    def predict_pytorch(self, model: HousingNeuralNetwork, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained PyTorch model.
        
        Args:
            model: Trained PyTorch model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        # Create temporary trainer for prediction
        pytorch_config_dict = self.config.pytorch.model_dump()
        pytorch_trainer = PyTorchNeuralNetworkTrainer(config=pytorch_config_dict)
        
        return pytorch_trainer.predict(model, X)


# For backward compatibility, alias the enhanced trainer
GPUModelTrainer = GPUModelTrainerWithCleanup