"""
PyTorch Neural Network with Mixed Precision Training

This module implements a comprehensive PyTorch neural network architecture with
configurable hidden layers, mixed precision training using torch.cuda.amp,
custom dataset and dataloader classes, training loop with early stopping,
learning rate scheduling, validation, and comprehensive logging.
"""

import os
import logging
import time
import json
import gc
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .mlflow_config import MLflowExperimentManager, ExperimentMetrics, ModelArtifacts

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Data class for training metrics per epoch."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_rmse: Optional[float] = None
    val_rmse: Optional[float] = None
    train_mae: Optional[float] = None
    val_mae: Optional[float] = None
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    learning_rate: float = 0.0
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    epoch_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_rmse': self.train_rmse,
            'val_rmse': self.val_rmse,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'learning_rate': self.learning_rate,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_utilization': self.gpu_utilization,
            'epoch_time': self.epoch_time
        }


class CaliforniaHousingDataset(Dataset):
    """Custom dataset class for California Housing data with proper tensor handling."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Optional[Callable] = None):
        """
        Initialize dataset.
        
        Args:
            X: Feature array
            y: Target array
            transform: Optional transform to apply to features
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # Add dimension for regression
        self.transform = transform
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (features, target)
        """
        features = self.X[idx]
        target = self.y[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, target


class HousingNeuralNetwork(nn.Module):
    """
    Configurable neural network architecture for California Housing prediction.
    
    Features:
    - Configurable hidden layers
    - Multiple activation functions
    - Dropout regularization
    - Batch normalization
    - Residual connections for deeper networks
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], 
                 activation: str = 'relu', dropout_rate: float = 0.2,
                 use_batch_norm: bool = True, use_residual: bool = False):
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(HousingNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU-like activations
                if self.activation_name in ['relu', 'leaky_relu', 'elu']:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # Xavier initialization for other activations
                    nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = None
        
        for i, layer in enumerate(self.layers):
            # Store residual connection
            if self.use_residual and i > 0 and x.shape[1] == layer.out_features:
                residual = x
            
            # Linear transformation
            x = layer(x)
            
            # Batch normalization
            if self.use_batch_norm and self.batch_norms:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Add residual connection
            if self.use_residual and residual is not None and x.shape == residual.shape:
                x = x + residual
            
            # Dropout
            x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'HousingNeuralNetwork',
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'use_residual': self.use_residual,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info(f"Restored best weights from {self.patience} epochs ago")
        
        return self.early_stop


class PyTorchNeuralNetworkTrainer:
    """
    Comprehensive PyTorch neural network trainer with mixed precision,
    early stopping, learning rate scheduling, and comprehensive logging.
    """
    
    def __init__(self, config: Dict[str, Any], mlflow_manager: Optional[MLflowExperimentManager] = None):
        """
        Initialize PyTorch trainer.
        
        Args:
            config: Training configuration dictionary
            mlflow_manager: Optional MLflow experiment manager
        """
        self.config = config
        self.mlflow_manager = mlflow_manager
        
        # Setup device
        self.device = self._setup_device()
        
        # Mixed precision training
        self.use_mixed_precision = config.get('mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.training_history = []
        
        logger.info(f"PyTorch trainer initialized with device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
    
    def _setup_device(self) -> torch.device:
        """Setup and validate CUDA device."""
        device_str = self.config.get('device', 'cuda')
        
        if device_str == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            
            # Log GPU memory info
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU memory: {total_memory:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.warning("Using CPU device")
        
        return device
    
    def _create_model(self, input_size: int) -> HousingNeuralNetwork:
        """Create neural network model."""
        model = HousingNeuralNetwork(
            input_size=input_size,
            hidden_layers=self.config.get('hidden_layers', [512, 256, 128, 64]),
            activation=self.config.get('activation', 'relu'),
            dropout_rate=self.config.get('dropout_rate', 0.2),
            use_batch_norm=self.config.get('use_batch_norm', True),
            use_residual=self.config.get('use_residual', False)
        )
        
        return model.to(self.device)
    
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, 
                         total_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('lr_scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=self.config.get('learning_rate', 0.001) * 0.01
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=total_epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
        
        return scheduler
    
    def _create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, 
                           y_val: Optional[np.ndarray] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create data loaders."""
        batch_size = self.config.get('batch_size', 2048)
        
        # Create datasets
        train_dataset = CaliforniaHousingDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=self.device.type == 'cuda'
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = CaliforniaHousingDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device.type == 'cuda'
            )
        
        return train_loader, val_loader
    
    def _calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Calculate regression metrics."""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true_np, y_pred_np)),
            'mae': mean_absolute_error(y_true_np, y_pred_np),
            'r2': r2_score(y_true_np, y_pred_np)
        }
    
    def _get_gpu_metrics(self) -> Dict[str, Optional[float]]:
        """Get current GPU metrics."""
        if not torch.cuda.is_available():
            return {'gpu_memory_mb': None, 'gpu_utilization': None}
        
        try:
            # PyTorch GPU memory
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            # Try to get NVIDIA metrics
            gpu_utilization = None
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except:
                pass
            
            return {
                'gpu_memory_mb': memory_allocated,
                'gpu_utilization': gpu_utilization
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU metrics: {e}")
            return {'gpu_memory_mb': None, 'gpu_utilization': None}
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_targets = batch_targets.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with autocast('cuda'):
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_targets)
                else:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                all_targets.append(batch_targets)
                all_predictions.append(outputs)
        
        # Calculate metrics
        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)
        metrics = self._calculate_metrics(all_targets, all_predictions)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> nn.Module:
        """
        Train the PyTorch neural network with comprehensive features.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained model
        """
        logger.info("Starting PyTorch neural network training...")
        
        # Create model
        self.model = self._create_model(X_train.shape[1])
        model_info = self.model.get_model_info()
        logger.info(f"Model architecture: {model_info}")
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(self.model)
        epochs = self.config.get('epochs', 500)
        self.scheduler = self._create_scheduler(self.optimizer, epochs)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X_train, y_train, X_val, y_val)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Early stopping
        if val_loader is not None:
            patience = self.config.get('early_stopping_patience', 50)
            self.early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        self.training_history = []
        best_val_loss = float('inf')
        
        # Warmup epochs
        warmup_epochs = self.config.get('warmup_epochs', 10)
        
        logger.info(f"Training for up to {epochs} epochs with warmup: {warmup_epochs}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Warmup learning rate
            if epoch < warmup_epochs and self.scheduler:
                warmup_lr = self.config.get('learning_rate', 0.001) * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Train epoch
            train_loss = self._train_epoch(self.model, train_loader, self.optimizer, criterion)
            
            # Validation
            val_loss = None
            val_metrics = {}
            if val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(self.model, val_loader, criterion)
            
            # Calculate training metrics
            self.model.eval()
            with torch.no_grad():
                train_outputs = []
                train_targets = []
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    if self.use_mixed_precision:
                        with autocast('cuda'):
                            outputs = self.model(batch_features)
                    else:
                        outputs = self.model(batch_features)
                    
                    train_outputs.append(outputs)
                    train_targets.append(batch_targets)
                
                train_outputs = torch.cat(train_outputs)
                train_targets = torch.cat(train_targets)
                train_metrics = self._calculate_metrics(train_targets, train_outputs)
            
            # Learning rate scheduling
            if self.scheduler and epoch >= warmup_epochs:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Get GPU metrics
            gpu_metrics = self._get_gpu_metrics()
            
            # Create training metrics
            epoch_time = time.time() - epoch_start_time
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_rmse=train_metrics.get('rmse'),
                val_rmse=val_metrics.get('rmse'),
                train_mae=train_metrics.get('mae'),
                val_mae=val_metrics.get('mae'),
                train_r2=train_metrics.get('r2'),
                val_r2=val_metrics.get('r2'),
                learning_rate=current_lr,
                gpu_memory_mb=gpu_metrics.get('gpu_memory_mb'),
                gpu_utilization=gpu_metrics.get('gpu_utilization'),
                epoch_time=epoch_time
            )
            
            self.training_history.append(metrics)
            
            # Log to MLflow
            if self.mlflow_manager:
                try:
                    metrics_dict = metrics.to_dict()
                    for key, value in metrics_dict.items():
                        if value is not None:
                            self.mlflow_manager.client.log_metric(
                                run_id=self.mlflow_manager.client.active_run().info.run_id,
                                key=key,
                                value=value,
                                step=epoch
                            )
                except Exception as e:
                    logger.debug(f"Failed to log metrics to MLflow: {e}")
            
            # Progress logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = f"Epoch {epoch + 1}/{epochs}: "
                log_msg += f"Train Loss: {train_loss:.6f}, Train RMSE: {train_metrics.get('rmse', 0):.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}, Val RMSE: {val_metrics.get('rmse', 0):.4f}"
                log_msg += f", LR: {current_lr:.2e}, Time: {epoch_time:.2f}s"
                logger.info(log_msg)
            
            # Early stopping
            if self.early_stopping and val_loss is not None:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
        
        logger.info("PyTorch training completed")
        return self.model
    
    def predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions
        """
        model.eval()
        dataset = CaliforniaHousingDataset(X, np.zeros(len(X)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=self.config.get('batch_size', 2048), shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_features, _ in loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        outputs = model(batch_features)
                else:
                    outputs = model(batch_features)
                
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).flatten()
    
    def save_training_curves(self, save_path: str) -> str:
        """Save training curves plot."""
        if not self.training_history:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PyTorch Neural Network Training Progress', fontsize=16)
        
        epochs = [m.epoch for m in self.training_history]
        
        # Loss curves
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history if m.val_loss is not None]
        
        axes[0, 0].plot(epochs, train_losses, label='Training Loss', color='blue')
        if val_losses:
            axes[0, 0].plot(epochs[:len(val_losses)], val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RMSE curves
        train_rmse = [m.train_rmse for m in self.training_history if m.train_rmse is not None]
        val_rmse = [m.val_rmse for m in self.training_history if m.val_rmse is not None]
        
        if train_rmse:
            axes[0, 1].plot(epochs[:len(train_rmse)], train_rmse, label='Training RMSE', color='blue')
        if val_rmse:
            axes[0, 1].plot(epochs[:len(val_rmse)], val_rmse, label='Validation RMSE', color='red')
        axes[0, 1].set_title('RMSE Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        learning_rates = [m.learning_rate for m in self.training_history]
        axes[0, 2].plot(epochs, learning_rates, color='green')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # GPU memory usage
        gpu_memory = [m.gpu_memory_mb for m in self.training_history if m.gpu_memory_mb is not None]
        if gpu_memory:
            axes[1, 0].plot(epochs[:len(gpu_memory)], gpu_memory, color='purple')
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU Memory\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # GPU utilization
        gpu_util = [m.gpu_utilization for m in self.training_history if m.gpu_utilization is not None]
        if gpu_util:
            axes[1, 1].plot(epochs[:len(gpu_util)], gpu_util, color='orange')
            axes[1, 1].set_title('GPU Utilization')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Utilization (%)')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'GPU Utilization\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Epoch time
        epoch_times = [m.epoch_time for m in self.training_history]
        axes[1, 2].plot(epochs, epoch_times, color='brown')
        axes[1, 2].set_title('Epoch Training Time')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def save_model_checkpoint(self, model: nn.Module, save_path: str) -> str:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info(),
            'config': self.config,
            'training_history': [m.to_dict() for m in self.training_history]
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model checkpoint saved to {save_path}")
        return save_path
    
    def load_model_checkpoint(self, checkpoint_path: str, input_size: int) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with same architecture
        model = self._create_model(input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training history
        if 'training_history' in checkpoint:
            self.training_history = [
                TrainingMetrics(**m) for m in checkpoint['training_history']
            ]
        
        logger.info(f"Model checkpoint loaded from {checkpoint_path}")
        return model