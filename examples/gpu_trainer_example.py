"""
GPU Model Trainer Example

This example demonstrates how to use the GPU-accelerated model training infrastructure
with comprehensive configuration, monitoring, and experiment tracking.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu_model_trainer_clean import (
    GPUModelTrainer, ModelConfig, XGBoostConfig, LightGBMConfig, 
    PyTorchConfig, CuMLConfig, GPUMonitor, GPUMemoryManager
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("GPU Model Trainer Example")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\n1. Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 2. Setup GPU monitoring
    print("\n2. Setting up GPU monitoring...")
    gpu_monitor = GPUMonitor()
    device_info = gpu_monitor.get_device_info()
    print(f"GPU Available: {device_info.get('available', False)}")
    if device_info.get('available'):
        print(f"GPU Name: {device_info.get('name', 'Unknown')}")
        print(f"Driver Version: {device_info.get('driver_version', 'Unknown')}")
        print(f"CUDA Version: {device_info.get('cuda_version', 'Unknown')}")
        
        # Get current GPU metrics
        metrics = gpu_monitor.get_metrics()
        if metrics:
            print(f"GPU Utilization: {metrics.utilization_percent}%")
            print(f"GPU Memory Used: {metrics.memory_used_mb:.1f} MB")
            print(f"GPU Temperature: {metrics.temperature_celsius}°C")
    
    # 3. Configure models
    print("\n3. Configuring model parameters...")
    
    # Custom XGBoost configuration for faster training in example
    xgboost_config = XGBoostConfig(
        n_estimators=100,  # Reduced for faster example
        max_depth=6,
        learning_rate=0.1,
        tree_method='gpu_hist' if device_info.get('available') else 'hist'
    )
    
    # Custom LightGBM configuration
    lightgbm_config = LightGBMConfig(
        n_estimators=100,  # Reduced for faster example
        num_leaves=31,
        learning_rate=0.1,
        device_type='gpu' if device_info.get('available') else 'cpu'
    )
    
    # Custom PyTorch configuration
    pytorch_config = PyTorchConfig(
        hidden_layers=[256, 128, 64],  # Smaller network for example
        epochs=50,  # Reduced for faster example
        batch_size=1024,
        learning_rate=0.001,
        device='cuda' if device_info.get('available') else 'cpu'
    )
    
    # Create complete model configuration
    model_config = ModelConfig(
        xgboost=xgboost_config,
        lightgbm=lightgbm_config,
        pytorch=pytorch_config
    )
    
    print("Model configurations:")
    print(f"  XGBoost: {model_config.xgboost.n_estimators} estimators, {model_config.xgboost.tree_method}")
    print(f"  LightGBM: {model_config.lightgbm.n_estimators} estimators, {model_config.lightgbm.device_type}")
    print(f"  PyTorch: {model_config.pytorch.epochs} epochs, {model_config.pytorch.device}")
    
    # 4. Setup MLflow experiment tracking
    print("\n4. Setting up MLflow experiment tracking...")
    try:
        mlflow_config = MLflowConfig(
            tracking_uri="sqlite:///mlflow_example.db",
            experiment_name="gpu_trainer_example"
        )
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        print(f"MLflow tracking URI: {mlflow_config.tracking_uri}")
        print(f"Experiment: {mlflow_config.experiment_name}")
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        print("Continuing without MLflow tracking...")
        mlflow_manager = None
    
    # 5. Initialize GPU model trainer
    print("\n5. Initializing GPU model trainer...")
    if mlflow_manager:
        trainer = GPUModelTrainer(model_config, mlflow_manager)
        print("GPU Model Trainer initialized with MLflow integration")
    else:
        print("Skipping trainer initialization due to MLflow setup failure")
        return
    
    # 6. Demonstrate configuration validation
    print("\n6. Model configuration validation:")
    try:
        # Test configuration serialization
        config_dict = model_config.model_dump()
        print("✓ Configuration serialization successful")
        
        # Test individual configurations
        print(f"✓ XGBoost tree method: {model_config.xgboost.tree_method}")
        print(f"✓ LightGBM device type: {model_config.lightgbm.device_type}")
        print(f"✓ PyTorch device: {model_config.pytorch.device}")
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # 7. Demonstrate GPU metrics collection
    print("\n7. GPU metrics collection:")
    if gpu_monitor.available:
        for i in range(3):
            metrics = gpu_monitor.get_metrics()
            if metrics:
                print(f"  Sample {i+1}: {metrics.utilization_percent}% utilization, "
                      f"{metrics.memory_used_mb:.0f}MB memory, {metrics.temperature_celsius}°C")
            else:
                print(f"  Sample {i+1}: No metrics available")
    else:
        print("  GPU monitoring not available")
    
    # 8. Demonstrate training infrastructure (without actual training)
    print("\n8. Training infrastructure demonstration:")
    print(f"  Trainer device info: {trainer.get_device_info()}")
    print(f"  GPU available: {trainer.is_gpu_available()}")
    print(f"  Current training session: {trainer.current_training}")
    
    # Test metrics calculation
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    metrics = trainer._calculate_metrics(y_true, y_pred)
    print(f"  Sample metrics calculation: RMSE={metrics['rmse']:.3f}, "
          f"MAE={metrics['mae']:.3f}, R²={metrics['r2_score']:.3f}")
    
    # 9. Demonstrate PyTorch model creation
    print("\n9. PyTorch model architecture:")
    try:
        model = trainer._create_pytorch_model(X_train.shape[1], model_config.pytorch)
        print(f"  Model created successfully")
        print(f"  Input features: {X_train.shape[1]}")
        print(f"  Hidden layers: {model_config.pytorch.hidden_layers}")
        print(f"  Activation: {model_config.pytorch.activation}")
        
        # Test forward pass
        import torch
        test_input = torch.randn(1, X_train.shape[1])
        with torch.no_grad():
            output = model(test_input)
        print(f"  Test forward pass output shape: {output.shape}")
        
    except Exception as e:
        print(f"  Model creation failed: {e}")
    
    # 10. Demonstrate VRAM cleanup functionality
    print("\n10. VRAM cleanup and memory management:")
    
    # Get initial memory report
    memory_report = trainer.get_memory_usage_report()
    print(f"  Initial memory report:")
    pytorch_memory = memory_report.get('pytorch_memory', {})
    if pytorch_memory.get('available'):
        print(f"    PyTorch allocated: {pytorch_memory.get('allocated_gb', 0):.3f} GB")
        print(f"    PyTorch reserved: {pytorch_memory.get('reserved_gb', 0):.3f} GB")
    else:
        print("    PyTorch memory info not available (CPU mode)")
    
    nvidia_metrics = memory_report.get('nvidia_metrics')
    if nvidia_metrics:
        print(f"    NVIDIA memory used: {nvidia_metrics.get('memory_used_mb', 0):.0f} MB")
        print(f"    NVIDIA memory utilization: {nvidia_metrics.get('memory_utilization_percent', 0):.1f}%")
    else:
        print("    NVIDIA metrics not available")
    
    # Show recommendations
    recommendations = memory_report.get('recommendations', [])
    if recommendations:
        print("  Memory recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")
    else:
        print("  No memory optimization recommendations")
    
    # Demonstrate cleanup functionality
    print("\n  Testing GPU memory cleanup...")
    cleanup_results = trainer.cleanup_gpu_memory()
    
    if cleanup_results.get('success'):
        print("  ✓ GPU memory cleanup successful")
        
        if cleanup_results.get('memory_freed_gb') is not None:
            memory_freed = cleanup_results.get('memory_freed_gb', 0)
            print(f"    Memory freed: {memory_freed:.3f} GB ({memory_freed * 1024:.1f} MB)")
            
            if cleanup_results.get('cleanup_effective'):
                print("    ✓ Cleanup was effective (>10MB freed)")
            else:
                print("    ℹ Minimal memory freed (cleanup still successful)")
        else:
            print("    Memory cleanup completed (GPU not available)")
    else:
        error = cleanup_results.get('error', 'Unknown error')
        print(f"  ✗ GPU memory cleanup failed: {error}")
    
    # Get final memory report
    final_memory_report = trainer.get_memory_usage_report()
    print(f"\n  Final memory report:")
    final_pytorch_memory = final_memory_report.get('pytorch_memory', {})
    if final_pytorch_memory.get('available'):
        print(f"    PyTorch allocated: {final_pytorch_memory.get('allocated_gb', 0):.3f} GB")
        print(f"    PyTorch reserved: {final_pytorch_memory.get('reserved_gb', 0):.3f} GB")
    
    # 11. Summary
    print("\n" + "=" * 50)
    print("GPU Model Trainer Example Summary:")
    print(f"  ✓ Data loaded: {X_train.shape[0]} training samples")
    print(f"  ✓ GPU monitoring: {'Available' if gpu_monitor.available else 'Not available'}")
    print(f"  ✓ Model configurations: XGBoost, LightGBM, PyTorch")
    print(f"  ✓ MLflow integration: {'Enabled' if mlflow_manager else 'Disabled'}")
    print(f"  ✓ Training infrastructure: Ready")
    print(f"  ✓ VRAM cleanup: {'Functional' if cleanup_results.get('success') else 'Failed'}")
    print("\nKey VRAM management features:")
    print("  • Automatic cleanup after each model training")
    print("  • Manual cleanup with trainer.cleanup_gpu_memory()")
    print("  • Memory usage reporting with trainer.get_memory_usage_report()")
    print("  • Context managers for automatic memory management")
    print("  • Comprehensive tensor and model reference cleanup")
    print("\nTo run actual training, use the async training methods:")
    print("  trainer.start_training_async(X_train, y_train, X_val, y_val)")
    print("  # Training will automatically clean up VRAM after each model")
    print("=" * 50)


if __name__ == "__main__":
    main()