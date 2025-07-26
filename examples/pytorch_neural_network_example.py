"""
PyTorch Neural Network with Mixed Precision Training Example

This example demonstrates the comprehensive PyTorch neural network implementation
with configurable architecture, mixed precision training, early stopping,
learning rate scheduling, and comprehensive logging capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import time
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pytorch_neural_network import (
    PyTorchNeuralNetworkTrainer, HousingNeuralNetwork, 
    CaliforniaHousingDataset, EarlyStopping, TrainingMetrics
)
from src.gpu_model_trainer_clean import GPUModelTrainer, ModelConfig, PyTorchConfig
from src.mlflow_config import MLflowExperimentManager, MLflowConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main example function demonstrating PyTorch neural network training."""
    print("PyTorch Neural Network with Mixed Precision Training Example")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Feature names: {housing.feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # 2. Setup MLflow experiment tracking
    print("\n2. Setting up MLflow experiment tracking...")
    try:
        mlflow_config = MLflowConfig(
            tracking_uri="sqlite:///pytorch_neural_network_example.db",
            experiment_name="pytorch_neural_network_example"
        )
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        print(f"✓ MLflow tracking URI: {mlflow_config.tracking_uri}")
        print(f"✓ Experiment: {mlflow_config.experiment_name}")
    except Exception as e:
        print(f"✗ MLflow setup failed: {e}")
        print("Continuing without MLflow tracking...")
        mlflow_manager = None
    
    # 3. Configure PyTorch neural network
    print("\n3. Configuring PyTorch neural network...")
    
    # Test different configurations
    configurations = [
        {
            "name": "Small Network (Fast Training)",
            "config": {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'batch_size': 1024,
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'mixed_precision': True,  # Enable mixed precision for GPU
                'early_stopping_patience': 15,
                'lr_scheduler': 'cosine',
                'warmup_epochs': 5,
                'use_batch_norm': True,
                'use_residual': False,
                'optimizer': 'adamw'
            }
        },
        {
            "name": "Deep Network (Production-like)",
            "config": {
                'hidden_layers': [512, 256, 128, 64],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'batch_size': 2048,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'mixed_precision': True,
                'early_stopping_patience': 25,
                'lr_scheduler': 'cosine',
                'warmup_epochs': 10,
                'use_batch_norm': True,
                'use_residual': True,
                'optimizer': 'adamw'
            }
        }
    ]
    
    # Select configuration (use small for example)
    selected_config = configurations[0]
    print(f"Selected configuration: {selected_config['name']}")
    
    config_dict = selected_config['config']
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    # 4. Test PyTorch components individually
    print("\n4. Testing PyTorch components...")
    
    # Test custom dataset
    print("  Testing CaliforniaHousingDataset...")
    dataset = CaliforniaHousingDataset(X_train_scaled[:100], y_train[:100])
    print(f"    ✓ Dataset created with {len(dataset)} samples")
    sample_features, sample_target = dataset[0]
    print(f"    ✓ Sample shape: features={sample_features.shape}, target={sample_target.shape}")
    
    # Test neural network architecture
    print("  Testing HousingNeuralNetwork...")
    model = HousingNeuralNetwork(
        input_size=X_train_scaled.shape[1],
        hidden_layers=config_dict['hidden_layers'],
        activation=config_dict['activation'],
        dropout_rate=config_dict['dropout_rate'],
        use_batch_norm=config_dict['use_batch_norm'],
        use_residual=config_dict['use_residual']
    )
    model_info = model.get_model_info()
    print(f"    ✓ Model created: {model_info['total_parameters']:,} parameters")
    print(f"    ✓ Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"    ✓ Architecture: {model_info['hidden_layers']}")
    
    # Test forward pass
    import torch
    test_input = torch.FloatTensor(X_train_scaled[:10])
    with torch.no_grad():
        output = model(test_input)
    print(f"    ✓ Forward pass successful: input={test_input.shape}, output={output.shape}")
    
    # Test early stopping
    print("  Testing EarlyStopping...")
    early_stopping = EarlyStopping(patience=5, min_delta=1e-6)
    print(f"    ✓ Early stopping initialized with patience={early_stopping.patience}")
    
    # 5. Train PyTorch neural network
    print("\n5. Training PyTorch neural network...")
    
    # Create trainer
    pytorch_trainer = PyTorchNeuralNetworkTrainer(
        config=config_dict,
        mlflow_manager=mlflow_manager
    )
    
    print(f"  Device: {pytorch_trainer.device}")
    print(f"  Mixed precision: {pytorch_trainer.use_mixed_precision}")
    
    # Start MLflow run if available
    run_id = None
    if mlflow_manager:
        try:
            run_id = mlflow_manager.start_run(
                run_name=f"pytorch_nn_{selected_config['name'].lower().replace(' ', '_')}",
                tags={
                    'model_type': 'pytorch_neural_network',
                    'configuration': selected_config['name'],
                    'mixed_precision': str(pytorch_trainer.use_mixed_precision),
                    'device': str(pytorch_trainer.device)
                }
            )
            
            # Log configuration parameters
            mlflow_manager.log_parameters(config_dict)
            print(f"  ✓ MLflow run started: {run_id}")
            
        except Exception as e:
            print(f"  ✗ Failed to start MLflow run: {e}")
            run_id = None
    
    # Train the model
    print("  Starting training...")
    start_time = time.time()
    
    try:
        trained_model = pytorch_trainer.train(
            X_train_scaled, y_train, 
            X_val_scaled, y_val
        )
        
        training_time = time.time() - start_time
        print(f"  ✓ Training completed in {training_time:.2f} seconds")
        
        # Get training history
        history = pytorch_trainer.training_history
        print(f"  ✓ Training epochs: {len(history)}")
        
        if history:
            final_metrics = history[-1]
            print(f"  ✓ Final train loss: {final_metrics.train_loss:.6f}")
            if final_metrics.val_loss:
                print(f"  ✓ Final validation loss: {final_metrics.val_loss:.6f}")
            if final_metrics.train_rmse:
                print(f"  ✓ Final train RMSE: {final_metrics.train_rmse:.4f}")
            if final_metrics.val_rmse:
                print(f"  ✓ Final validation RMSE: {final_metrics.val_rmse:.4f}")
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        if mlflow_manager and run_id:
            mlflow_manager.end_run("FAILED")
        return
    
    # 6. Evaluate model performance
    print("\n6. Evaluating model performance...")
    
    # Make predictions
    train_predictions = pytorch_trainer.predict(trained_model, X_train_scaled)
    val_predictions = pytorch_trainer.predict(trained_model, X_val_scaled)
    test_predictions = pytorch_trainer.predict(trained_model, X_test_scaled)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    def calculate_metrics(y_true, y_pred, dataset_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"  {dataset_name} Metrics:")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE:  {mae:.4f}")
        print(f"    R²:   {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    train_metrics = calculate_metrics(y_train, train_predictions, "Training")
    val_metrics = calculate_metrics(y_val, val_predictions, "Validation")
    test_metrics = calculate_metrics(y_test, test_predictions, "Test")
    
    # 7. Save artifacts and visualizations
    print("\n7. Saving artifacts and visualizations...")
    
    # Save training curves
    curves_path = "plots/pytorch_neural_network_training_curves.png"
    os.makedirs("plots", exist_ok=True)
    
    try:
        pytorch_trainer.save_training_curves(curves_path)
        print(f"  ✓ Training curves saved: {curves_path}")
    except Exception as e:
        print(f"  ✗ Failed to save training curves: {e}")
    
    # Save model checkpoint
    checkpoint_path = "plots/pytorch_neural_network_checkpoint.pth"
    try:
        pytorch_trainer.save_model_checkpoint(trained_model, checkpoint_path)
        print(f"  ✓ Model checkpoint saved: {checkpoint_path}")
    except Exception as e:
        print(f"  ✗ Failed to save checkpoint: {e}")
    
    # Save training history as JSON
    history_path = "plots/pytorch_training_history.json"
    try:
        history_data = [m.to_dict() for m in pytorch_trainer.training_history]
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        print(f"  ✓ Training history saved: {history_path}")
    except Exception as e:
        print(f"  ✗ Failed to save training history: {e}")
    
    # 8. Log final results to MLflow
    if mlflow_manager and run_id:
        print("\n8. Logging results to MLflow...")
        try:
            # Create experiment metrics
            from src.mlflow_config import ExperimentMetrics
            
            final_metrics = ExperimentMetrics(
                rmse=test_metrics['rmse'],
                mae=test_metrics['mae'],
                r2_score=test_metrics['r2'],
                training_time=training_time,
                model_size_mb=model_info['model_size_mb']
            )
            
            # Log metrics
            mlflow_manager.log_metrics(final_metrics)
            
            # Log model
            mlflow_manager.log_model(trained_model, 'pytorch')
            
            # Log artifacts
            from src.mlflow_config import ModelArtifacts
            artifacts = ModelArtifacts(
                model_path=checkpoint_path,
                training_curves_plot=curves_path
            )
            mlflow_manager.log_artifacts(artifacts)
            
            # End run successfully
            mlflow_manager.end_run("FINISHED")
            print(f"  ✓ MLflow run completed successfully")
            
        except Exception as e:
            print(f"  ✗ Failed to log to MLflow: {e}")
            mlflow_manager.end_run("FAILED")
    
    # 9. Test GPU model trainer integration
    print("\n9. Testing GPU model trainer integration...")
    
    try:
        # Create model configuration
        pytorch_config = PyTorchConfig(**config_dict)
        model_config = ModelConfig(pytorch=pytorch_config)
        
        # Create GPU trainer
        gpu_trainer = GPUModelTrainer(model_config, mlflow_manager)
        print(f"  ✓ GPU trainer initialized")
        print(f"  ✓ GPU available: {gpu_trainer.is_gpu_available()}")
        print(f"  ✓ Device info: {gpu_trainer.get_device_info()}")
        
        # Test memory management
        memory_report = gpu_trainer.get_memory_usage_report()
        print(f"  ✓ Memory report generated")
        
        pytorch_memory = memory_report.get('pytorch_memory', {})
        if pytorch_memory.get('available'):
            print(f"    PyTorch allocated: {pytorch_memory.get('allocated_gb', 0):.3f} GB")
            print(f"    PyTorch reserved: {pytorch_memory.get('reserved_gb', 0):.3f} GB")
        
        # Test cleanup
        cleanup_results = gpu_trainer.cleanup_gpu_memory()
        if cleanup_results.get('success'):
            print(f"  ✓ GPU memory cleanup successful")
            if cleanup_results.get('memory_freed_gb') is not None:
                print(f"    Memory freed: {cleanup_results.get('memory_freed_gb', 0):.3f} GB")
        
    except Exception as e:
        print(f"  ✗ GPU trainer integration test failed: {e}")
    
    # 10. Summary and recommendations
    print("\n" + "=" * 70)
    print("PyTorch Neural Network Training Summary")
    print("=" * 70)
    
    print(f"Configuration: {selected_config['name']}")
    print(f"Model Architecture: {model_info['hidden_layers']}")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training Epochs: {len(pytorch_trainer.training_history)}")
    print(f"Mixed Precision: {pytorch_trainer.use_mixed_precision}")
    print(f"Device Used: {pytorch_trainer.device}")
    
    print(f"\nFinal Performance:")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    print(f"  Test R²:   {test_metrics['r2']:.4f}")
    
    print(f"\nKey Features Demonstrated:")
    print(f"  ✓ Configurable neural network architecture")
    print(f"  ✓ Mixed precision training with torch.cuda.amp")
    print(f"  ✓ Custom dataset and dataloader classes")
    print(f"  ✓ Training loop with early stopping")
    print(f"  ✓ Learning rate scheduling (cosine annealing)")
    print(f"  ✓ Comprehensive logging of training curves")
    print(f"  ✓ Model checkpoints and artifact saving")
    print(f"  ✓ MLflow experiment tracking integration")
    print(f"  ✓ GPU memory management and cleanup")
    print(f"  ✓ Batch normalization and residual connections")
    print(f"  ✓ Multiple optimizer support (AdamW, Adam, SGD)")
    print(f"  ✓ Validation and performance metrics")
    
    print(f"\nFiles Generated:")
    print(f"  • Training curves: {curves_path}")
    print(f"  • Model checkpoint: {checkpoint_path}")
    print(f"  • Training history: {history_path}")
    if mlflow_manager:
        print(f"  • MLflow experiment: {mlflow_config.experiment_name}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()