#!/usr/bin/env python3
"""
LightGBM GPU Training Example

This example demonstrates comprehensive LightGBM GPU training with:
- GPU acceleration and optimized parameters
- Feature importance extraction and visualization
- Cross-validation and model evaluation
- MLflow experiment tracking
- Performance comparison utilities

Usage:
    python examples/lightgbm_gpu_example.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_manager import DataManager
from src.mlflow_config import MLflowExperimentManager, MLflowConfig
from src.gpu_model_trainer import GPUModelTrainer, ModelConfig, LightGBMConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate LightGBM GPU training."""
    print("=" * 80)
    print("LightGBM GPU Training Example")
    print("=" * 80)
    
    try:
        # 1. Setup data management
        print("\n1. Setting up data management...")
        data_manager = DataManager()
        
        # Load California Housing dataset
        print("   Loading California Housing dataset...")
        features_df, targets_series = data_manager.download_raw_data()
        print(f"   Dataset shape: {features_df.shape}")
        
        # Prepare features and target
        X = features_df.values
        y = targets_series.values
        
        print(f"   Features: {features_df.shape[1]}")
        print(f"   Samples: {len(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"   Training samples: {len(y_train)}")
        print(f"   Validation samples: {len(y_val)}")
        print(f"   Test samples: {len(y_test)}")
        
        # 2. Setup MLflow experiment tracking
        print("\n2. Setting up MLflow experiment tracking...")
        mlflow_config = MLflowConfig(
            tracking_uri="file:./mlruns",  # Use local file-based tracking
            experiment_name="lightgbm_gpu_training_demo"
        )
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        
        print(f"   MLflow experiment: {mlflow_config.experiment_name}")
        print(f"   Experiment ID: {mlflow_manager.experiment_id}")
        
        # 3. Configure LightGBM with optimized parameters
        print("\n3. Configuring LightGBM with GPU acceleration...")
        
        # Custom LightGBM configuration optimized for regression
        lightgbm_config = LightGBMConfig(
            device_type='gpu',  # Enable GPU acceleration
            n_estimators=2000,  # Increased for better performance
            num_leaves=255,     # Optimized for regression
            max_depth=12,       # Deep trees for complex patterns
            learning_rate=0.05, # Moderate learning rate
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,      # L1 regularization
            reg_lambda=1.0,     # L2 regularization
            early_stopping_rounds=100,
            random_state=42
        )
        
        model_config = ModelConfig(lightgbm=lightgbm_config)
        
        print("   LightGBM Configuration:")
        print(f"     Device: {lightgbm_config.device_type}")
        print(f"     Estimators: {lightgbm_config.n_estimators}")
        print(f"     Num leaves: {lightgbm_config.num_leaves}")
        print(f"     Max depth: {lightgbm_config.max_depth}")
        print(f"     Learning rate: {lightgbm_config.learning_rate}")
        
        # 4. Initialize GPU model trainer
        print("\n4. Initializing GPU model trainer...")
        trainer = GPUModelTrainer(model_config, mlflow_manager)
        
        # Check GPU availability
        gpu_info = trainer.get_device_info()
        print(f"   GPU available: {gpu_info.get('available', False)}")
        if gpu_info.get('available'):
            print(f"   GPU name: {gpu_info.get('name', 'Unknown')}")
            print(f"   CUDA version: {gpu_info.get('cuda_version', 'Unknown')}")
        
        # 5. Train LightGBM model
        print("\n5. Training LightGBM model with GPU acceleration...")
        print("   This may take several minutes depending on your GPU...")
        
        # Start MLflow run
        run_name = "lightgbm_gpu_demo_run"
        run_id = mlflow_manager.start_run(
            run_name=run_name,
            tags={
                "model_type": "lightgbm",
                "gpu_enabled": str(trainer.is_gpu_available()),
                "example_run": "true"
            }
        )
        
        # Log configuration
        mlflow_manager.log_parameters(lightgbm_config.model_dump())
        
        # Train the model
        model_result = trainer._train_lightgbm(X_train, y_train, X_val, y_val)
        
        print("   Training completed!")
        print(f"   Best iteration: {model_result.get('best_iteration', 'N/A')}")
        print(f"   Training time: {model_result.get('training_time', 0):.2f} seconds")
        
        # 6. Model evaluation and performance comparison
        print("\n6. Evaluating model performance...")
        
        # Get the trained model
        trained_model = model_result['model']
        
        # Make predictions
        train_pred = trained_model.predict(X_train)
        val_pred = trained_model.predict(X_val)
        test_pred = trained_model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred),
            'r2': r2_score(y_train, train_pred)
        }
        
        val_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        }
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        }
        
        print("   Performance Metrics:")
        print(f"     Training   - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"     Validation - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"     Test       - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        # Log metrics to MLflow
        mlflow_manager.log_parameters({
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2']
        })
        
        # 7. Feature importance analysis
        print("\n7. Analyzing feature importance...")
        
        feature_importance = model_result.get('feature_importance', [])
        feature_names = model_result.get('feature_names', [])
        
        if len(feature_importance) > 0 and len(feature_names) > 0:
            # Sort features by importance
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print("   Top 5 Most Important Features:")
            for i, (name, importance) in enumerate(importance_pairs[:5]):
                print(f"     {i+1}. {name}: {importance:.0f}")
        
        # 8. Display plots information
        print("\n8. Generated visualizations:")
        plots_info = model_result.get('plots', {})
        for plot_type, plot_path in plots_info.items():
            if os.path.exists(plot_path):
                print(f"   {plot_type.replace('_', ' ').title()}: {plot_path}")
        
        # 9. GPU metrics summary
        print("\n9. GPU metrics summary:")
        final_gpu_metrics = trainer.get_gpu_metrics()
        if final_gpu_metrics:
            print(f"   GPU utilization: {final_gpu_metrics.utilization_percent:.1f}%")
            print(f"   GPU memory used: {final_gpu_metrics.memory_used_mb:.1f} MB")
            print(f"   GPU temperature: {final_gpu_metrics.temperature_celsius:.1f}°C")
            print(f"   GPU power usage: {final_gpu_metrics.power_usage_watts:.1f} W")
        else:
            print("   GPU metrics not available")
        
        # End MLflow run
        mlflow_manager.end_run("FINISHED")
        
        print("\n" + "=" * 80)
        print("LightGBM GPU Training Example Completed Successfully!")
        print("=" * 80)
        print(f"MLflow UI: http://localhost:5000")
        print(f"Check the 'plots' directory for generated visualizations")
        
        return {
            'model_result': model_result,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'plots': plots_info
        }
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nError: {e}")
        
        # End MLflow run with failure
        try:
            mlflow_manager.end_run("FAILED")
        except:
            pass
        
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print("\nExample completed successfully!")
    else:
        print("\nExample failed!")
        sys.exit(1)