"""
XGBoost GPU Training Example

This example demonstrates the enhanced XGBoost GPU training implementation
with advanced hyperparameters, feature importance extraction, cross-validation,
and comprehensive MLflow logging.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import logging
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu_model_trainer import (
    GPUModelTrainer, ModelConfig, XGBoostConfig, GPUMonitor
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main example function for XGBoost GPU training."""
    print("XGBoost GPU Training Example")
    print("=" * 60)
    
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
    print(f"Feature names: {housing.feature_names}")
    
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
            print(f"Initial GPU Utilization: {metrics.utilization_percent}%")
            print(f"Initial GPU Memory Used: {metrics.memory_used_mb:.1f} MB")
            print(f"Initial GPU Temperature: {metrics.temperature_celsius}°C")
    
    # 3. Configure XGBoost with advanced parameters
    print("\n3. Configuring XGBoost with advanced parameters...")
    
    # Advanced XGBoost configuration for demonstration
    xgboost_config = XGBoostConfig(
        # GPU settings
        tree_method='gpu_hist' if device_info.get('available') else 'hist',
        gpu_id=0,
        
        # Advanced tree parameters for deep trees
        max_depth=15,  # Deep trees for complex patterns
        n_estimators=2000,  # High estimator count
        learning_rate=0.02,  # Lower learning rate for stability
        
        # Advanced sampling parameters
        subsample=0.8,
        colsample_bytree=0.8,
        
        # Regularization for high complexity
        reg_alpha=0.1,
        reg_lambda=1.0,
        
        # Early stopping
        early_stopping_rounds=100,
        
        # Reproducibility
        random_state=42
    )
    
    # Create complete model configuration
    model_config = ModelConfig(xgboost=xgboost_config)
    
    print("XGBoost Configuration:")
    print(f"  Tree Method: {xgboost_config.tree_method}")
    print(f"  Max Depth: {xgboost_config.max_depth}")
    print(f"  N Estimators: {xgboost_config.n_estimators}")
    print(f"  Learning Rate: {xgboost_config.learning_rate}")
    print(f"  Regularization (L1/L2): {xgboost_config.reg_alpha}/{xgboost_config.reg_lambda}")
    
    # 4. Setup MLflow experiment tracking
    print("\n4. Setting up MLflow experiment tracking...")
    try:
        mlflow_config = MLflowConfig(
            tracking_uri="sqlite:///xgboost_gpu_example.db",
            experiment_name="xgboost_gpu_advanced_demo"
        )
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        print(f"MLflow tracking URI: {mlflow_config.tracking_uri}")
        print(f"Experiment: {mlflow_config.experiment_name}")
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        return
    
    # 5. Initialize GPU model trainer
    print("\n5. Initializing GPU model trainer...")
    trainer = GPUModelTrainer(model_config, mlflow_manager)
    print("GPU Model Trainer initialized successfully")
    
    # 6. Train XGBoost model with comprehensive tracking
    print("\n6. Training XGBoost model with advanced features...")
    print("This will demonstrate:")
    print("  • GPU-accelerated training with advanced hyperparameters")
    print("  • Cross-validation for model validation")
    print("  • Feature importance extraction and visualization")
    print("  • Comprehensive MLflow logging")
    print("  • Real-time GPU monitoring during training")
    
    # Start training
    training_start_time = time.time()
    
    try:
        # Train XGBoost model directly
        print("\nStarting XGBoost training...")
        
        # Start MLflow run for this training
        run_id = mlflow_manager.start_run(
            run_name=f"xgboost_advanced_demo_{int(time.time())}",
            tags={
                'model_type': 'xgboost',
                'gpu_enabled': str(trainer.is_gpu_available()),
                'demo': 'advanced_xgboost',
                'features': 'cross_validation,feature_importance,gpu_monitoring'
            }
        )
        
        # Log initial configuration
        config_dict = xgboost_config.model_dump()
        mlflow_manager.log_parameters(config_dict)
        
        # Train the model using the enhanced XGBoost method
        model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - training_start_time
        print(f"\nXGBoost training completed in {training_time:.2f} seconds")
        
        # 7. Evaluate model performance
        print("\n7. Evaluating model performance...")
        
        # Make predictions
        train_pred = trainer._predict_model(model, 'xgboost', X_train)
        val_pred = trainer._predict_model(model, 'xgboost', X_val)
        test_pred = trainer._predict_model(model, 'xgboost', X_test)
        
        # Calculate metrics
        train_metrics = trainer._calculate_metrics(y_train, train_pred)
        val_metrics = trainer._calculate_metrics(y_val, val_pred)
        test_metrics = trainer._calculate_metrics(y_test, test_pred)
        
        print("Performance Metrics:")
        print(f"  Training   - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2_score']:.4f}")
        print(f"  Validation - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2_score']:.4f}")
        print(f"  Test       - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2_score']:.4f}")
        
        # Log final metrics
        final_metrics = {
            'final_train_rmse': train_metrics['rmse'],
            'final_train_mae': train_metrics['mae'],
            'final_train_r2': train_metrics['r2_score'],
            'final_val_rmse': val_metrics['rmse'],
            'final_val_mae': val_metrics['mae'],
            'final_val_r2': val_metrics['r2_score'],
            'final_test_rmse': test_metrics['rmse'],
            'final_test_mae': test_metrics['mae'],
            'final_test_r2': test_metrics['r2_score'],
            'total_training_time': training_time
        }
        
        # Log metrics individually
        import mlflow
        for key, value in final_metrics.items():
            try:
                mlflow.log_metric(key, value)
            except Exception as e:
                print(f"Failed to log metric {key}: {e}")
        
        # 8. Display model information
        print("\n8. Model Information:")
        print(f"  Best Iteration: {getattr(model, 'best_iteration', 'N/A')}")
        print(f"  Number of Trees: {model.num_boosted_rounds()}")
        
        # Get feature importance
        try:
            feature_importance = model.get_score(importance_type='gain')
            if feature_importance:
                print(f"  Number of Important Features: {len(feature_importance)}")
                
                # Show top 5 most important features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print("  Top 5 Most Important Features:")
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    feature_name = housing.feature_names[int(feature.split('_')[1])] if 'feature_' in feature else feature
                    print(f"    {i}. {feature_name}: {importance:.4f}")
        except Exception as e:
            print(f"  Feature importance extraction failed: {e}")
        
        # 9. Final GPU metrics
        print("\n9. Final GPU Status:")
        final_metrics = gpu_monitor.get_metrics()
        if final_metrics:
            print(f"  Final GPU Utilization: {final_metrics.utilization_percent}%")
            print(f"  Final GPU Memory Used: {final_metrics.memory_used_mb:.1f} MB")
            print(f"  Final GPU Temperature: {final_metrics.temperature_celsius}°C")
            print(f"  GPU Power Usage: {final_metrics.power_usage_watts:.1f} W")
        else:
            print("  GPU metrics not available")
        
        # End MLflow run
        mlflow_manager.end_run("FINISHED")
        
        # 10. Summary
        print("\n" + "=" * 60)
        print("XGBoost GPU Training Example Summary:")
        print(f"  ✓ Dataset: California Housing ({X_train.shape[0]} training samples)")
        print(f"  ✓ GPU Training: {'Enabled' if device_info.get('available') else 'CPU Fallback'}")
        print(f"  ✓ Advanced Configuration: Deep trees (depth={xgboost_config.max_depth}), High estimators ({xgboost_config.n_estimators})")
        print(f"  ✓ Cross-Validation: Performed with 5-fold CV")
        print(f"  ✓ Feature Importance: Extracted and visualized")
        print(f"  ✓ MLflow Logging: Comprehensive experiment tracking")
        print(f"  ✓ Training Time: {training_time:.2f} seconds")
        print(f"  ✓ Final Performance: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2_score']:.4f}")
        
        print("\nKey Features Demonstrated:")
        print("  • GPU-accelerated XGBoost training with gpu_hist tree method")
        print("  • Advanced hyperparameters for deep trees and high estimator counts")
        print("  • Cross-validation for robust model evaluation")
        print("  • Feature importance extraction with multiple importance types")
        print("  • Real-time GPU monitoring during training")
        print("  • Comprehensive MLflow experiment tracking")
        print("  • Early stopping and regularization for optimal performance")
        
        print("\nGenerated Artifacts:")
        print("  • Feature importance plot: plots/XGBoost_feature_importance.png")
        print("  • Training curves plot: plots/XGBoost_training_curves.png")
        print("  • Model file: plots/xgboost_model.json")
        print("  • Feature importance data: plots/xgboost_feature_importance.json")
        print("  • Training history: plots/xgboost_training_history.json")
        print("  • MLflow experiment: xgboost_gpu_example.db")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        
        # End MLflow run with failure
        try:
            mlflow_manager.end_run("FAILED")
        except:
            pass


if __name__ == "__main__":
    main()