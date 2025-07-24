"""
RTX 5090 Performance Test for XGBoost GPU Training

This script tests the XGBoost implementation on high-end hardware
to ensure optimal performance with modern GPU configurations.
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append('.')

from src.gpu_model_trainer import (
    GPUModelTrainer, ModelConfig, XGBoostConfig, GPUMonitor
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig

def test_rtx5090_performance():
    """Test XGBoost performance on RTX 5090."""
    print("[PERF] RTX 5090 XGBoost Performance Test")
    print("=" * 50)
    
    # 1. GPU Detection
    print("\n1. GPU Detection:")
    gpu_monitor = GPUMonitor()
    device_info = gpu_monitor.get_device_info()
    
    if device_info.get('available'):
        print(f"[GPU] GPU: {device_info.get('name', 'Unknown')}")
        print(f"[GPU] Driver: {device_info.get('driver_version', 'Unknown')}")
        print(f"[GPU] CUDA: {device_info.get('cuda_version', 'Unknown')}")
        
        metrics = gpu_monitor.get_metrics()
        if metrics:
            print(f"[MEM] Memory: {metrics.memory_used_mb:.0f}/{metrics.memory_total_mb:.0f} MB")
            print(f"[TEMP] Temperature: {metrics.temperature_celsius}°C")
    else:
        print("[ERROR] GPU not available")
        return
    
    # 2. Dataset Preparation
    print("\n2. Dataset Preparation:")
    
    # Test with different dataset sizes
    datasets = [
        ("Small", 1000, 10),
        ("Medium", 10000, 20), 
        ("Large", 50000, 50),
        ("California Housing", None, None)  # Real dataset
    ]
    
    results = []
    
    for name, n_samples, n_features in datasets:
        print(f"\n[TEST] Testing with {name} dataset...")
        
        if name == "California Housing":
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            n_samples, n_features = X.shape
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                                 noise=0.1, random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"   Data shape: {X_train.shape}")
        
        # 3. GPU-Optimized Configuration
        gpu_config = XGBoostConfig(
            tree_method='gpu_hist',  # Use GPU
            max_depth=8,
            n_estimators=500,  # Reasonable for performance test
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50
        )
        
        # 4. Performance Test
        try:
            import xgboost as xgb
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Modern GPU parameters
            params = {
                'tree_method': 'hist',
                'device': 'cuda',  # Modern XGBoost 2.x GPU API
                'max_depth': gpu_config.max_depth,
                'learning_rate': gpu_config.learning_rate,
                'subsample': gpu_config.subsample,
                'colsample_bytree': gpu_config.colsample_bytree,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'verbosity': 0
            }
            
            print(f"   [TRAIN] Starting GPU training...")
            start_time = time.time()
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=gpu_config.n_estimators,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=gpu_config.early_stopping_rounds,
                verbose_eval=False
            )
            
            training_time = time.time() - start_time
            
            # Prediction test
            pred_start = time.time()
            predictions = model.predict(dval)
            pred_time = time.time() - pred_start
            
            # Calculate performance metrics
            samples_per_sec = n_samples / training_time
            pred_samples_per_sec = len(predictions) / pred_time if pred_time > 0 else float('inf')
            
            result = {
                'dataset': name,
                'samples': n_samples,
                'features': n_features,
                'training_time': training_time,
                'samples_per_sec': samples_per_sec,
                'prediction_time': pred_time,
                'pred_samples_per_sec': pred_samples_per_sec,
                'trees': model.num_boosted_rounds()
            }
            results.append(result)
            
            print(f"   [OK] Training: {training_time:.2f}s ({samples_per_sec:.0f} samples/sec)")
            print(f"   [OK] Prediction: {pred_time:.3f}s ({pred_samples_per_sec:.0f} samples/sec)")
            print(f"   [TREES] Trees: {model.num_boosted_rounds()}")
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            continue
    
    # 5. Performance Summary
    print("\n" + "=" * 50)
    print("[SUMMARY] RTX 5090 Performance Summary:")
    print("=" * 50)
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Performance analysis
        max_throughput = df['samples_per_sec'].max()
        avg_throughput = df['samples_per_sec'].mean()
        
        print(f"\n[PEAK] Peak Performance: {max_throughput:.0f} samples/second")
        print(f"[AVG] Average Performance: {avg_throughput:.0f} samples/second")
        
        # Expected performance for RTX 5090
        if max_throughput > 50000:
            print("[EXCELLENT] Your RTX 5090 is performing at expected levels!")
        elif max_throughput > 20000:
            print("[GOOD] Performance is solid, but could be optimized further")
        else:
            print("[SUBOPTIMAL] Performance is below expected for RTX 5090")
            print("   Consider checking CUDA installation and XGBoost GPU support")
    
    print("\n[TIPS] Optimization Tips for RTX 5090:")
    print("   • Use tree_method='hist' with device='cuda' (modern API)")
    print("   • Increase batch sizes for larger datasets")
    print("   • Use mixed precision if available")
    print("   • Monitor GPU utilization during training")

if __name__ == "__main__":
    test_rtx5090_performance()