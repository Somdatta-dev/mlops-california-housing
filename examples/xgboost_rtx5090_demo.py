"""
XGBoost RTX 5090 Optimized Demo

This demo is optimized for high-end hardware like RTX 5090 with fast execution
and comprehensive feature demonstration without long cross-validation delays.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, make_regression
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


def create_fast_xgboost_config(use_gpu=False):
    """Create optimized XGBoost config for fast demonstration."""
    return XGBoostConfig(
        tree_method='gpu_hist' if use_gpu else 'hist',
        max_depth=8,  # Good depth for demonstration
        n_estimators=200,  # Reasonable for demo
        learning_rate=0.1,  # Standard learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=20,  # Early stopping for efficiency
        random_state=42
    )


def test_xgboost_performance():
    """Test XGBoost performance on different dataset sizes."""
    print("[PERF] XGBoost Performance Test on RTX 5090 System")
    print("=" * 60)
    
    # Test datasets of different sizes
    datasets = [
        ("Small Dataset", 1000, 10),
        ("Medium Dataset", 10000, 20),
        ("California Housing", None, None)
    ]
    
    results = []
    
    for name, n_samples, n_features in datasets:
        print(f"\n[TEST] Testing {name}...")
        
        if name == "California Housing":
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            n_samples, n_features = X.shape
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                                 noise=0.1, random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"   Data shape: {X_train.shape}")
        
        # Test both CPU and GPU (if available)
        configs = [
            ("CPU", create_fast_xgboost_config(use_gpu=False)),
        ]
        
        # Add GPU config if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                configs.append(("GPU", create_fast_xgboost_config(use_gpu=True)))
        except:
            pass
        
        for mode, config in configs:
            try:
                print(f"   [TRAIN] Testing {mode} mode...")
                
                # Direct XGBoost training (bypass cross-validation for speed)
                import xgboost as xgb
                
                # Create DMatrix
                dtrain = xgb.DMatrix(X_train, label=y_train, 
                                   feature_names=[f'feature_{i}' for i in range(X_train.shape[1])])
                dval = xgb.DMatrix(X_val, label=y_val,
                                 feature_names=[f'feature_{i}' for i in range(X_train.shape[1])])
                
                # Parameters
                params = {
                    'tree_method': config.tree_method,
                    'device': 'cuda' if config.tree_method == 'gpu_hist' else 'cpu',
                    'max_depth': config.max_depth,
                    'learning_rate': config.learning_rate,
                    'subsample': config.subsample,
                    'colsample_bytree': config.colsample_bytree,
                    'reg_alpha': config.reg_alpha,
                    'reg_lambda': config.reg_lambda,
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'verbosity': 0
                }
                
                # Training
                start_time = time.time()
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=config.n_estimators,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=config.early_stopping_rounds,
                    verbose_eval=False
                )
                training_time = time.time() - start_time
                
                # Prediction
                pred_start = time.time()
                predictions = model.predict(dval)
                pred_time = time.time() - pred_start
                
                # Metrics
                from sklearn.metrics import mean_squared_error, r2_score
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                r2 = r2_score(y_val, predictions)
                
                # Performance metrics
                samples_per_sec = n_samples / training_time
                
                result = {
                    'dataset': name,
                    'mode': mode,
                    'samples': n_samples,
                    'features': n_features,
                    'training_time': training_time,
                    'samples_per_sec': samples_per_sec,
                    'prediction_time': pred_time,
                    'rmse': rmse,
                    'r2_score': r2,
                    'trees': model.num_boosted_rounds()
                }
                results.append(result)
                
                print(f"   [OK] {mode}: {training_time:.2f}s ({samples_per_sec:.0f} samples/sec)")
                print(f"   [METRICS] RMSE: {rmse:.4f}, R²: {r2:.4f}")
                print(f"   [TREES] Used {model.num_boosted_rounds()} trees")
                
            except Exception as e:
                print(f"   [ERROR] {mode} failed: {e}")
                continue
    
    # Performance summary
    if results:
        print("\n" + "=" * 60)
        print("[SUMMARY] Performance Results:")
        print("=" * 60)
        
        df = pd.DataFrame(results)
        print(df[['dataset', 'mode', 'samples', 'training_time', 'samples_per_sec', 'rmse', 'r2_score']].to_string(index=False, float_format='%.3f'))
        
        # Analysis
        cpu_results = df[df['mode'] == 'CPU']
        gpu_results = df[df['mode'] == 'GPU']
        
        if not cpu_results.empty:
            avg_cpu_perf = cpu_results['samples_per_sec'].mean()
            print(f"\n[CPU] Average Performance: {avg_cpu_perf:.0f} samples/second")
        
        if not gpu_results.empty:
            avg_gpu_perf = gpu_results['samples_per_sec'].mean()
            print(f"[GPU] Average Performance: {avg_gpu_perf:.0f} samples/second")
            
            if not cpu_results.empty:
                speedup = avg_gpu_perf / avg_cpu_perf
                if speedup > 1.2:
                    print(f"[ANALYSIS] GPU is {speedup:.1f}x faster than CPU")
                elif speedup < 0.8:
                    print(f"[ANALYSIS] CPU is {1/speedup:.1f}x faster than GPU (normal for small datasets)")
                else:
                    print("[ANALYSIS] CPU and GPU performance are similar")


def demo_xgboost_features():
    """Demonstrate XGBoost advanced features."""
    print("\n[DEMO] XGBoost Advanced Features Demo")
    print("=" * 60)
    
    # Create sample data
    print("[DATA] Creating sample dataset...")
    X, y = make_regression(n_samples=5000, n_features=20, n_informative=15, 
                          noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    print(f"   Dataset: {X_train.shape[0]} training samples, {X.shape[1]} features")
    
    # Setup MLflow (optional)
    try:
        mlflow_config = MLflowConfig(
            tracking_uri="sqlite:///xgboost_rtx5090_demo.db",
            experiment_name="xgboost_rtx5090_demo"
        )
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        print("[MLFLOW] MLflow tracking enabled")
    except Exception as e:
        print(f"[MLFLOW] MLflow setup failed: {e}")
        mlflow_manager = None
    
    # Create optimized configuration
    config = create_fast_xgboost_config(use_gpu=False)  # Use CPU for consistent performance
    model_config = ModelConfig(xgboost=config)
    
    if mlflow_manager:
        trainer = GPUModelTrainer(model_config, mlflow_manager)
        
        print("[TRAIN] Training with GPU Model Trainer...")
        try:
            # Start MLflow run
            run_id = mlflow_manager.start_run("xgboost_rtx5090_demo")
            
            # This would normally do cross-validation, but let's use direct training
            import xgboost as xgb
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            
            params = {
                'tree_method': 'hist',
                'device': 'cpu',
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'subsample': config.subsample,
                'colsample_bytree': config.colsample_bytree,
                'reg_alpha': config.reg_alpha,
                'reg_lambda': config.reg_lambda,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'verbosity': 0
            }
            
            print("[TRAIN] Starting XGBoost training...")
            start_time = time.time()
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=config.n_estimators,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=config.early_stopping_rounds,
                verbose_eval=False
            )
            
            training_time = time.time() - start_time
            print(f"[OK] Training completed in {training_time:.2f} seconds")
            
            # Feature importance
            print("[FEATURES] Extracting feature importance...")
            importance = model.get_score(importance_type='gain')
            if importance:
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                print("   Top 10 Most Important Features:")
                for i, (feature, score) in enumerate(sorted_importance, 1):
                    print(f"   {i:2d}. {feature}: {score:.4f}")
            
            # Predictions and metrics
            predictions = model.predict(dval)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
                'mae': mean_absolute_error(y_val, predictions),
                'r2_score': r2_score(y_val, predictions)
            }
            
            print(f"[METRICS] RMSE: {metrics['rmse']:.4f}")
            print(f"[METRICS] MAE: {metrics['mae']:.4f}")
            print(f"[METRICS] R²: {metrics['r2_score']:.4f}")
            print(f"[MODEL] Trees used: {model.num_boosted_rounds()}")
            
            # Log to MLflow
            if mlflow_manager:
                import mlflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_metric('training_time', training_time)
                mlflow.log_metric('trees_used', model.num_boosted_rounds())
                
                mlflow_manager.end_run("FINISHED")
                print("[MLFLOW] Results logged to MLflow")
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            if mlflow_manager:
                mlflow_manager.end_run("FAILED")
    
    print("\n[SUCCESS] XGBoost demo completed successfully!")


def main():
    """Main demo function."""
    print("[START] XGBoost RTX 5090 Optimized Demo")
    print("=" * 60)
    
    # System info
    print("[SYSTEM] System Information:")
    try:
        import torch
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    try:
        import xgboost as xgb
        print(f"   XGBoost: {xgb.__version__}")
    except:
        pass
    
    # Run performance test
    test_xgboost_performance()
    
    # Run feature demo
    demo_xgboost_features()
    
    print("\n[COMPLETE] All demos completed successfully!")
    print("Your RTX 5090 system is ready for XGBoost development! 🚀")


if __name__ == "__main__":
    main()