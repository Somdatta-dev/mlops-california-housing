#!/usr/bin/env python3
"""
MLflow Configuration and Experiment Management Example

This script demonstrates how to use the MLflow configuration and experiment
management utilities for the MLOps platform.
"""

import os
import sys
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlflow_config import (
    MLflowConfig,
    MLflowExperimentManager,
    ExperimentMetrics,
    ModelArtifacts,
    create_mlflow_manager
)


def main():
    """Demonstrate MLflow experiment tracking with California Housing dataset."""
    
    print("🚀 MLflow Experiment Tracking Demo")
    print("=" * 50)
    
    # 1. Create MLflow configuration
    print("\n1. Setting up MLflow configuration...")
    config = MLflowConfig(
        tracking_uri="sqlite:///mlflow_demo.db",  # Local SQLite database
        experiment_name="california-housing-demo"
    )
    print(f"   Tracking URI: {config.tracking_uri}")
    print(f"   Experiment: {config.experiment_name}")
    
    # 2. Initialize experiment manager
    print("\n2. Initializing experiment manager...")
    manager = create_mlflow_manager(config)
    
    if manager.fallback_mode:
        print("   ⚠️  Using fallback URI due to configuration issues")
    else:
        print("   ✅ MLflow setup successful")
    
    # 3. Load and prepare data
    print("\n3. Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # 4. Train model with experiment tracking
    print("\n4. Training model with MLflow tracking...")
    
    # Start MLflow run
    run_id = manager.start_run(
        run_name="linear-regression-demo",
        tags={
            "model_type": "linear_regression",
            "dataset": "california_housing",
            "demo": "true"
        }
    )
    print(f"   Started run: {run_id}")
    
    try:
        # Log parameters
        params = {
            "model_type": "LinearRegression",
            "test_size": 0.2,
            "random_state": 42,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0]
        }
        manager.log_parameters(params)
        print("   ✅ Parameters logged")
        
        # Train model
        start_time = time.time()
        model = LinearRegression()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   Model trained in {training_time:.2f} seconds")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Log metrics
        metrics = ExperimentMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            training_time=training_time,
            model_size_mb=sys.getsizeof(model) / (1024 * 1024)
        )
        manager.log_metrics(metrics)
        print("   ✅ Metrics logged")
        
        # Log model
        manager.log_model(model, "sklearn")
        print("   ✅ Model logged")
        
        # End run successfully
        manager.end_run("FINISHED")
        print("   ✅ Run completed successfully")
        
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        manager.end_run("FAILED")
        raise
    
    # 5. Demonstrate experiment querying
    print("\n5. Querying experiment results...")
    
    # Get all runs
    runs = manager.get_experiment_runs(max_results=10)
    print(f"   Total runs in experiment: {len(runs)}")
    
    if runs:
        latest_run = runs[0]
        print(f"   Latest run ID: {latest_run.info.run_id}")
        print(f"   Latest run status: {latest_run.info.status}")
        
        if latest_run.data.metrics:
            print("   Latest run metrics:")
            for metric_name, metric_value in latest_run.data.metrics.items():
                print(f"     {metric_name}: {metric_value:.4f}")
    
    # Get best run
    best_run = manager.get_best_run("rmse", ascending=True)
    if best_run:
        best_rmse = best_run.data.metrics.get("rmse", "N/A")
        print(f"   Best run RMSE: {best_rmse}")
    
    # 6. Demonstrate model registry (optional)
    print("\n6. Model registry operations...")
    try:
        if best_run:
            model_name = "california-housing-lr"
            version = manager.register_model(
                run_id=best_run.info.run_id,
                model_name=model_name,
                stage="Staging"
            )
            print(f"   ✅ Model registered: {model_name} v{version} (Staging)")
            
            # Load model from registry
            loaded_model = manager.load_model(model_name, "Staging")
            print("   ✅ Model loaded from registry")
            
            # Test loaded model
            test_prediction = loaded_model.predict(X_test[:1])
            print(f"   Test prediction: {test_prediction[0]:.4f}")
            
    except Exception as e:
        print(f"   ⚠️  Model registry operations failed: {e}")
    
    # 7. Cleanup demonstration
    print("\n7. Cleanup operations...")
    try:
        deleted_count = manager.cleanup_old_runs(keep_last_n=5)
        print(f"   Cleaned up {deleted_count} old runs")
    except Exception as e:
        print(f"   ⚠️  Cleanup failed: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print(f"MLflow tracking database: {config.tracking_uri}")
    print("You can explore the results using MLflow UI:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow_demo.db")


if __name__ == "__main__":
    main()