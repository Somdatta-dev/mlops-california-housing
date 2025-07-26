#!/usr/bin/env python3
"""
Model Comparison and Selection System Example

This example demonstrates the comprehensive model comparison system that:
1. Trains all 5 GPU-accelerated models
2. Performs cross-validation evaluation
3. Conducts statistical significance testing
4. Selects the best model based on multiple criteria
5. Registers the best model in MLflow Model Registry
6. Creates comprehensive visualizations and reports

Usage:
    python examples/model_comparison_example.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_manager import DataManager, DataConfig
from mlflow_config import MLflowExperimentManager, MLflowConfig
from model_comparison import ModelComparisonSystem, ModelSelectionCriteria
from gpu_model_trainer import GPUModelTrainer, ModelConfig
from cuml_models import CuMLModelTrainer, CuMLModelConfig
from pytorch_neural_network import PyTorchNeuralNetworkTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating model comparison system."""
    logger.info("Starting Model Comparison and Selection System Example")
    
    try:
        # 1. Setup data and MLflow
        logger.info("Setting up data and MLflow configuration")
        
        # Data configuration
        data_config = DataConfig()
        data_manager = DataManager(data_config)
        
        # MLflow configuration
        mlflow_config = MLflowConfig(
            experiment_name="model-comparison-demo",
            tracking_uri="sqlite:///model_comparison_demo.db"
        )
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        
        # 2. Load and prepare data
        logger.info("Loading and preparing California Housing dataset")
        
        # Load data
        data = data_manager.load_california_housing()
        logger.info(f"Dataset shape: {data.shape}")
        
        # Prepare features and targets
        feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                          'Population', 'AveOccup', 'Latitude', 'Longitude']
        X = data[feature_columns].values
        y = data['target'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        # 3. Configure model selection criteria
        logger.info("Configuring model selection criteria")
        
        selection_criteria = ModelSelectionCriteria(
            primary_metric="rmse",
            secondary_metrics=["mae", "r2_score"],
            weights={
                "rmse": 0.4,
                "mae": 0.3,
                "r2_score": 0.2,
                "training_time": 0.1
            },
            minimize_metrics=["rmse", "mae", "training_time"],
            significance_threshold=0.05,
            cv_folds=5
        )
        
        # 4. Initialize model comparison system
        logger.info("Initializing model comparison system")
        
        comparison_system = ModelComparisonSystem(
            mlflow_manager=mlflow_manager,
            selection_criteria=selection_criteria
        )
        
        # 5. Train all models (optional - can be provided pre-trained)
        logger.info("Training all 5 GPU-accelerated models")
        
        trained_models = {}
        
        # Train cuML models
        logger.info("Training cuML models...")
        cuml_config = CuMLModelConfig()
        cuml_trainer = CuMLModelTrainer(cuml_config, mlflow_manager)
        
        # Start MLflow run for cuML Linear Regression
        run_id = mlflow_manager.start_run("cuML_LinearRegression_comparison")
        lr_result = cuml_trainer.train_linear_regression(X_train, y_train, X_val, y_val)
        mlflow_manager.end_run()
        
        trained_models['cuML_LinearRegression'] = {
            'model': lr_result.model,
            'training_time': lr_result.training_time,
            'model_type': 'cuml',
            'metrics': lr_result.metrics,
            'run_id': run_id
        }
        
        # Start MLflow run for cuML Random Forest
        run_id = mlflow_manager.start_run("cuML_RandomForest_comparison")
        rf_result = cuml_trainer.train_random_forest(X_train, y_train, X_val, y_val)
        mlflow_manager.end_run()
        
        trained_models['cuML_RandomForest'] = {
            'model': rf_result.model,
            'training_time': rf_result.training_time,
            'model_type': 'cuml',
            'metrics': rf_result.metrics,
            'run_id': run_id
        }
        
        # Train GPU models
        logger.info("Training GPU-accelerated models...")
        model_config = ModelConfig()
        gpu_trainer = GPUModelTrainer(model_config, mlflow_manager)
        
        # XGBoost
        run_id = mlflow_manager.start_run("XGBoost_comparison")
        xgb_model, xgb_time, xgb_metrics = gpu_trainer.train_xgboost_gpu(X_train, y_train, X_val, y_val)
        mlflow_manager.end_run()
        
        trained_models['XGBoost'] = {
            'model': xgb_model,
            'training_time': xgb_time,
            'model_type': 'xgboost',
            'metrics': xgb_metrics,
            'run_id': run_id
        }
        
        # LightGBM
        run_id = mlflow_manager.start_run("LightGBM_comparison")
        lgb_model, lgb_time, lgb_metrics = gpu_trainer.train_lightgbm_gpu(X_train, y_train, X_val, y_val)
        mlflow_manager.end_run()
        
        trained_models['LightGBM'] = {
            'model': lgb_model,
            'training_time': lgb_time,
            'model_type': 'lightgbm',
            'metrics': lgb_metrics,
            'run_id': run_id
        }
        
        # PyTorch Neural Network
        logger.info("Training PyTorch neural network...")
        run_id = mlflow_manager.start_run("PyTorch_comparison")
        pytorch_trainer = PyTorchNeuralNetworkTrainer(mlflow_manager)
        pytorch_result = pytorch_trainer.train(X_train, y_train, X_val, y_val)
        mlflow_manager.end_run()
        
        trained_models['PyTorch'] = {
            'model': pytorch_result['model'],
            'training_time': pytorch_result['training_time'],
            'model_type': 'pytorch',
            'metrics': pytorch_result['metrics'],
            'run_id': run_id
        }
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        
        # 6. Perform comprehensive model comparison
        logger.info("Performing comprehensive model comparison")
        
        comparison_result = comparison_system.compare_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            trained_models=trained_models
        )
        
        # 7. Display results
        logger.info("=" * 80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("=" * 80)
        
        print(f"\nBest Model: {comparison_result.best_model}")
        print(f"Selection Score: {comparison_result.comparison_summary['best_score']:.4f}")
        print(f"Comparison Timestamp: {comparison_result.timestamp}")
        
        print("\nAll Model Performance:")
        print("-" * 100)
        print(f"{'Model':<20} {'Type':<10} {'Val RMSE':<10} {'Val MAE':<10} {'Val R²':<10} {'Time (s)':<10} {'Size (MB)':<10}")
        print("-" * 100)
        
        for metrics in comparison_result.all_models:
            print(f"{metrics.model_name:<20} {metrics.model_type:<10} "
                  f"{metrics.val_rmse:<10.4f} {metrics.val_mae:<10.4f} "
                  f"{metrics.val_r2:<10.4f} {metrics.training_time:<10.1f} "
                  f"{metrics.model_size_mb:<10.2f}")
        
        print("\nBest Model Details:")
        print("-" * 50)
        best_metrics = comparison_result.best_model_metrics
        print(f"Model Name: {best_metrics.model_name}")
        print(f"Model Type: {best_metrics.model_type}")
        print(f"Validation RMSE: {best_metrics.val_rmse:.4f}")
        print(f"Validation MAE: {best_metrics.val_mae:.4f}")
        print(f"Validation R²: {best_metrics.val_r2:.4f}")
        if best_metrics.test_rmse:
            print(f"Test RMSE: {best_metrics.test_rmse:.4f}")
            print(f"Test MAE: {best_metrics.test_mae:.4f}")
            print(f"Test R²: {best_metrics.test_r2:.4f}")
        print(f"Training Time: {best_metrics.training_time:.1f} seconds")
        print(f"Model Size: {best_metrics.model_size_mb:.2f} MB")
        
        if best_metrics.cv_rmse_mean:
            print(f"Cross-validation RMSE: {best_metrics.cv_rmse_mean:.4f} ± {best_metrics.cv_rmse_std:.4f}")
            print(f"Cross-validation MAE: {best_metrics.cv_mae_mean:.4f} ± {best_metrics.cv_mae_std:.4f}")
            print(f"Cross-validation R²: {best_metrics.cv_r2_mean:.4f} ± {best_metrics.cv_r2_std:.4f}")
        
        # 8. Show selection criteria and weights
        print("\nSelection Criteria:")
        print("-" * 30)
        criteria = comparison_result.selection_criteria
        print(f"Primary Metric: {criteria['primary_metric']}")
        print(f"Weights: {criteria['weights']}")
        print(f"CV Folds: {criteria['cv_folds']}")
        
        # 9. Statistical significance results
        if comparison_result.statistical_tests:
            print("\nStatistical Significance Tests:")
            print("-" * 40)
            for test_name, results in comparison_result.statistical_tests.items():
                print(f"\n{test_name}:")
                for metric, p_value in results.items():
                    if 'p_value' in metric:
                        significance = "Significant" if p_value < 0.05 else "Not Significant"
                        print(f"  {metric}: {p_value:.4f} ({significance})")
        
        # 10. Export comprehensive report
        logger.info("Exporting comprehensive comparison report")
        
        comparison_system.export_comparison_report("model_comparison_report.html")
        
        # 11. Show generated files
        plots_dir = Path("plots")
        if plots_dir.exists():
            print(f"\nGenerated Files:")
            print("-" * 20)
            for file_path in plots_dir.glob("*comparison*"):
                print(f"  {file_path}")
            for file_path in plots_dir.glob("*selection*"):
                print(f"  {file_path}")
        
        if Path("model_comparison_report.html").exists():
            print(f"  model_comparison_report.html")
        
        logger.info("Model comparison and selection completed successfully!")
        
        # 12. Demonstrate model loading from registry
        logger.info("Demonstrating model loading from MLflow Model Registry")
        
        try:
            # Load the best model from registry
            best_model_from_registry = mlflow_manager.load_model(
                model_name="california-housing-best-model",
                stage="Staging"
            )
            
            # Make a test prediction
            test_prediction = best_model_from_registry.predict(X_test[:5])
            print(f"\nTest predictions from registered model: {test_prediction}")
            
        except Exception as e:
            logger.warning(f"Could not load model from registry: {e}")
        
        return comparison_result
        
    except Exception as e:
        logger.error(f"Error in model comparison example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    result = main()
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Best Model Selected: {result.best_model}")
    print("Check the generated plots and HTML report for detailed analysis.")