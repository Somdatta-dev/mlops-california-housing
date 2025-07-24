"""
cuML Model Training Example

This example demonstrates how to use the cuML-based Linear Regression and Random Forest
training with GPU acceleration, comprehensive evaluation, and MLflow integration.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime

# Import our modules
try:
    from src.data_manager import DataManager, PreprocessingConfig
    from src.mlflow_config import MLflowConfig, create_mlflow_manager
    from src.cuml_models import CuMLModelTrainer, CuMLModelConfig, create_cuml_trainer
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_manager import DataManager, PreprocessingConfig
    from src.mlflow_config import MLflowConfig, create_mlflow_manager
    from src.cuml_models import CuMLModelTrainer, CuMLModelConfig, create_cuml_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function demonstrating cuML model training workflow.
    """
    logger.info("Starting cuML Model Training Example")
    logger.info("=" * 60)
    
    try:
        # Step 1: Setup data management
        logger.info("Step 1: Setting up data management...")
        
        data_config = PreprocessingConfig(
            test_size=0.2,
            random_state=42,
            scaler_type="standard",
            handle_outliers=True,
            feature_engineering=True,
            validation_split=0.2
        )
        
        data_manager = DataManager(config=data_config)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing California Housing data...")
        features_df, targets_series = data_manager.load_raw_data()
        
        # Validate data quality
        quality_report = data_manager.validate_data_quality(features_df, targets_series)
        logger.info(f"Data quality validation: {'PASSED' if quality_report.is_valid else 'FAILED'}")
        logger.info(f"Total samples: {quality_report.total_samples:,}")
        logger.info(f"Total features: {quality_report.total_features}")
        
        # Preprocess data
        processed_data = data_manager.preprocess_data(features_df, targets_series)
        
        logger.info("Data preprocessing completed:")
        logger.info(f"  Training samples: {processed_data['X_train'].shape[0]:,}")
        logger.info(f"  Validation samples: {processed_data['X_val'].shape[0]:,}")
        logger.info(f"  Test samples: {processed_data['X_test'].shape[0]:,}")
        logger.info(f"  Features: {processed_data['X_train'].shape[1]}")
        
        # Step 2: Setup MLflow experiment tracking
        logger.info("\nStep 2: Setting up MLflow experiment tracking...")
        
        mlflow_config = MLflowConfig(
            tracking_uri="sqlite:///mlflow_cuml_example.db",
            experiment_name="cuml-california-housing-example"
        )
        
        mlflow_manager = create_mlflow_manager(mlflow_config)
        logger.info(f"MLflow experiment setup complete. Experiment ID: {mlflow_manager.experiment_id}")
        
        # Step 3: Configure cuML models
        logger.info("\nStep 3: Configuring cuML models...")
        
        cuml_config = CuMLModelConfig(
            use_gpu=True,  # Will fallback to CPU if GPU/cuML not available
            random_state=42,
            cross_validation_folds=5,
            linear_regression={
                'fit_intercept': True,
                'normalize': False,
                'algorithm': 'eig'  # 'eig', 'svd', 'cd' for cuML
            },
            random_forest={
                'n_estimators': 100,
                'max_depth': 16,
                'max_features': 'sqrt',
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'n_streams': 4,  # cuML specific
                'split_criterion': 'mse',
                'quantile_per_tree': False,
                'bootstrap_features': False
            }
        )
        
        # Step 4: Initialize cuML trainer
        logger.info("\nStep 4: Initializing cuML trainer...")
        
        trainer = create_cuml_trainer(mlflow_manager, cuml_config)
        logger.info(f"cuML trainer initialized. GPU available: {trainer.gpu_available}")
        
        # Step 5: Train individual models
        logger.info("\nStep 5: Training individual models...")
        
        # Train Linear Regression
        logger.info("\n--- Training cuML Linear Regression ---")
        lr_results = trainer.train_linear_regression(
            processed_data['X_train'], processed_data['y_train'],
            processed_data['X_val'], processed_data['y_val'],
            processed_data['X_test'], processed_data['y_test']
        )
        
        logger.info("Linear Regression Results:")
        logger.info(f"  Training time: {lr_results.training_time:.2f} seconds")
        logger.info(f"  GPU memory used: {lr_results.gpu_memory_used:.3f} GB")
        logger.info(f"  Model size: {lr_results.model_size_mb:.2f} MB")
        logger.info(f"  Validation RMSE: {lr_results.val_metrics['rmse']:.4f}")
        logger.info(f"  Validation MAE: {lr_results.val_metrics['mae']:.4f}")
        logger.info(f"  Validation R²: {lr_results.val_metrics['r2_score']:.4f}")
        
        if lr_results.test_metrics:
            logger.info(f"  Test RMSE: {lr_results.test_metrics['rmse']:.4f}")
            logger.info(f"  Test R²: {lr_results.test_metrics['r2_score']:.4f}")
        
        # Log to MLflow
        lr_run_id = trainer.log_to_mlflow(lr_results, list(processed_data['X_train'].columns))
        logger.info(f"  MLflow run ID: {lr_run_id}")
        
        # Train Random Forest
        logger.info("\n--- Training cuML Random Forest ---")
        rf_results = trainer.train_random_forest(
            processed_data['X_train'], processed_data['y_train'],
            processed_data['X_val'], processed_data['y_val'],
            processed_data['X_test'], processed_data['y_test']
        )
        
        logger.info("Random Forest Results:")
        logger.info(f"  Training time: {rf_results.training_time:.2f} seconds")
        logger.info(f"  GPU memory used: {rf_results.gpu_memory_used:.3f} GB")
        logger.info(f"  Model size: {rf_results.model_size_mb:.2f} MB")
        logger.info(f"  Validation RMSE: {rf_results.val_metrics['rmse']:.4f}")
        logger.info(f"  Validation MAE: {rf_results.val_metrics['mae']:.4f}")
        logger.info(f"  Validation R²: {rf_results.val_metrics['r2_score']:.4f}")
        
        if rf_results.test_metrics:
            logger.info(f"  Test RMSE: {rf_results.test_metrics['rmse']:.4f}")
            logger.info(f"  Test R²: {rf_results.test_metrics['r2_score']:.4f}")
        
        # Log to MLflow
        rf_run_id = trainer.log_to_mlflow(rf_results, list(processed_data['X_train'].columns))
        logger.info(f"  MLflow run ID: {rf_run_id}")
        
        # Step 6: Train both models together and compare
        logger.info("\nStep 6: Training both models together for comparison...")
        
        all_results = trainer.train_both_models(
            processed_data['X_train'], processed_data['y_train'],
            processed_data['X_val'], processed_data['y_val'],
            processed_data['X_test'], processed_data['y_test']
        )
        
        # Step 7: Model comparison and analysis
        logger.info("\nStep 7: Model comparison and analysis...")
        
        comparison_data = []
        for model_name, results in all_results.items():
            comparison_data.append({
                'Model': results.model_name,
                'Training_Time_s': results.training_time,
                'GPU_Memory_GB': results.gpu_memory_used,
                'Model_Size_MB': results.model_size_mb,
                'Val_RMSE': results.val_metrics['rmse'],
                'Val_MAE': results.val_metrics['mae'],
                'Val_R2': results.val_metrics['r2_score'],
                'Test_RMSE': results.test_metrics['rmse'] if results.test_metrics else None,
                'Test_R2': results.test_metrics['r2_score'] if results.test_metrics else None
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("\nModel Comparison Summary:")
        logger.info("=" * 80)
        for _, row in comparison_df.iterrows():
            logger.info(f"{row['Model']}:")
            logger.info(f"  Training Time: {row['Training_Time_s']:.2f}s")
            logger.info(f"  GPU Memory: {row['GPU_Memory_GB']:.3f} GB")
            logger.info(f"  Model Size: {row['Model_Size_MB']:.2f} MB")
            logger.info(f"  Validation RMSE: {row['Val_RMSE']:.4f}")
            logger.info(f"  Validation R²: {row['Val_R2']:.4f}")
            if row['Test_RMSE'] is not None:
                logger.info(f"  Test RMSE: {row['Test_RMSE']:.4f}")
                logger.info(f"  Test R²: {row['Test_R2']:.4f}")
            logger.info("")
        
        # Find best model
        best_model_idx = comparison_df['Val_RMSE'].idxmin()
        best_model = comparison_df.iloc[best_model_idx]
        
        logger.info(f"Best Model (by validation RMSE): {best_model['Model']}")
        logger.info(f"Best Validation RMSE: {best_model['Val_RMSE']:.4f}")
        
        # Step 8: Feature importance analysis (for Random Forest)
        if 'random_forest' in all_results and all_results['random_forest'].feature_importance is not None:
            logger.info("\nStep 8: Feature importance analysis...")
            
            feature_names = list(processed_data['X_train'].columns)
            feature_importance = all_results['random_forest'].feature_importance
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(feature_importance)],
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            logger.info("Top 10 Most Important Features (Random Forest):")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                logger.info(f"  {i+1:2d}. {row['Feature']:<20} {row['Importance']:.4f}")
        
        # Step 9: Save results summary
        logger.info("\nStep 9: Saving results summary...")
        
        results_summary = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'data_samples': {
                    'train': processed_data['X_train'].shape[0],
                    'val': processed_data['X_val'].shape[0],
                    'test': processed_data['X_test'].shape[0]
                },
                'features': processed_data['X_train'].shape[1],
                'feature_names': list(processed_data['X_train'].columns)
            },
            'model_results': comparison_df.to_dict('records'),
            'best_model': {
                'name': best_model['Model'],
                'val_rmse': best_model['Val_RMSE'],
                'val_r2': best_model['Val_R2']
            }
        }
        
        # Save to JSON
        import json
        results_file = Path("cuml_training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Results summary saved to: {results_file}")
        
        # Save comparison CSV
        comparison_file = Path("cuml_model_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Model comparison saved to: {comparison_file}")
        
        logger.info("\n" + "=" * 60)
        logger.info("cuML Model Training Example completed successfully!")
        logger.info("=" * 60)
        
        # Print final summary
        logger.info("\nFINAL SUMMARY:")
        logger.info(f"✓ Trained {len(all_results)} cuML models")
        logger.info(f"✓ Best model: {best_model['Model']} (RMSE: {best_model['Val_RMSE']:.4f})")
        logger.info(f"✓ Results logged to MLflow experiment: {mlflow_manager.config.experiment_name}")
        logger.info(f"✓ Plots saved to: plots/")
        logger.info(f"✓ Summary files: {results_file}, {comparison_file}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error in cuML training example: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the example
    results = main()