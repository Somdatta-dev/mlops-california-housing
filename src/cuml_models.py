"""
cuML GPU-Accelerated Model Training Implementation

This module provides cuML-based Linear Regression and Random Forest training
with GPU acceleration, comprehensive evaluation metrics, and MLflow integration
for the MLOps platform.
"""

import os
import logging
import time
import json
import gc
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from pydantic import BaseModel, Field, field_validator

# GPU memory management
import torch

# Try to import cuML - handle gracefully if not available
try:
    import cuml
    from cuml.linear_model import LinearRegression as cuLinearRegression
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.metrics import mean_squared_error as cu_mse
    from cuml.metrics import mean_absolute_error as cu_mae
    from cuml.metrics import r2_score as cu_r2
    import cudf
    CUML_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("cuML is available - GPU acceleration enabled")
except ImportError as e:
    CUML_AVAILABLE = False
    cuml = None
    cudf = None
    logger = logging.getLogger(__name__)
    logger.warning(f"cuML not available: {e}. Falling back to CPU-based sklearn models")
    
    # Fallback imports
    from sklearn.linear_model import LinearRegression as cuLinearRegression
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor

# MLflow integration
from .mlflow_config import MLflowExperimentManager, ExperimentMetrics, ModelArtifacts
from .gpu_model_trainer_clean import GPUMemoryManager


@dataclass
class CuMLModelConfig:
    """Configuration for cuML models."""
    # Linear Regression config
    linear_regression: Dict[str, Any] = None
    
    # Random Forest config
    random_forest: Dict[str, Any] = None
    
    # General config
    use_gpu: bool = True
    random_state: int = 42
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        """Set default configurations."""
        if self.linear_regression is None:
            self.linear_regression = {
                'fit_intercept': True,
                'normalize': False,
                'algorithm': 'eig'  # 'eig', 'svd', 'cd' for cuML
            }
        
        if self.random_forest is None:
            self.random_forest = {
                'n_estimators': 100,
                'max_depth': 16,
                'max_features': 'sqrt',
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'n_streams': 4,  # cuML specific - number of parallel streams
                'split_criterion': 'mse',  # 'mse' or 'mae'
                'quantile_per_tree': False,
                'bootstrap_features': False
            }


@dataclass
class CuMLTrainingResults:
    """Results from cuML model training."""
    model_name: str
    model: Any
    training_time: float
    gpu_memory_used: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    cross_val_scores: Optional[Dict[str, float]]
    feature_importance: Optional[np.ndarray]
    predictions: Dict[str, np.ndarray]
    model_size_mb: float
    gpu_utilization: Optional[float]


class CuMLModelTrainer:
    """
    cuML-based model trainer with GPU acceleration and comprehensive evaluation.
    """
    
    def __init__(self, config: CuMLModelConfig, mlflow_manager: MLflowExperimentManager):
        """
        Initialize cuML model trainer.
        
        Args:
            config: cuML model configuration
            mlflow_manager: MLflow experiment manager
        """
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.gpu_available = CUML_AVAILABLE and torch.cuda.is_available()
        
        # Create plots directory
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"CuMLModelTrainer initialized. GPU available: {self.gpu_available}")
        
        if not self.gpu_available:
            logger.warning("GPU not available or cuML not installed. Using CPU fallback.")
    
    def _convert_to_cudf(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union['cudf.DataFrame', 'cudf.Series', np.ndarray]:
        """
        Convert pandas DataFrame/Series to cuDF for GPU processing.
        
        Args:
            data: Input data (pandas DataFrame, Series, or numpy array)
            
        Returns:
            cuDF DataFrame/Series if cuML available, otherwise original data
        """
        if not CUML_AVAILABLE or not self.config.use_gpu:
            return data
        
        try:
            if isinstance(data, pd.DataFrame):
                return cudf.from_pandas(data)
            elif isinstance(data, pd.Series):
                return cudf.from_pandas(data)
            else:
                return data
        except Exception as e:
            logger.warning(f"Failed to convert to cuDF: {e}. Using CPU data.")
            return data
    
    def _convert_from_cudf(self, data: Union['cudf.DataFrame', 'cudf.Series', np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Convert cuDF DataFrame/Series back to pandas.
        
        Args:
            data: cuDF data
            
        Returns:
            Pandas DataFrame/Series or numpy array
        """
        if not CUML_AVAILABLE:
            return data
        
        try:
            if hasattr(data, 'to_pandas'):
                return data.to_pandas()
            elif hasattr(data, 'to_numpy'):
                return data.to_numpy()
            else:
                return data
        except Exception as e:
            logger.warning(f"Failed to convert from cuDF: {e}")
            return data
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, use_gpu: bool = False) -> Dict[str, float]:
        """
        Calculate regression metrics using GPU if available.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            use_gpu: Whether to use GPU for metric calculation
            
        Returns:
            Dictionary of metrics
        """
        if CUML_AVAILABLE and use_gpu and self.config.use_gpu:
            try:
                # Convert to cuDF for GPU metrics calculation
                y_true_cu = cudf.Series(y_true) if not hasattr(y_true, 'to_pandas') else y_true
                y_pred_cu = cudf.Series(y_pred) if not hasattr(y_pred, 'to_pandas') else y_pred
                
                rmse = float(np.sqrt(cu_mse(y_true_cu, y_pred_cu)))
                mae = float(cu_mae(y_true_cu, y_pred_cu))
                r2 = float(cu_r2(y_true_cu, y_pred_cu))
                
                return {
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2
                }
            except Exception as e:
                logger.warning(f"GPU metrics calculation failed: {e}. Using CPU fallback.")
        
        # CPU fallback
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2_score': float(r2_score(y_true, y_pred))
        }
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_model_size(self, model: Any) -> float:
        """
        Estimate model size in MB.
        
        Args:
            model: Trained model
            
        Returns:
            Model size in MB
        """
        try:
            # For cuML models, estimate based on parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Rough estimation based on model type and parameters
                if hasattr(model, 'n_estimators'):
                    # Random Forest
                    n_estimators = params.get('n_estimators', 100)
                    max_depth = params.get('max_depth', 16)
                    # Rough estimate: each tree ~1KB per depth level
                    size_mb = (n_estimators * max_depth * 1024) / (1024 * 1024)
                else:
                    # Linear model - much smaller
                    size_mb = 0.1
                
                return max(size_mb, 0.01)  # Minimum 0.01 MB
            
            return 0.1  # Default estimate
            
        except Exception as e:
            logger.warning(f"Failed to estimate model size: {e}")
            return 0.1
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_val: pd.DataFrame, y_val: pd.Series,
                              X_test: Optional[pd.DataFrame] = None, 
                              y_test: Optional[pd.Series] = None) -> CuMLTrainingResults:
        """
        Train cuML Linear Regression with GPU acceleration.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets
            
        Returns:
            CuMLTrainingResults object
        """
        logger.info("Training cuML Linear Regression...")
        
        with GPUMemoryManager.gpu_memory_context():
            start_time = time.time()
            initial_memory = self._get_gpu_memory_usage()
            
            # Convert data to cuDF if using GPU
            X_train_cu = self._convert_to_cudf(X_train)
            y_train_cu = self._convert_to_cudf(y_train)
            X_val_cu = self._convert_to_cudf(X_val)
            y_val_cu = self._convert_to_cudf(y_val)
            
            # Initialize model
            model_params = self.config.linear_regression.copy()
            if CUML_AVAILABLE and self.config.use_gpu:
                model = cuLinearRegression(**model_params)
            else:
                # Remove cuML-specific parameters for sklearn fallback
                sklearn_params = {k: v for k, v in model_params.items() 
                                if k in ['fit_intercept']}  # 'normalize' deprecated in sklearn
                model = cuLinearRegression(**sklearn_params)
            
            # Train model
            model.fit(X_train_cu, y_train_cu)
            
            training_time = time.time() - start_time
            final_memory = self._get_gpu_memory_usage()
            gpu_memory_used = final_memory - initial_memory
            
            # Make predictions
            y_train_pred = model.predict(X_train_cu)
            y_val_pred = model.predict(X_val_cu)
            
            # Convert predictions back to numpy
            y_train_pred = self._convert_from_cudf(y_train_pred)
            y_val_pred = self._convert_from_cudf(y_val_pred)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train.values, y_train_pred, use_gpu=True)
            val_metrics = self._calculate_metrics(y_val.values, y_val_pred, use_gpu=True)
            
            # Test metrics if provided
            test_metrics = None
            y_test_pred = None
            if X_test is not None and y_test is not None:
                X_test_cu = self._convert_to_cudf(X_test)
                y_test_pred = model.predict(X_test_cu)
                y_test_pred = self._convert_from_cudf(y_test_pred)
                test_metrics = self._calculate_metrics(y_test.values, y_test_pred, use_gpu=True)
            
            # Cross-validation (using CPU for compatibility)
            cv_scores = None
            try:
                if not CUML_AVAILABLE:  # Only do CV for sklearn fallback
                    cv_rmse = cross_val_score(model, X_train, y_train, 
                                            cv=self.config.cross_validation_folds, 
                                            scoring='neg_mean_squared_error')
                    cv_scores = {
                        'cv_rmse_mean': float(np.sqrt(-cv_rmse.mean())),
                        'cv_rmse_std': float(np.sqrt(cv_rmse.std()))
                    }
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
            
            # Feature importance (coefficients for linear regression)
            feature_importance = None
            try:
                if hasattr(model, 'coef_'):
                    coef = model.coef_
                    if hasattr(coef, 'to_numpy'):
                        coef = coef.to_numpy()
                    feature_importance = np.abs(coef).flatten()
            except Exception as e:
                logger.warning(f"Failed to extract feature importance: {e}")
            
            # Prepare predictions dictionary
            predictions = {
                'train': y_train_pred,
                'val': y_val_pred
            }
            if y_test_pred is not None:
                predictions['test'] = y_test_pred
            
            # Calculate model size
            model_size_mb = self._get_model_size(model)
            
            results = CuMLTrainingResults(
                model_name="cuML_LinearRegression",
                model=model,
                training_time=training_time,
                gpu_memory_used=gpu_memory_used,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                cross_val_scores=cv_scores,
                feature_importance=feature_importance,
                predictions=predictions,
                model_size_mb=model_size_mb,
                gpu_utilization=None  # Would need nvidia-ml-py for this
            )
            
            logger.info(f"Linear Regression training completed in {training_time:.2f}s")
            logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2_score']:.4f}")
            
            return results
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series,
                           X_test: Optional[pd.DataFrame] = None, 
                           y_test: Optional[pd.Series] = None) -> CuMLTrainingResults:
        """
        Train cuML Random Forest with GPU acceleration.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets
            
        Returns:
            CuMLTrainingResults object
        """
        logger.info("Training cuML Random Forest...")
        
        with GPUMemoryManager.gpu_memory_context():
            start_time = time.time()
            initial_memory = self._get_gpu_memory_usage()
            
            # Convert data to cuDF if using GPU
            X_train_cu = self._convert_to_cudf(X_train)
            y_train_cu = self._convert_to_cudf(y_train)
            X_val_cu = self._convert_to_cudf(X_val)
            y_val_cu = self._convert_to_cudf(y_val)
            
            # Initialize model
            model_params = self.config.random_forest.copy()
            model_params['random_state'] = self.config.random_state
            
            if CUML_AVAILABLE and self.config.use_gpu:
                model = cuRandomForestRegressor(**model_params)
            else:
                # Remove cuML-specific parameters for sklearn fallback
                sklearn_params = {k: v for k, v in model_params.items() 
                                if k in ['n_estimators', 'max_depth', 'max_features', 
                                        'min_samples_split', 'min_samples_leaf', 
                                        'bootstrap', 'random_state']}
                model = cuRandomForestRegressor(**sklearn_params)
            
            # Train model
            model.fit(X_train_cu, y_train_cu)
            
            training_time = time.time() - start_time
            final_memory = self._get_gpu_memory_usage()
            gpu_memory_used = final_memory - initial_memory
            
            # Make predictions
            y_train_pred = model.predict(X_train_cu)
            y_val_pred = model.predict(X_val_cu)
            
            # Convert predictions back to numpy
            y_train_pred = self._convert_from_cudf(y_train_pred)
            y_val_pred = self._convert_from_cudf(y_val_pred)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train.values, y_train_pred, use_gpu=True)
            val_metrics = self._calculate_metrics(y_val.values, y_val_pred, use_gpu=True)
            
            # Test metrics if provided
            test_metrics = None
            y_test_pred = None
            if X_test is not None and y_test is not None:
                X_test_cu = self._convert_to_cudf(X_test)
                y_test_pred = model.predict(X_test_cu)
                y_test_pred = self._convert_from_cudf(y_test_pred)
                test_metrics = self._calculate_metrics(y_test.values, y_test_pred, use_gpu=True)
            
            # Cross-validation (using CPU for compatibility)
            cv_scores = None
            try:
                if not CUML_AVAILABLE:  # Only do CV for sklearn fallback
                    cv_rmse = cross_val_score(model, X_train, y_train, 
                                            cv=self.config.cross_validation_folds, 
                                            scoring='neg_mean_squared_error')
                    cv_scores = {
                        'cv_rmse_mean': float(np.sqrt(-cv_rmse.mean())),
                        'cv_rmse_std': float(np.sqrt(cv_rmse.std()))
                    }
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
            
            # Feature importance
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    if hasattr(importance, 'to_numpy'):
                        importance = importance.to_numpy()
                    feature_importance = importance.flatten()
            except Exception as e:
                logger.warning(f"Failed to extract feature importance: {e}")
            
            # Prepare predictions dictionary
            predictions = {
                'train': y_train_pred,
                'val': y_val_pred
            }
            if y_test_pred is not None:
                predictions['test'] = y_test_pred
            
            # Calculate model size
            model_size_mb = self._get_model_size(model)
            
            results = CuMLTrainingResults(
                model_name="cuML_RandomForest",
                model=model,
                training_time=training_time,
                gpu_memory_used=gpu_memory_used,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                cross_val_scores=cv_scores,
                feature_importance=feature_importance,
                predictions=predictions,
                model_size_mb=model_size_mb,
                gpu_utilization=None  # Would need nvidia-ml-py for this
            )
            
            logger.info(f"Random Forest training completed in {training_time:.2f}s")
            logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2_score']:.4f}")
            
            return results
    
    def create_feature_importance_plot(self, results: CuMLTrainingResults, 
                                     feature_names: List[str]) -> Optional[str]:
        """
        Create feature importance plot.
        
        Args:
            results: Training results containing feature importance
            feature_names: List of feature names
            
        Returns:
            Path to saved plot or None if failed
        """
        if results.feature_importance is None:
            logger.warning("No feature importance available for plotting")
            return None
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Sort features by importance
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(results.feature_importance)],
                'importance': results.feature_importance
            }).sort_values('importance', ascending=True)
            
            # Create horizontal bar plot
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{results.model_name} - Feature Importance')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f"{results.model_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Failed to create feature importance plot: {e}")
            plt.close()
            return None
    
    def create_prediction_plots(self, results: CuMLTrainingResults, 
                              y_train: pd.Series, y_val: pd.Series,
                              y_test: Optional[pd.Series] = None) -> Optional[str]:
        """
        Create prediction vs actual plots.
        
        Args:
            results: Training results containing predictions
            y_train: Training targets
            y_val: Validation targets
            y_test: Optional test targets
            
        Returns:
            Path to saved plot or None if failed
        """
        try:
            fig, axes = plt.subplots(1, 3 if y_test is not None else 2, figsize=(15, 5))
            if y_test is None:
                axes = [axes[0], axes[1]]
            
            # Training plot
            axes[0].scatter(y_train.values, results.predictions['train'], alpha=0.6)
            axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title(f'Training Set\nRMSE: {results.train_metrics["rmse"]:.4f}')
            
            # Validation plot
            axes[1].scatter(y_val.values, results.predictions['val'], alpha=0.6)
            axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            axes[1].set_xlabel('Actual Values')
            axes[1].set_ylabel('Predicted Values')
            axes[1].set_title(f'Validation Set\nRMSE: {results.val_metrics["rmse"]:.4f}')
            
            # Test plot if available
            if y_test is not None and 'test' in results.predictions:
                axes[2].scatter(y_test.values, results.predictions['test'], alpha=0.6)
                axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[2].set_xlabel('Actual Values')
                axes[2].set_ylabel('Predicted Values')
                axes[2].set_title(f'Test Set\nRMSE: {results.test_metrics["rmse"]:.4f}')
            
            plt.suptitle(f'{results.model_name} - Predictions vs Actual')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f"{results.model_name}_predictions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Prediction plots saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Failed to create prediction plots: {e}")
            plt.close()
            return None
    
    def log_to_mlflow(self, results: CuMLTrainingResults, feature_names: List[str],
                     run_name: Optional[str] = None) -> str:
        """
        Log training results to MLflow.
        
        Args:
            results: Training results to log
            feature_names: List of feature names
            run_name: Optional run name
            
        Returns:
            MLflow run ID
        """
        # Start MLflow run
        tags = {
            "model_type": results.model_name,
            "gpu_accelerated": str(CUML_AVAILABLE and self.config.use_gpu),
            "framework": "cuML" if CUML_AVAILABLE else "sklearn"
        }
        
        run_id = self.mlflow_manager.start_run(
            run_name=run_name or f"{results.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=tags
        )
        
        try:
            # Log parameters
            if results.model_name == "cuML_LinearRegression":
                params = self.config.linear_regression.copy()
            else:
                params = self.config.random_forest.copy()
            
            params.update({
                "use_gpu": self.config.use_gpu,
                "random_state": self.config.random_state,
                "cross_validation_folds": self.config.cross_validation_folds,
                "cuml_available": CUML_AVAILABLE
            })
            
            self.mlflow_manager.log_parameters(params)
            
            # Log metrics
            experiment_metrics = ExperimentMetrics(
                rmse=results.val_metrics['rmse'],
                mae=results.val_metrics['mae'],
                r2_score=results.val_metrics['r2_score'],
                training_time=results.training_time,
                gpu_utilization=results.gpu_utilization,
                gpu_memory_used=results.gpu_memory_used,
                model_size_mb=results.model_size_mb
            )
            
            self.mlflow_manager.log_metrics(experiment_metrics)
            
            # Log additional metrics
            for split, metrics in [('train', results.train_metrics), ('val', results.val_metrics)]:
                for metric_name, value in metrics.items():
                    self.mlflow_manager.client.log_metric(run_id, f"{split}_{metric_name}", value)
            
            if results.test_metrics:
                for metric_name, value in results.test_metrics.items():
                    self.mlflow_manager.client.log_metric(run_id, f"test_{metric_name}", value)
            
            if results.cross_val_scores:
                for metric_name, value in results.cross_val_scores.items():
                    self.mlflow_manager.client.log_metric(run_id, metric_name, value)
            
            # Create and log plots
            feature_importance_plot = self.create_feature_importance_plot(results, feature_names)
            prediction_plot = self.create_prediction_plots(results, 
                                                         pd.Series(results.predictions['train']),
                                                         pd.Series(results.predictions['val']))
            
            # Log artifacts
            artifacts = ModelArtifacts(
                model_path="model",  # MLflow will handle this
                feature_importance_plot=feature_importance_plot,
                training_curves_plot=prediction_plot
            )
            
            self.mlflow_manager.log_artifacts(artifacts)
            
            # Log model
            model_type = "sklearn"  # Both cuML and sklearn models can be logged as sklearn
            self.mlflow_manager.log_model(results.model, model_type)
            
            logger.info(f"Results logged to MLflow run: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
            self.mlflow_manager.end_run(status="FAILED")
            raise
        finally:
            self.mlflow_manager.end_run(status="FINISHED")
        
        return run_id
    
    def train_both_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_val: pd.DataFrame, y_val: pd.Series,
                         X_test: Optional[pd.DataFrame] = None, 
                         y_test: Optional[pd.Series] = None) -> Dict[str, CuMLTrainingResults]:
        """
        Train both Linear Regression and Random Forest models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets
            
        Returns:
            Dictionary of training results for both models
        """
        logger.info("Training both cuML models...")
        
        results = {}
        
        # Train Linear Regression
        try:
            lr_results = self.train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test)
            results['linear_regression'] = lr_results
            
            # Log to MLflow
            self.log_to_mlflow(lr_results, list(X_train.columns), "cuML_LinearRegression")
            
        except Exception as e:
            logger.error(f"Linear Regression training failed: {e}")
        
        # Train Random Forest
        try:
            rf_results = self.train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
            results['random_forest'] = rf_results
            
            # Log to MLflow
            self.log_to_mlflow(rf_results, list(X_train.columns), "cuML_RandomForest")
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
        
        # Compare models
        if len(results) > 1:
            self._compare_models(results)
        
        logger.info(f"Training completed for {len(results)} models")
        return results
    
    def _compare_models(self, results: Dict[str, CuMLTrainingResults]) -> None:
        """
        Compare model performance and log comparison.
        
        Args:
            results: Dictionary of training results
        """
        logger.info("Comparing model performance...")
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': result.model_name,
                'Val_RMSE': result.val_metrics['rmse'],
                'Val_MAE': result.val_metrics['mae'],
                'Val_R2': result.val_metrics['r2_score'],
                'Training_Time': result.training_time,
                'GPU_Memory_MB': result.gpu_memory_used * 1024,
                'Model_Size_MB': result.model_size_mb
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model by validation RMSE
        best_idx = comparison_df['Val_RMSE'].idxmin()
        best_model = comparison_df.iloc[best_idx]['Model']
        
        logger.info("Model Comparison:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        logger.info(f"\nBest model by validation RMSE: {best_model}")
        
        # Save comparison to file
        comparison_path = self.plots_dir / "cuml_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved: {comparison_path}")


def create_cuml_trainer(mlflow_manager: MLflowExperimentManager, 
                       config: Optional[CuMLModelConfig] = None) -> CuMLModelTrainer:
    """
    Factory function to create a cuML model trainer.
    
    Args:
        mlflow_manager: MLflow experiment manager
        config: Optional cuML configuration
        
    Returns:
        CuMLModelTrainer instance
    """
    if config is None:
        config = CuMLModelConfig()
    
    return CuMLModelTrainer(config, mlflow_manager)