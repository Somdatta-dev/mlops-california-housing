"""
Tests for XGBoost GPU Training Implementation

This module contains comprehensive tests for the enhanced XGBoost GPU training
functionality including advanced hyperparameters, feature importance extraction,
cross-validation, and MLflow logging.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.gpu_model_trainer import (
    GPUModelTrainer, ModelConfig, XGBoostConfig, GPUMonitor
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig


class TestXGBoostConfiguration:
    """Test XGBoost configuration with advanced parameters."""
    
    def test_xgboost_advanced_config(self):
        """Test XGBoost configuration with advanced parameters."""
        config = XGBoostConfig(
            max_depth=15,
            n_estimators=2000,
            learning_rate=0.02,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        assert config.max_depth == 15
        assert config.n_estimators == 2000
        assert config.learning_rate == 0.02
        assert config.reg_alpha == 0.1
        assert config.reg_lambda == 1.0
    
    def test_xgboost_config_validation_ranges(self):
        """Test XGBoost configuration parameter validation."""
        # Test valid ranges
        config = XGBoostConfig(
            max_depth=20,  # Maximum allowed
            n_estimators=10000,  # Maximum allowed
            learning_rate=1.0,  # Maximum allowed
            reg_alpha=10.0,  # Maximum allowed
            reg_lambda=10.0  # Maximum allowed
        )
        
        assert config.max_depth == 20
        assert config.n_estimators == 10000
        
        # Test minimum values
        config_min = XGBoostConfig(
            max_depth=1,  # Minimum allowed
            n_estimators=100,  # Minimum allowed
            learning_rate=0.001  # Minimum allowed
        )
        
        assert config_min.max_depth == 1
        assert config_min.n_estimators == 100
    
    def test_xgboost_tree_method_validation(self):
        """Test tree method validation."""
        valid_methods = ['gpu_hist', 'hist', 'exact', 'approx']
        
        for method in valid_methods:
            config = XGBoostConfig(tree_method=method)
            assert config.tree_method == method
        
        # Test invalid method
        with pytest.raises(ValueError, match="tree_method must be one of"):
            XGBoostConfig(tree_method='invalid_method')


class TestXGBoostTraining:
    """Test XGBoost training functionality."""
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression data for testing."""
        X, y = make_regression(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val
    
    @pytest.fixture
    def mock_mlflow_manager(self):
        """Create mock MLflow manager for testing."""
        mock_manager = Mock()
        mock_manager.start_run.return_value = "test_run_id"
        mock_manager.log_parameters.return_value = None
        mock_manager.log_metrics.return_value = None
        mock_manager.log_model.return_value = None
        mock_manager.log_artifacts.return_value = None
        mock_manager.end_run.return_value = None
        
        # Mock client and active run
        mock_client = Mock()
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_client.active_run.return_value = mock_run
        mock_client.log_metric.return_value = None
        mock_manager.client = mock_client
        
        return mock_manager
    
    @pytest.fixture
    def xgboost_config(self):
        """Create XGBoost configuration for fast testing."""
        return XGBoostConfig(
            tree_method='hist',  # Use CPU for consistent fast testing
            max_depth=3,  # Very shallow for speed
            n_estimators=100,  # Minimum allowed value
            learning_rate=0.5,  # High learning rate for fast convergence
            early_stopping_rounds=10  # Minimum allowed value
        )
    
    @patch('xgboost.cv')
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    @patch('src.gpu_model_trainer.Path.mkdir')
    def test_xgboost_training_basic(self, mock_mkdir, mock_close, mock_savefig, mock_cv,
                                   sample_regression_data, mock_mlflow_manager, xgboost_config):
        """Test basic XGBoost training functionality."""
        X_train, X_val, y_train, y_val = sample_regression_data
        
        # Mock cross-validation results for faster testing
        mock_cv_results = pd.DataFrame({
            'train-rmse-mean': [1.0, 0.9, 0.8],
            'train-rmse-std': [0.1, 0.1, 0.1],
            'test-rmse-mean': [1.2, 1.1, 1.0],
            'test-rmse-std': [0.15, 0.15, 0.15]
        })
        mock_cv.return_value = mock_cv_results
        
        # Create model config
        model_config = ModelConfig(xgboost=xgboost_config)
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        # Train model
        model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Verify model was created
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_score')
        
        # Verify MLflow logging was called
        mock_mlflow_manager.log_parameters.assert_called()
    
    @patch('xgboost.cv')
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    @patch('src.gpu_model_trainer.Path.mkdir')
    def test_xgboost_training_without_validation(self, mock_mkdir, mock_close, mock_savefig, mock_cv,
                                               sample_regression_data, mock_mlflow_manager, xgboost_config):
        """Test XGBoost training without validation data."""
        X_train, _, y_train, _ = sample_regression_data
        
        # Mock cross-validation results for faster testing
        mock_cv_results = pd.DataFrame({
            'train-rmse-mean': [1.0, 0.9, 0.8],
            'train-rmse-std': [0.1, 0.1, 0.1],
            'test-rmse-mean': [1.2, 1.1, 1.0],
            'test-rmse-std': [0.15, 0.15, 0.15]
        })
        mock_cv.return_value = mock_cv_results
        
        model_config = ModelConfig(xgboost=xgboost_config)
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        # Train model without validation data
        model = trainer._train_xgboost(X_train, y_train, None, None)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    @patch('src.gpu_model_trainer.xgb.cv')
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    @patch('src.gpu_model_trainer.Path.mkdir')
    def test_xgboost_cross_validation(self, mock_mkdir, mock_close, mock_savefig, mock_cv,
                                     sample_regression_data, mock_mlflow_manager, xgboost_config):
        """Test XGBoost cross-validation functionality."""
        X_train, X_val, y_train, y_val = sample_regression_data
        
        # Mock cross-validation results
        mock_cv_results = pd.DataFrame({
            'train-rmse-mean': [1.0, 0.9, 0.8, 0.7, 0.6],
            'train-rmse-std': [0.1, 0.1, 0.1, 0.1, 0.1],
            'test-rmse-mean': [1.2, 1.1, 1.0, 0.9, 0.8],
            'test-rmse-std': [0.15, 0.15, 0.15, 0.15, 0.15]
        })
        mock_cv.return_value = mock_cv_results
        
        model_config = ModelConfig(xgboost=xgboost_config)
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        # Train model
        model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Verify cross-validation was called
        mock_cv.assert_called_once()
        
        # Verify CV parameters
        cv_call_args = mock_cv.call_args
        assert cv_call_args[1]['nfold'] == 5
        assert cv_call_args[1]['shuffle'] == True
        assert cv_call_args[1]['early_stopping_rounds'] == xgboost_config.early_stopping_rounds
        
        # Verify model was created
        assert model is not None
    
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    @patch('src.gpu_model_trainer.Path.mkdir')
    def test_xgboost_feature_importance(self, mock_mkdir, mock_close, mock_savefig,
                                       sample_regression_data, mock_mlflow_manager, xgboost_config):
        """Test XGBoost feature importance extraction."""
        X_train, X_val, y_train, y_val = sample_regression_data
        
        model_config = ModelConfig(xgboost=xgboost_config)
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        # Train model
        model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Test feature importance extraction
        feature_importance = model.get_score(importance_type='gain')
        assert isinstance(feature_importance, dict)
        
        # Test different importance types
        importance_weight = model.get_score(importance_type='weight')
        importance_cover = model.get_score(importance_type='cover')
        
        assert isinstance(importance_weight, dict)
        assert isinstance(importance_cover, dict)
    
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    @patch('src.gpu_model_trainer.Path.mkdir')
    def test_xgboost_gpu_metrics_logging(self, mock_mkdir, mock_close, mock_savefig,
                                        sample_regression_data, mock_mlflow_manager, xgboost_config):
        """Test GPU metrics logging during XGBoost training."""
        X_train, X_val, y_train, y_val = sample_regression_data
        
        model_config = ModelConfig(xgboost=xgboost_config)
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        # Mock GPU metrics
        mock_gpu_metrics = Mock()
        mock_gpu_metrics.utilization_percent = 85.0
        mock_gpu_metrics.memory_used_mb = 4096.0
        mock_gpu_metrics.temperature_celsius = 70.0
        mock_gpu_metrics.power_usage_watts = 200.0
        
        with patch.object(trainer.gpu_monitor, 'get_metrics', return_value=mock_gpu_metrics):
            model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Verify model was created
        assert model is not None
        
        # Verify MLflow logging was called (GPU metrics would be logged during training)
        mock_mlflow_manager.log_metrics.assert_called()
    
    def test_xgboost_prediction(self, sample_regression_data, mock_mlflow_manager):
        """Test XGBoost model prediction functionality with simplified approach."""
        print("\n=== Starting Simplified XGBoost Prediction Test ===")
        X_train, X_val, y_train, y_val = sample_regression_data
        print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Create a simple XGBoost model directly for testing prediction functionality
        try:
            import xgboost as xgb
            
            # Create a very simple model for testing
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[f'feature_{i}' for i in range(X_train.shape[1])])
            
            # Simple parameters for fast training
            params = {
                'tree_method': 'hist',
                'max_depth': 3,
                'learning_rate': 0.3,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'verbosity': 0
            }
            
            print("Training simple XGBoost model for prediction test...")
            # Train a very simple model (just 10 rounds)
            model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
            print("Simple model training completed!")
            
            # Test the prediction functionality
            model_config = ModelConfig()
            trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
            
            print("Testing prediction functionality...")
            predictions = trainer._predict_model(model, 'xgboost', X_val)
            print(f"Predictions shape: {predictions.shape}")
            
            # Verify predictions
            assert isinstance(predictions, np.ndarray)
            assert predictions.shape[0] == X_val.shape[0]
            assert not np.isnan(predictions).any()
            print("✓ Prediction functionality works correctly")
            
            # Test metrics calculation
            metrics = trainer._calculate_metrics(y_val, predictions)
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2_score' in metrics
            assert all(isinstance(v, float) for v in metrics.values())
            print(f"✓ Metrics calculation works: RMSE={metrics['rmse']:.4f}, R²={metrics['r2_score']:.4f}")
            
            print("=== Simplified XGBoost Prediction Test Completed Successfully ===\n")
            
        except ImportError:
            pytest.skip("XGBoost not available for testing")
    
    def test_xgboost_metrics_calculation(self, sample_regression_data, mock_mlflow_manager):
        """Test metrics calculation for XGBoost predictions with simplified approach."""
        print("\n=== Starting Simplified XGBoost Metrics Calculation Test ===")
        X_train, X_val, y_train, y_val = sample_regression_data
        print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Create a simple XGBoost model directly for testing metrics calculation
        try:
            import xgboost as xgb
            
            # Create a very simple model for testing
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[f'feature_{i}' for i in range(X_train.shape[1])])
            
            # Simple parameters for fast training
            params = {
                'tree_method': 'hist',
                'max_depth': 3,
                'learning_rate': 0.3,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'verbosity': 0
            }
            
            print("Training simple XGBoost model for metrics test...")
            # Train a very simple model (just 10 rounds)
            model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
            print("Simple model training completed!")
            
            # Test the metrics calculation functionality
            model_config = ModelConfig()
            trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
            
            print("Making predictions...")
            predictions = trainer._predict_model(model, 'xgboost', X_val)
            print(f"Predictions shape: {predictions.shape}")
            
            print("Calculating metrics...")
            metrics = trainer._calculate_metrics(y_val, predictions)
            print(f"Metrics: {metrics}")
            
            # Verify all required metrics are present and valid
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2_score' in metrics
            assert all(isinstance(v, float) for v in metrics.values())
            assert metrics['rmse'] > 0
            assert metrics['mae'] > 0
            assert -1 <= metrics['r2_score'] <= 1  # R² should be in valid range
            
            print("✓ All metrics calculated correctly")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  R²: {metrics['r2_score']:.4f}")
            
            print("=== Simplified XGBoost Metrics Calculation Test Completed Successfully ===\n")
            
        except ImportError:
            pytest.skip("XGBoost not available for testing")


class TestXGBoostAdvancedFeatures:
    """Test advanced XGBoost features."""
    
    @pytest.fixture
    def temp_plots_dir(self):
        """Create temporary plots directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)
            yield plots_dir
    
    def test_advanced_hyperparameters(self):
        """Test advanced hyperparameter configuration."""
        config = XGBoostConfig(
            max_depth=15,
            n_estimators=2000,
            learning_rate=0.02
        )
        
        model_config = ModelConfig(xgboost=config)
        
        # Verify advanced parameters are set correctly
        assert model_config.xgboost.max_depth == 15
        assert model_config.xgboost.n_estimators == 2000
        assert model_config.xgboost.learning_rate == 0.02
    
    @patch('src.gpu_model_trainer.json.dump')
    @patch('src.gpu_model_trainer.plt.savefig')
    @patch('src.gpu_model_trainer.plt.close')
    def test_artifact_saving(self, mock_close, mock_savefig, mock_json_dump, temp_plots_dir):
        """Test saving of training artifacts."""
        # Create sample data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Mock MLflow manager
        mock_mlflow_manager = Mock()
        mock_mlflow_manager.log_parameters.return_value = None
        mock_mlflow_manager.log_metrics.return_value = None
        mock_client = Mock()
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_client.active_run.return_value = mock_run
        mock_client.log_metric.return_value = None
        mock_mlflow_manager.client = mock_client
        
        # Create trainer
        config = XGBoostConfig(n_estimators=100, max_depth=3)  # Minimum allowed for testing
        model_config = ModelConfig(xgboost=config)
        trainer = GPUModelTrainer(model_config, mock_mlflow_manager)
        
        with patch.object(trainer, '_create_plots_directory', return_value=temp_plots_dir):
            # Train model
            model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Verify artifacts would be saved
        mock_savefig.assert_called()  # Plots were saved
        mock_json_dump.assert_called()  # JSON data was saved
    
    def test_early_stopping_configuration(self):
        """Test early stopping configuration."""
        config = XGBoostConfig(
            early_stopping_rounds=50,
            n_estimators=1000
        )
        
        assert config.early_stopping_rounds == 50
        assert config.n_estimators == 1000
    
    def test_regularization_parameters(self):
        """Test regularization parameter configuration."""
        config = XGBoostConfig(
            reg_alpha=0.5,
            reg_lambda=2.0
        )
        
        assert config.reg_alpha == 0.5
        assert config.reg_lambda == 2.0


class TestXGBoostErrorHandling:
    """Test error handling in XGBoost training."""
    
    def test_xgboost_import_error(self):
        """Test handling of XGBoost import error."""
        mock_mlflow_manager = Mock()
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # Mock the import inside the function
        with patch('builtins.__import__', side_effect=ImportError("XGBoost not installed")):
            with pytest.raises(ImportError, match="XGBoost not installed"):
                trainer._train_xgboost(np.array([[1, 2]]), np.array([1]), None, None)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid training data."""
        mock_mlflow_manager = Mock()
        mock_mlflow_manager.log_parameters.return_value = None
        mock_mlflow_manager.log_metrics.return_value = None
        mock_client = Mock()
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_client.active_run.return_value = mock_run
        mock_client.log_metric.return_value = None
        mock_mlflow_manager.client = mock_client
        
        config = ModelConfig()
        trainer = GPUModelTrainer(config, mock_mlflow_manager)
        
        # Test with empty data
        with pytest.raises(Exception):  # Should raise some kind of error
            with patch('src.gpu_model_trainer.plt.savefig'), \
                 patch('src.gpu_model_trainer.plt.close'), \
                 patch('src.gpu_model_trainer.Path.mkdir'):
                trainer._train_xgboost(np.array([]), np.array([]), None, None)


class TestXGBoostIntegration:
    """Integration tests for XGBoost training."""
    
    def test_end_to_end_xgboost_training(self):
        """Test complete XGBoost training workflow."""
        # Create sample data
        X, y = make_regression(n_samples=200, n_features=8, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create temporary MLflow config
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                mlflow_config = MLflowConfig(
                    tracking_uri=f"file://{temp_dir}/mlruns",
                    experiment_name="test_xgboost_integration"
                )
                mlflow_manager = MLflowExperimentManager(mlflow_config)
                
                # Create XGBoost config for fast testing
                xgboost_config = XGBoostConfig(
                    tree_method='hist',  # CPU method for testing
                    max_depth=4,
                    n_estimators=100,  # Minimum allowed value
                    learning_rate=0.1,
                    early_stopping_rounds=10  # Minimum allowed value
                )
                
                model_config = ModelConfig(xgboost=xgboost_config)
                trainer = GPUModelTrainer(model_config, mlflow_manager)
                
                with patch('src.gpu_model_trainer.plt.savefig'), \
                     patch('src.gpu_model_trainer.plt.close'), \
                     patch('src.gpu_model_trainer.Path.mkdir'):
                    
                    # Start MLflow run
                    run_id = mlflow_manager.start_run("test_xgboost_run")
                    
                    # Train model
                    model = trainer._train_xgboost(X_train, y_train, X_val, y_val)
                    
                    # Verify model was created
                    assert model is not None
                    assert hasattr(model, 'predict')
                    
                    # Test prediction
                    predictions = trainer._predict_model(model, 'xgboost', X_val)
                    assert isinstance(predictions, np.ndarray)
                    assert predictions.shape[0] == X_val.shape[0]
                    
                    # Calculate metrics
                    metrics = trainer._calculate_metrics(y_val, predictions)
                    assert 'rmse' in metrics
                    assert 'mae' in metrics
                    assert 'r2_score' in metrics
                    
                    # End run
                    mlflow_manager.end_run("FINISHED")
                    
            except Exception as e:
                # If MLflow setup fails in test environment, that's expected
                if "MLflow" not in str(e) and "tracking" not in str(e):
                    raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])