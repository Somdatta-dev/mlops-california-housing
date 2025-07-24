"""
Unit tests for cuML model training and evaluation.

This module provides comprehensive tests for cuML-based Linear Regression
and Random Forest training with GPU acceleration and MLflow integration.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the modules to test
from src.cuml_models import (
    CuMLModelTrainer, CuMLModelConfig, CuMLTrainingResults,
    create_cuml_trainer, CUML_AVAILABLE
)
from src.mlflow_config import MLflowExperimentManager, MLflowConfig

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_data():
    """Create sample California Housing-like data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # Generate features similar to California Housing dataset
    X = pd.DataFrame({
        'MedInc': np.random.uniform(0.5, 15.0, n_samples),
        'HouseAge': np.random.uniform(1.0, 52.0, n_samples),
        'AveRooms': np.random.uniform(1.0, 20.0, n_samples),
        'AveBedrms': np.random.uniform(0.2, 5.0, n_samples),
        'Population': np.random.uniform(3.0, 35000.0, n_samples),
        'AveOccup': np.random.uniform(0.5, 1200.0, n_samples),
        'Latitude': np.random.uniform(32.5, 42.0, n_samples),
        'Longitude': np.random.uniform(-124.5, -114.0, n_samples)
    })
    
    # Generate target with some correlation to features
    y = (X['MedInc'] * 0.5 + 
         X['AveRooms'] * 0.1 + 
         np.random.normal(0, 0.5, n_samples))
    y = pd.Series(y, name='target')
    
    # Split data
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


@pytest.fixture
def mock_mlflow_manager():
    """Create a mock MLflow manager for testing."""
    mock_manager = Mock(spec=MLflowExperimentManager)
    mock_manager.start_run.return_value = "test_run_id"
    mock_manager.log_parameters.return_value = None
    mock_manager.log_metrics.return_value = None
    mock_manager.log_artifacts.return_value = None
    mock_manager.log_model.return_value = None
    mock_manager.end_run.return_value = None
    mock_manager.client = Mock()
    mock_manager.client.log_metric.return_value = None
    return mock_manager


@pytest.fixture
def temp_plots_dir():
    """Create a temporary directory for plots."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def cuml_config():
    """Create a test cuML configuration."""
    return CuMLModelConfig(
        use_gpu=True,
        random_state=42,
        cross_validation_folds=3,
        linear_regression={
            'fit_intercept': True,
            'normalize': False,
            'algorithm': 'eig'
        },
        random_forest={
            'n_estimators': 10,  # Small for testing
            'max_depth': 5,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'n_streams': 2,
            'split_criterion': 'mse'
        }
    )


class TestCuMLModelConfig:
    """Test cuML model configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = CuMLModelConfig()
        
        assert config.use_gpu is True
        assert config.random_state == 42
        assert config.cross_validation_folds == 5
        assert config.linear_regression is not None
        assert config.random_forest is not None
        
        # Check default linear regression config
        assert config.linear_regression['fit_intercept'] is True
        assert 'algorithm' in config.linear_regression
        
        # Check default random forest config
        assert config.random_forest['n_estimators'] == 100
        assert config.random_forest['max_depth'] == 16
        assert 'n_streams' in config.random_forest
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_lr_config = {'fit_intercept': False, 'algorithm': 'svd'}
        custom_rf_config = {'n_estimators': 50, 'max_depth': 10}
        
        config = CuMLModelConfig(
            use_gpu=False,
            random_state=123,
            cross_validation_folds=3,
            linear_regression=custom_lr_config,
            random_forest=custom_rf_config
        )
        
        assert config.use_gpu is False
        assert config.random_state == 123
        assert config.cross_validation_folds == 3
        assert config.linear_regression == custom_lr_config
        assert config.random_forest == custom_rf_config


class TestCuMLModelTrainer:
    """Test cuML model trainer functionality."""
    
    def test_trainer_initialization(self, mock_mlflow_manager, cuml_config):
        """Test trainer initialization."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        assert trainer.config == cuml_config
        assert trainer.mlflow_manager == mock_mlflow_manager
        assert trainer.plots_dir.exists()
        assert isinstance(trainer.gpu_available, bool)
    
    def test_convert_to_cudf_fallback(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test cuDF conversion with fallback."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Test with pandas DataFrame
        result = trainer._convert_to_cudf(sample_data['X_train'])
        assert isinstance(result, pd.DataFrame)
        
        # Test with pandas Series
        result = trainer._convert_to_cudf(sample_data['y_train'])
        assert isinstance(result, pd.Series)
        
        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = trainer._convert_to_cudf(arr)
        assert isinstance(result, np.ndarray)
    
    def test_calculate_metrics(self, mock_mlflow_manager, cuml_config):
        """Test metrics calculation."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Create test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2_score'] <= 1
    
    def test_get_model_size(self, mock_mlflow_manager, cuml_config):
        """Test model size estimation."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Mock model with parameters
        mock_model = Mock()
        mock_model.get_params.return_value = {'n_estimators': 100, 'max_depth': 10}
        mock_model.n_estimators = 100
        
        size = trainer._get_model_size(mock_model)
        assert isinstance(size, float)
        assert size > 0
    
    @patch('src.cuml_models.torch.cuda.is_available', return_value=True)
    @patch('src.cuml_models.torch.cuda.memory_allocated', return_value=1024**3)  # 1GB
    def test_get_gpu_memory_usage(self, mock_memory, mock_cuda, mock_mlflow_manager, cuml_config):
        """Test GPU memory usage calculation."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        memory_gb = trainer._get_gpu_memory_usage()
        assert isinstance(memory_gb, float)
        assert memory_gb == 1.0  # 1GB
    
    def test_linear_regression_training(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test Linear Regression training."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        results = trainer.train_linear_regression(
            sample_data['X_train'], sample_data['y_train'],
            sample_data['X_val'], sample_data['y_val'],
            sample_data['X_test'], sample_data['y_test']
        )
        
        # Verify results structure
        assert isinstance(results, CuMLTrainingResults)
        assert results.model_name == "cuML_LinearRegression"
        assert results.model is not None
        assert results.training_time > 0
        assert isinstance(results.gpu_memory_used, float)
        
        # Verify metrics
        assert 'rmse' in results.train_metrics
        assert 'mae' in results.train_metrics
        assert 'r2_score' in results.train_metrics
        assert 'rmse' in results.val_metrics
        assert 'mae' in results.val_metrics
        assert 'r2_score' in results.val_metrics
        
        # Verify predictions
        assert 'train' in results.predictions
        assert 'val' in results.predictions
        assert 'test' in results.predictions
        assert len(results.predictions['train']) == len(sample_data['y_train'])
        assert len(results.predictions['val']) == len(sample_data['y_val'])
        assert len(results.predictions['test']) == len(sample_data['y_test'])
        
        # Verify model size
        assert results.model_size_mb > 0
    
    def test_random_forest_training(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test Random Forest training."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        results = trainer.train_random_forest(
            sample_data['X_train'], sample_data['y_train'],
            sample_data['X_val'], sample_data['y_val'],
            sample_data['X_test'], sample_data['y_test']
        )
        
        # Verify results structure
        assert isinstance(results, CuMLTrainingResults)
        assert results.model_name == "cuML_RandomForest"
        assert results.model is not None
        assert results.training_time > 0
        
        # Verify metrics
        assert 'rmse' in results.train_metrics
        assert 'mae' in results.train_metrics
        assert 'r2_score' in results.train_metrics
        assert 'rmse' in results.val_metrics
        assert 'mae' in results.val_metrics
        assert 'r2_score' in results.val_metrics
        
        # Verify predictions
        assert 'train' in results.predictions
        assert 'val' in results.predictions
        assert 'test' in results.predictions
        
        # Verify feature importance (Random Forest should have this)
        if results.feature_importance is not None:
            assert len(results.feature_importance) == len(sample_data['X_train'].columns)
    
    def test_training_without_test_data(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test training without test data."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        results = trainer.train_linear_regression(
            sample_data['X_train'], sample_data['y_train'],
            sample_data['X_val'], sample_data['y_val']
        )
        
        assert results.test_metrics is None
        assert 'test' not in results.predictions
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_feature_importance_plot(self, mock_close, mock_savefig, 
                                          mock_mlflow_manager, cuml_config, sample_data):
        """Test feature importance plot creation."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Create mock results with feature importance
        results = CuMLTrainingResults(
            model_name="test_model",
            model=Mock(),
            training_time=1.0,
            gpu_memory_used=0.1,
            train_metrics={'rmse': 0.5, 'mae': 0.4, 'r2_score': 0.8},
            val_metrics={'rmse': 0.6, 'mae': 0.5, 'r2_score': 0.7},
            test_metrics=None,
            cross_val_scores=None,
            feature_importance=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            predictions={'train': np.array([1, 2, 3]), 'val': np.array([1, 2, 3])},
            model_size_mb=1.0,
            gpu_utilization=None
        )
        
        feature_names = list(sample_data['X_train'].columns)
        plot_path = trainer.create_feature_importance_plot(results, feature_names)
        
        assert plot_path is not None
        assert mock_savefig.called
        assert mock_close.called
    
    def test_create_feature_importance_plot_no_importance(self, mock_mlflow_manager, cuml_config):
        """Test feature importance plot creation when no importance available."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Create mock results without feature importance
        results = CuMLTrainingResults(
            model_name="test_model",
            model=Mock(),
            training_time=1.0,
            gpu_memory_used=0.1,
            train_metrics={'rmse': 0.5, 'mae': 0.4, 'r2_score': 0.8},
            val_metrics={'rmse': 0.6, 'mae': 0.5, 'r2_score': 0.7},
            test_metrics=None,
            cross_val_scores=None,
            feature_importance=None,
            predictions={'train': np.array([1, 2, 3]), 'val': np.array([1, 2, 3])},
            model_size_mb=1.0,
            gpu_utilization=None
        )
        
        plot_path = trainer.create_feature_importance_plot(results, ['feature1', 'feature2'])
        assert plot_path is None
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_prediction_plots(self, mock_close, mock_savefig, 
                                   mock_mlflow_manager, cuml_config, sample_data):
        """Test prediction plots creation."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Create mock results
        results = CuMLTrainingResults(
            model_name="test_model",
            model=Mock(),
            training_time=1.0,
            gpu_memory_used=0.1,
            train_metrics={'rmse': 0.5, 'mae': 0.4, 'r2_score': 0.8},
            val_metrics={'rmse': 0.6, 'mae': 0.5, 'r2_score': 0.7},
            test_metrics={'rmse': 0.7, 'mae': 0.6, 'r2_score': 0.6},
            cross_val_scores=None,
            feature_importance=None,
            predictions={
                'train': sample_data['y_train'].values,
                'val': sample_data['y_val'].values,
                'test': sample_data['y_test'].values
            },
            model_size_mb=1.0,
            gpu_utilization=None
        )
        
        plot_path = trainer.create_prediction_plots(
            results, sample_data['y_train'], sample_data['y_val'], sample_data['y_test']
        )
        
        assert plot_path is not None
        assert mock_savefig.called
        assert mock_close.called
    
    def test_log_to_mlflow(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test MLflow logging."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Create mock results
        results = CuMLTrainingResults(
            model_name="cuML_LinearRegression",
            model=Mock(),
            training_time=1.0,
            gpu_memory_used=0.1,
            train_metrics={'rmse': 0.5, 'mae': 0.4, 'r2_score': 0.8},
            val_metrics={'rmse': 0.6, 'mae': 0.5, 'r2_score': 0.7},
            test_metrics={'rmse': 0.7, 'mae': 0.6, 'r2_score': 0.6},
            cross_val_scores={'cv_rmse_mean': 0.65, 'cv_rmse_std': 0.05},
            feature_importance=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            predictions={
                'train': sample_data['y_train'].values,
                'val': sample_data['y_val'].values,
                'test': sample_data['y_test'].values
            },
            model_size_mb=1.0,
            gpu_utilization=80.0
        )
        
        feature_names = list(sample_data['X_train'].columns)
        
        with patch.object(trainer, 'create_feature_importance_plot', return_value="test_plot.png"):
            with patch.object(trainer, 'create_prediction_plots', return_value="pred_plot.png"):
                run_id = trainer.log_to_mlflow(results, feature_names)
        
        assert run_id == "test_run_id"
        assert mock_mlflow_manager.start_run.called
        assert mock_mlflow_manager.log_parameters.called
        assert mock_mlflow_manager.log_metrics.called
        assert mock_mlflow_manager.log_artifacts.called
        assert mock_mlflow_manager.log_model.called
        assert mock_mlflow_manager.end_run.called
    
    def test_train_both_models(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test training both models."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        with patch.object(trainer, 'log_to_mlflow', return_value="test_run_id"):
            results = trainer.train_both_models(
                sample_data['X_train'], sample_data['y_train'],
                sample_data['X_val'], sample_data['y_val'],
                sample_data['X_test'], sample_data['y_test']
            )
        
        assert 'linear_regression' in results
        assert 'random_forest' in results
        assert isinstance(results['linear_regression'], CuMLTrainingResults)
        assert isinstance(results['random_forest'], CuMLTrainingResults)
    
    def test_compare_models(self, mock_mlflow_manager, cuml_config, sample_data):
        """Test model comparison."""
        trainer = CuMLModelTrainer(cuml_config, mock_mlflow_manager)
        
        # Create mock results for comparison
        results = {
            'linear_regression': CuMLTrainingResults(
                model_name="cuML_LinearRegression",
                model=Mock(),
                training_time=1.0,
                gpu_memory_used=0.1,
                train_metrics={'rmse': 0.5, 'mae': 0.4, 'r2_score': 0.8},
                val_metrics={'rmse': 0.6, 'mae': 0.5, 'r2_score': 0.7},
                test_metrics=None,
                cross_val_scores=None,
                feature_importance=None,
                predictions={'train': np.array([1, 2, 3]), 'val': np.array([1, 2, 3])},
                model_size_mb=0.1,
                gpu_utilization=None
            ),
            'random_forest': CuMLTrainingResults(
                model_name="cuML_RandomForest",
                model=Mock(),
                training_time=2.0,
                gpu_memory_used=0.2,
                train_metrics={'rmse': 0.4, 'mae': 0.3, 'r2_score': 0.9},
                val_metrics={'rmse': 0.5, 'mae': 0.4, 'r2_score': 0.8},
                test_metrics=None,
                cross_val_scores=None,
                feature_importance=np.array([0.1, 0.2, 0.3, 0.4]),
                predictions={'train': np.array([1, 2, 3]), 'val': np.array([1, 2, 3])},
                model_size_mb=1.0,
                gpu_utilization=None
            )
        }
        
        # This should not raise an exception
        trainer._compare_models(results)
        
        # Check if comparison file was created
        comparison_file = trainer.plots_dir / "cuml_model_comparison.csv"
        assert comparison_file.exists()


class TestFactoryFunction:
    """Test factory function for creating cuML trainer."""
    
    def test_create_cuml_trainer_default_config(self, mock_mlflow_manager):
        """Test factory function with default config."""
        trainer = create_cuml_trainer(mock_mlflow_manager)
        
        assert isinstance(trainer, CuMLModelTrainer)
        assert isinstance(trainer.config, CuMLModelConfig)
        assert trainer.mlflow_manager == mock_mlflow_manager
    
    def test_create_cuml_trainer_custom_config(self, mock_mlflow_manager, cuml_config):
        """Test factory function with custom config."""
        trainer = create_cuml_trainer(mock_mlflow_manager, cuml_config)
        
        assert isinstance(trainer, CuMLModelTrainer)
        assert trainer.config == cuml_config
        assert trainer.mlflow_manager == mock_mlflow_manager


class TestIntegration:
    """Integration tests for cuML model training."""
    
    @pytest.mark.skipif(not CUML_AVAILABLE, reason="cuML not available")
    def test_cuml_integration(self, sample_data):
        """Test actual cuML integration if available."""
        # This test only runs if cuML is actually available
        from src.mlflow_config import MLflowConfig
        
        # Create a real MLflow manager with fallback URI
        mlflow_config = MLflowConfig(tracking_uri="sqlite:///:memory:")
        mlflow_manager = MLflowExperimentManager(mlflow_config)
        
        # Create trainer
        config = CuMLModelConfig(
            use_gpu=False,  # Use CPU for CI/CD compatibility
            random_state=42,
            cross_validation_folds=2,
            linear_regression={'fit_intercept': True},
            random_forest={'n_estimators': 5, 'max_depth': 3}
        )
        
        trainer = CuMLModelTrainer(config, mlflow_manager)
        
        # Test linear regression
        lr_results = trainer.train_linear_regression(
            sample_data['X_train'], sample_data['y_train'],
            sample_data['X_val'], sample_data['y_val']
        )
        
        assert lr_results.model is not None
        assert lr_results.val_metrics['rmse'] > 0
        
        # Test random forest
        rf_results = trainer.train_random_forest(
            sample_data['X_train'], sample_data['y_train'],
            sample_data['X_val'], sample_data['y_val']
        )
        
        assert rf_results.model is not None
        assert rf_results.val_metrics['rmse'] > 0
        assert rf_results.feature_importance is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])