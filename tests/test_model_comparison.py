#!/usr/bin/env python3
"""
Tests for Model Comparison and Selection System

This module contains comprehensive tests for the model comparison system,
including unit tests for individual components and integration tests for
the complete comparison workflow.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_comparison import (
    ModelComparisonSystem, ModelSelectionCriteria, ModelPerformanceMetrics,
    ModelComparisonResult
)
from mlflow_config import MLflowExperimentManager, MLflowConfig


class TestModelSelectionCriteria:
    """Test cases for ModelSelectionCriteria."""
    
    def test_default_criteria(self):
        """Test default selection criteria."""
        criteria = ModelSelectionCriteria()
        
        assert criteria.primary_metric == "rmse"
        assert "mae" in criteria.secondary_metrics
        assert "r2_score" in criteria.secondary_metrics
        assert criteria.weights["rmse"] == 0.4
        assert criteria.cv_folds == 5
        assert criteria.significance_threshold == 0.05
    
    def test_custom_criteria(self):
        """Test custom selection criteria."""
        criteria = ModelSelectionCriteria(
            primary_metric="mae",
            secondary_metrics=["rmse"],
            weights={"mae": 0.6, "rmse": 0.4},
            cv_folds=3,
            significance_threshold=0.01
        )
        
        assert criteria.primary_metric == "mae"
        assert criteria.secondary_metrics == ["rmse"]
        assert criteria.weights["mae"] == 0.6
        assert criteria.cv_folds == 3
        assert criteria.significance_threshold == 0.01
    
    def test_criteria_validation(self):
        """Test validation of selection criteria."""
        # Test invalid CV folds
        with pytest.raises(ValueError):
            ModelSelectionCriteria(cv_folds=2)  # Below minimum
        
        with pytest.raises(ValueError):
            ModelSelectionCriteria(cv_folds=15)  # Above maximum


class TestModelPerformanceMetrics:
    """Test cases for ModelPerformanceMetrics."""
    
    def test_metrics_creation(self):
        """Test creation of performance metrics."""
        metrics = ModelPerformanceMetrics(
            model_name="TestModel",
            model_type="test",
            train_rmse=0.5,
            val_rmse=0.6,
            test_rmse=0.65,
            train_mae=0.4,
            val_mae=0.45,
            test_mae=0.5,
            train_r2=0.8,
            val_r2=0.75,
            test_r2=0.7,
            training_time=120.5,
            model_size_mb=15.2
        )
        
        assert metrics.model_name == "TestModel"
        assert metrics.model_type == "test"
        assert metrics.val_rmse == 0.6
        assert metrics.training_time == 120.5
        assert metrics.model_size_mb == 15.2
    
    def test_metrics_to_dict(self):
        """Test conversion of metrics to dictionary."""
        metrics = ModelPerformanceMetrics(
            model_name="TestModel",
            model_type="test",
            train_rmse=0.5,
            val_rmse=0.6,
            test_rmse=None,
            train_mae=0.4,
            val_mae=0.45,
            test_mae=None,
            train_r2=0.8,
            val_r2=0.75,
            test_r2=None,
            training_time=120.5,
            model_size_mb=15.2
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["model_name"] == "TestModel"
        assert metrics_dict["val_rmse"] == 0.6
        assert metrics_dict["test_rmse"] is None


class TestModelComparisonSystem:
    """Test cases for ModelComparisonSystem."""
    
    @pytest.fixture
    def mock_mlflow_manager(self):
        """Create a mock MLflow manager."""
        mock_manager = Mock(spec=MLflowExperimentManager)
        mock_manager.start_run.return_value = "test_run_id"
        mock_manager.end_run.return_value = None
        mock_manager.get_best_run.return_value = Mock()
        mock_manager.get_best_run.return_value.info.run_id = "best_run_id"
        mock_manager.register_model.return_value = "1"
        mock_manager.client = Mock()
        return mock_manager
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 8
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Split data
        train_size = int(0.6 * n_samples)
        val_size = int(0.2 * n_samples)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    @pytest.fixture
    def mock_trained_models(self):
        """Create mock trained models."""
        models = {}
        
        # Mock sklearn-like models
        for model_name in ["Model1", "Model2", "Model3"]:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.randn(200)  # Mock predictions
            
            models[model_name] = {
                'model': mock_model,
                'training_time': np.random.uniform(10, 100),
                'model_type': 'sklearn',
                'metrics': {
                    'rmse': np.random.uniform(0.5, 1.0),
                    'mae': np.random.uniform(0.3, 0.8),
                    'r2_score': np.random.uniform(0.6, 0.9)
                }
            }
        
        return models
    
    def test_initialization(self, mock_mlflow_manager):
        """Test initialization of comparison system."""
        criteria = ModelSelectionCriteria()
        system = ModelComparisonSystem(mock_mlflow_manager, criteria)
        
        assert system.mlflow_manager == mock_mlflow_manager
        assert system.criteria == criteria
        assert system.comparison_results is None
        assert isinstance(system.models_cache, dict)
    
    def test_predict_with_model_sklearn(self, mock_mlflow_manager):
        """Test prediction with sklearn-like models."""
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Mock sklearn model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])
        
        X = np.random.randn(3, 5)
        predictions = system._predict_with_model(mock_model, X, 'sklearn')
        
        assert predictions is not None
        assert len(predictions) == 3
        mock_model.predict.assert_called_once_with(X)
    
    @patch('torch.cuda.is_available')
    @patch('torch.FloatTensor')
    def test_predict_with_model_pytorch(self, mock_tensor, mock_cuda, mock_mlflow_manager):
        """Test prediction with PyTorch models."""
        mock_cuda.return_value = False  # Use CPU for testing
        
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Mock PyTorch model
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_output = Mock()
        mock_output.cpu.return_value.numpy.return_value.flatten.return_value = np.array([1.0, 2.0, 3.0])
        mock_model.return_value = mock_output
        
        # Mock tensor
        mock_tensor_instance = Mock()
        mock_tensor.return_value = mock_tensor_instance
        
        X = np.random.randn(3, 5)
        predictions = system._predict_with_model(mock_model, X, 'pytorch')
        
        assert predictions is not None
        assert len(predictions) == 3
        mock_model.eval.assert_called_once()
    
    def test_estimate_model_size_sklearn(self, mock_mlflow_manager):
        """Test model size estimation for sklearn models."""
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        mock_model = Mock()
        
        with patch('joblib.dump') as mock_dump, \
             patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 1024 * 1024  # 1 MB
            
            size_mb = system._estimate_model_size(mock_model, 'sklearn')
            
            assert size_mb == 1.0
            mock_dump.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_estimate_model_size_pytorch(self, mock_cuda, mock_mlflow_manager):
        """Test model size estimation for PyTorch models."""
        mock_cuda.return_value = False
        
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Mock PyTorch model with parameters
        mock_model = Mock()
        mock_param = Mock()
        mock_param.nelement.return_value = 1000
        mock_param.element_size.return_value = 4  # 4 bytes per float
        mock_model.parameters.return_value = [mock_param]
        mock_model.buffers.return_value = []
        
        size_mb = system._estimate_model_size(mock_model, 'pytorch')
        
        expected_size = (1000 * 4) / (1024 * 1024)  # Convert to MB
        assert abs(size_mb - expected_size) < 1e-6
    
    def test_evaluate_model_comprehensive(self, mock_mlflow_manager, sample_data):
        """Test comprehensive model evaluation."""
        X_train, y_train, X_val, y_val, X_test, y_test = sample_data
        
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.side_effect = [
            y_train + np.random.normal(0, 0.1, len(y_train)),  # Train predictions
            y_val + np.random.normal(0, 0.1, len(y_val)),      # Val predictions
            y_test + np.random.normal(0, 0.1, len(y_test))     # Test predictions
        ]
        
        model_data = {
            'model': mock_model,
            'training_time': 45.5,
            'model_type': 'sklearn',
            'metrics': {}
        }
        
        with patch.object(system, '_perform_cross_validation') as mock_cv, \
             patch.object(system, '_estimate_model_size') as mock_size:
            
            mock_cv.return_value = {
                'rmse_mean': 0.5, 'rmse_std': 0.05,
                'mae_mean': 0.4, 'mae_std': 0.04,
                'r2_mean': 0.8, 'r2_std': 0.02
            }
            mock_size.return_value = 2.5
            
            metrics = system._evaluate_model_comprehensive(
                "TestModel", model_data, X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            assert isinstance(metrics, ModelPerformanceMetrics)
            assert metrics.model_name == "TestModel"
            assert metrics.model_type == "sklearn"
            assert metrics.training_time == 45.5
            assert metrics.model_size_mb == 2.5
            assert metrics.cv_rmse_mean == 0.5
            assert metrics.cv_rmse_std == 0.05
    
    def test_perform_statistical_tests(self, mock_mlflow_manager):
        """Test statistical significance testing."""
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Create mock metrics
        metrics1 = ModelPerformanceMetrics(
            model_name="Model1", model_type="test",
            train_rmse=0.5, val_rmse=0.6, test_rmse=0.65,
            train_mae=0.4, val_mae=0.45, test_mae=0.5,
            train_r2=0.8, val_r2=0.75, test_r2=0.7,
            cv_rmse_mean=0.6, cv_mae_mean=0.45, cv_r2_mean=0.75,
            training_time=60, model_size_mb=10
        )
        
        metrics2 = ModelPerformanceMetrics(
            model_name="Model2", model_type="test",
            train_rmse=0.7, val_rmse=0.8, test_rmse=0.85,
            train_mae=0.6, val_mae=0.65, test_mae=0.7,
            train_r2=0.6, val_r2=0.55, test_r2=0.5,
            cv_rmse_mean=0.8, cv_mae_mean=0.65, cv_r2_mean=0.55,
            training_time=90, model_size_mb=15
        )
        
        model_metrics = [metrics1, metrics2]
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        statistical_tests = system._perform_statistical_tests(model_metrics, X, y)
        
        assert isinstance(statistical_tests, dict)
        assert "Model1_vs_Model2" in statistical_tests
        
        test_results = statistical_tests["Model1_vs_Model2"]
        assert "rmse_p_value" in test_results
        assert "mae_p_value" in test_results
        assert "r2_p_value" in test_results
    
    def test_select_best_model(self, mock_mlflow_manager):
        """Test best model selection."""
        criteria = ModelSelectionCriteria(
            primary_metric="rmse",
            weights={"rmse": 0.5, "mae": 0.3, "r2_score": 0.2}
        )
        system = ModelComparisonSystem(mock_mlflow_manager, criteria)
        
        # Create mock metrics (Model1 should be better)
        metrics1 = ModelPerformanceMetrics(
            model_name="Model1", model_type="test",
            train_rmse=0.4, val_rmse=0.5, test_rmse=0.55,  # Better RMSE
            train_mae=0.3, val_mae=0.35, test_mae=0.4,     # Better MAE
            train_r2=0.85, val_r2=0.8, test_r2=0.75,       # Better R²
            training_time=60, model_size_mb=10
        )
        
        metrics2 = ModelPerformanceMetrics(
            model_name="Model2", model_type="test",
            train_rmse=0.7, val_rmse=0.8, test_rmse=0.85,  # Worse RMSE
            train_mae=0.6, val_mae=0.65, test_mae=0.7,     # Worse MAE
            train_r2=0.6, val_r2=0.55, test_r2=0.5,        # Worse R²
            training_time=90, model_size_mb=15
        )
        
        model_metrics = [metrics1, metrics2]
        statistical_tests = {}
        
        best_model, selection_summary = system._select_best_model(model_metrics, statistical_tests)
        
        assert best_model.model_name == "Model1"
        assert isinstance(selection_summary, dict)
        assert "selection_method" in selection_summary
        assert "best_model" in selection_summary
        assert selection_summary["best_model"] == "Model1"
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_comparison_visualizations(self, mock_close, mock_savefig, mock_mlflow_manager):
        """Test creation of comparison visualizations."""
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Create mock comparison result
        metrics1 = ModelPerformanceMetrics(
            model_name="Model1", model_type="test",
            train_rmse=0.5, val_rmse=0.6, test_rmse=0.65,
            train_mae=0.4, val_mae=0.45, test_mae=0.5,
            train_r2=0.8, val_r2=0.75, test_r2=0.7,
            training_time=60, model_size_mb=10
        )
        
        comparison_result = ModelComparisonResult(
            best_model="Model1",
            best_model_metrics=metrics1,
            all_models=[metrics1],
            comparison_summary={"best_score": 0.85, "model_scores": {"Model1": 0.85}},
            statistical_tests={},
            selection_criteria={"primary_metric": "rmse", "weights": {"rmse": 0.5}},
            timestamp=datetime.now()
        )
        
        with patch('pathlib.Path.mkdir'):
            system._create_comparison_visualizations(comparison_result)
        
        # Check that savefig was called (plots were created)
        assert mock_savefig.call_count >= 4  # At least 4 plots should be created
        assert mock_close.call_count >= 4
    
    def test_export_comparison_report(self, mock_mlflow_manager, tmp_path):
        """Test export of comparison report."""
        system = ModelComparisonSystem(mock_mlflow_manager)
        
        # Create mock comparison result
        metrics1 = ModelPerformanceMetrics(
            model_name="Model1", model_type="test",
            train_rmse=0.5, val_rmse=0.6, test_rmse=0.65,
            train_mae=0.4, val_mae=0.45, test_mae=0.5,
            train_r2=0.8, val_r2=0.75, test_r2=0.7,
            training_time=60, model_size_mb=10
        )
        
        comparison_result = ModelComparisonResult(
            best_model="Model1",
            best_model_metrics=metrics1,
            all_models=[metrics1],
            comparison_summary={"best_score": 0.85, "model_scores": {"Model1": 0.85}},
            statistical_tests={},
            selection_criteria={
                "primary_metric": "rmse", 
                "weights": {"rmse": 0.5},
                "cv_folds": 5
            },
            timestamp=datetime.now()
        )
        
        system.comparison_results = comparison_result
        
        # Export to temporary file
        output_path = tmp_path / "test_report.html"
        system.export_comparison_report(str(output_path))
        
        # Check that file was created and contains expected content
        assert output_path.exists()
        content = output_path.read_text()
        assert "Model Comparison Report" in content
        assert "Model1" in content
        assert "0.85" in content  # Best score


class TestIntegration:
    """Integration tests for the complete model comparison workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mlflow_manager(self, temp_dir):
        """Create MLflow manager with temporary database."""
        config = MLflowConfig(
            tracking_uri=f"sqlite:///{temp_dir}/test_mlflow.db",
            experiment_name="test_comparison"
        )
        return MLflowExperimentManager(config)
    
    def test_complete_comparison_workflow(self, mlflow_manager):
        """Test the complete model comparison workflow."""
        # This is a simplified integration test
        # In practice, you would use real models and data
        
        system = ModelComparisonSystem(mlflow_manager)
        
        # Create simple mock data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(50, 5)
        y_val = np.random.randn(50)
        
        # Create mock trained models
        trained_models = {}
        for i, model_name in enumerate(["Model1", "Model2"]):
            mock_model = Mock()
            # Make Model1 consistently better
            noise_level = 0.1 if i == 0 else 0.3
            mock_model.predict.side_effect = lambda x, noise=noise_level: (
                np.random.randn(len(x)) * noise
            )
            
            trained_models[model_name] = {
                'model': mock_model,
                'training_time': 30 + i * 20,
                'model_type': 'sklearn',
                'metrics': {}
            }
        
        # Mock the cross-validation to avoid complexity
        with patch.object(system, '_perform_cross_validation') as mock_cv:
            mock_cv.return_value = {
                'rmse_mean': 0.5, 'rmse_std': 0.05,
                'mae_mean': 0.4, 'mae_std': 0.04,
                'r2_mean': 0.8, 'r2_std': 0.02
            }
            
            # Run comparison
            result = system.compare_models(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                trained_models=trained_models
            )
        
        # Verify results
        assert isinstance(result, ModelComparisonResult)
        assert result.best_model in ["Model1", "Model2"]
        assert len(result.all_models) == 2
        assert isinstance(result.comparison_summary, dict)
        assert isinstance(result.timestamp, datetime)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])