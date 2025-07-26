"""
Tests for PyTorch Neural Network with Mixed Precision Training

This module contains comprehensive tests for the PyTorch neural network implementation
including architecture, training, mixed precision, early stopping, and integration tests.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import json
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pytorch_neural_network import (
    PyTorchNeuralNetworkTrainer, HousingNeuralNetwork, 
    CaliforniaHousingDataset, EarlyStopping, TrainingMetrics
)
from src.gpu_model_trainer_clean import GPUModelTrainer, ModelConfig, PyTorchConfig
from src.mlflow_config import MLflowExperimentManager, MLflowConfig


class TestCaliforniaHousingDataset(unittest.TestCase):
    """Test cases for CaliforniaHousingDataset."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.random.randn(100, 8).astype(np.float32)
        self.y = np.random.randn(100).astype(np.float32)
        self.dataset = CaliforniaHousingDataset(self.X, self.y)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        self.assertEqual(len(self.dataset), 100)
        self.assertEqual(self.dataset.X.shape, (100, 8))
        self.assertEqual(self.dataset.y.shape, (100, 1))
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        features, target = self.dataset[0]
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(features.shape, (8,))
        self.assertEqual(target.shape, (1,))
    
    def test_dataset_with_transform(self):
        """Test dataset with transform function."""
        def normalize_transform(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        dataset_with_transform = CaliforniaHousingDataset(
            self.X, self.y, transform=normalize_transform
        )
        
        features, _ = dataset_with_transform[0]
        self.assertIsInstance(features, torch.Tensor)


class TestHousingNeuralNetwork(unittest.TestCase):
    """Test cases for HousingNeuralNetwork."""
    
    def setUp(self):
        """Set up test model."""
        self.input_size = 8
        self.hidden_layers = [64, 32, 16]
        self.model = HousingNeuralNetwork(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            activation='relu',
            dropout_rate=0.2
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_layers, self.hidden_layers)
        self.assertEqual(self.model.activation_name, 'relu')
        self.assertEqual(self.model.dropout_rate, 0.2)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        batch_size = 10
        x = torch.randn(batch_size, self.input_size)
        
        with torch.no_grad():
            output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_model_info(self):
        """Test model info generation."""
        info = self.model.get_model_info()
        
        self.assertIn('architecture', info)
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        self.assertIn('model_size_mb', info)
        self.assertEqual(info['input_size'], self.input_size)
        self.assertEqual(info['hidden_layers'], self.hidden_layers)
        self.assertGreater(info['total_parameters'], 0)
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'tanh']
        
        for activation in activations:
            model = HousingNeuralNetwork(
                input_size=self.input_size,
                hidden_layers=[32, 16],
                activation=activation
            )
            
            x = torch.randn(5, self.input_size)
            with torch.no_grad():
                output = model(x)
            
            self.assertEqual(output.shape, (5, 1))
            self.assertFalse(torch.isnan(output).any())
    
    def test_batch_normalization(self):
        """Test model with batch normalization."""
        model = HousingNeuralNetwork(
            input_size=self.input_size,
            hidden_layers=[32, 16],
            use_batch_norm=True
        )
        
        x = torch.randn(10, self.input_size)
        output = model(x)
        
        self.assertEqual(output.shape, (10, 1))
        self.assertIsNotNone(model.batch_norms)
    
    def test_residual_connections(self):
        """Test model with residual connections."""
        model = HousingNeuralNetwork(
            input_size=self.input_size,
            hidden_layers=[self.input_size, self.input_size, 16],  # Same size for residual
            use_residual=True
        )
        
        x = torch.randn(5, self.input_size)
        output = model(x)
        
        self.assertEqual(output.shape, (5, 1))


class TestEarlyStopping(unittest.TestCase):
    """Test cases for EarlyStopping."""
    
    def setUp(self):
        """Set up early stopping."""
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        self.dummy_model = nn.Linear(10, 1)
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        self.assertEqual(self.early_stopping.patience, 3)
        self.assertEqual(self.early_stopping.min_delta, 0.01)
        self.assertFalse(self.early_stopping.early_stop)
        self.assertEqual(self.early_stopping.counter, 0)
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        # Simulate improving validation loss
        losses = [1.0, 0.8, 0.6, 0.4]
        
        for loss in losses:
            should_stop = self.early_stopping(loss, self.dummy_model)
            self.assertFalse(should_stop)
        
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertFalse(self.early_stopping.early_stop)
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        # Simulate no improvement in validation loss
        losses = [1.0, 1.1, 1.2, 1.3, 1.4]
        
        should_stop = False
        for i, loss in enumerate(losses):
            should_stop = self.early_stopping(loss, self.dummy_model)
            if i >= 3:  # After patience epochs
                self.assertTrue(should_stop)
                break
        
        self.assertTrue(should_stop)
        self.assertTrue(self.early_stopping.early_stop)


class TestTrainingMetrics(unittest.TestCase):
    """Test cases for TrainingMetrics."""
    
    def test_training_metrics_creation(self):
        """Test training metrics creation."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.5,
            val_loss=0.6,
            train_rmse=0.7,
            val_rmse=0.8,
            learning_rate=0.001,
            epoch_time=1.5
        )
        
        self.assertEqual(metrics.epoch, 10)
        self.assertEqual(metrics.train_loss, 0.5)
        self.assertEqual(metrics.val_loss, 0.6)
        self.assertEqual(metrics.learning_rate, 0.001)
    
    def test_training_metrics_to_dict(self):
        """Test training metrics conversion to dictionary."""
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.3,
            val_loss=0.4,
            learning_rate=0.001
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['epoch'], 5)
        self.assertEqual(metrics_dict['train_loss'], 0.3)
        self.assertEqual(metrics_dict['val_loss'], 0.4)


class TestPyTorchNeuralNetworkTrainer(unittest.TestCase):
    """Test cases for PyTorchNeuralNetworkTrainer."""
    
    def setUp(self):
        """Set up test data and trainer."""
        # Generate small test dataset
        np.random.seed(42)
        self.X_train = np.random.randn(100, 8).astype(np.float32)
        self.y_train = np.random.randn(100).astype(np.float32)
        self.X_val = np.random.randn(20, 8).astype(np.float32)
        self.y_val = np.random.randn(20).astype(np.float32)
        
        # Configuration for fast testing
        self.config = {
            'hidden_layers': [32, 16],
            'activation': 'relu',
            'dropout_rate': 0.1,
            'batch_size': 32,
            'epochs': 5,  # Small for testing
            'learning_rate': 0.01,
            'weight_decay': 1e-4,
            'device': 'cpu',  # Force CPU for testing
            'mixed_precision': False,  # Disable for CPU
            'early_stopping_patience': 3,
            'lr_scheduler': 'cosine',
            'warmup_epochs': 1,
            'optimizer': 'adamw'
        }
        
        self.trainer = PyTorchNeuralNetworkTrainer(self.config)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.device.type, 'cpu')
        self.assertFalse(self.trainer.use_mixed_precision)
        self.assertIsNone(self.trainer.scaler)
    
    def test_device_setup(self):
        """Test device setup."""
        # Test CPU device
        cpu_config = self.config.copy()
        cpu_config['device'] = 'cpu'
        cpu_trainer = PyTorchNeuralNetworkTrainer(cpu_config)
        self.assertEqual(cpu_trainer.device.type, 'cpu')
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            cuda_config = self.config.copy()
            cuda_config['device'] = 'cuda'
            cuda_trainer = PyTorchNeuralNetworkTrainer(cuda_config)
            self.assertEqual(cuda_trainer.device.type, 'cuda')
    
    def test_model_creation(self):
        """Test model creation."""
        model = self.trainer._create_model(8)
        
        self.assertIsInstance(model, HousingNeuralNetwork)
        self.assertEqual(model.input_size, 8)
        self.assertEqual(model.hidden_layers, [32, 16])
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        model = self.trainer._create_model(8)
        optimizer = self.trainer._create_optimizer(model)
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
    
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        model = self.trainer._create_model(8)
        optimizer = self.trainer._create_optimizer(model)
        scheduler = self.trainer._create_scheduler(optimizer, 10)
        
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_data_loader_creation(self):
        """Test data loader creation."""
        train_loader, val_loader = self.trainer._create_data_loaders(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(train_loader.batch_size, 32)
        self.assertEqual(val_loader.batch_size, 32)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = self.trainer._calculate_metrics(y_true, y_pred)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_training(self):
        """Test model training."""
        model = self.trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        self.assertIsInstance(model, HousingNeuralNetwork)
        self.assertGreater(len(self.trainer.training_history), 0)
        
        # Check that training history contains expected fields
        first_metric = self.trainer.training_history[0]
        self.assertIsInstance(first_metric, TrainingMetrics)
        self.assertEqual(first_metric.epoch, 1)
        self.assertIsNotNone(first_metric.train_loss)
    
    def test_prediction(self):
        """Test model prediction."""
        model = self.trainer.train(self.X_train, self.y_train)
        predictions = self.trainer.predict(model, self.X_val)
        
        self.assertEqual(predictions.shape, (len(self.X_val),))
        self.assertFalse(np.isnan(predictions).any())
        self.assertFalse(np.isinf(predictions).any())
    
    def test_training_curves_saving(self):
        """Test training curves saving."""
        # Train model first
        model = self.trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            curves_path = self.trainer.save_training_curves(tmp_path)
            
            self.assertEqual(curves_path, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            # Cleanup with error handling
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except PermissionError:
                pass  # Ignore permission errors on Windows
    
    def test_model_checkpoint_saving(self):
        """Test model checkpoint saving and loading."""
        # Train model first
        model = self.trainer.train(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = self.trainer.save_model_checkpoint(model, tmp.name)
            
            self.assertEqual(checkpoint_path, tmp.name)
            self.assertTrue(os.path.exists(tmp.name))
            
            # Test loading
            loaded_model = self.trainer.load_model_checkpoint(tmp.name, 8)
            self.assertIsInstance(loaded_model, HousingNeuralNetwork)
            
            # Cleanup
            os.unlink(tmp.name)


class TestGPUModelTrainerIntegration(unittest.TestCase):
    """Test cases for GPU model trainer integration."""
    
    def setUp(self):
        """Set up test data."""
        # Use California Housing dataset for realistic testing
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        # Use small subset for testing
        X_small = X[:200]
        y_small = y[:200]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_small, y_small, test_size=0.2, random_state=42
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def test_pytorch_config_integration(self):
        """Test PyTorch configuration integration."""
        pytorch_config = PyTorchConfig(
            hidden_layers=[64, 32],
            epochs=5,
            batch_size=32,
            device='cpu',
            mixed_precision=False
        )
        
        model_config = ModelConfig(pytorch=pytorch_config)
        
        self.assertEqual(model_config.pytorch.hidden_layers, [64, 32])
        self.assertEqual(model_config.pytorch.epochs, 5)
        self.assertEqual(model_config.pytorch.device, 'cpu')
    
    def test_gpu_trainer_pytorch_training(self):
        """Test PyTorch training through GPU trainer."""
        # Create configuration
        pytorch_config = PyTorchConfig(
            hidden_layers=[32, 16],
            epochs=3,  # Very small for testing
            batch_size=32,
            device='cpu',
            mixed_precision=False,
            early_stopping_patience=2
        )
        
        model_config = ModelConfig(pytorch=pytorch_config)
        
        # Create trainer without MLflow for testing
        gpu_trainer = GPUModelTrainer(model_config, None)
        
        # Test training
        trained_model, results = gpu_trainer.train_pytorch_neural_network(
            self.X_train_scaled, self.y_train,
            self.X_test_scaled, self.y_test
        )
        
        self.assertIsInstance(trained_model, HousingNeuralNetwork)
        self.assertIn('model_info', results)
        self.assertIn('train_metrics', results)
        self.assertIn('val_metrics', results)
        self.assertIn('training_time', results)
        
        # Test prediction
        predictions = gpu_trainer.predict_pytorch(trained_model, self.X_test_scaled)
        self.assertEqual(len(predictions), len(self.X_test_scaled))


class TestRealDataIntegration(unittest.TestCase):
    """Integration tests with real California Housing data."""
    
    def setUp(self):
        """Set up real dataset."""
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        # Use subset for faster testing
        X_subset = X[:500]
        y_subset = y[:500]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
    
    def test_end_to_end_training(self):
        """Test end-to-end training with real data."""
        config = {
            'hidden_layers': [64, 32, 16],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'batch_size': 64,
            'epochs': 10,
            'learning_rate': 0.001,
            'device': 'cpu',
            'mixed_precision': False,
            'early_stopping_patience': 5,
            'lr_scheduler': 'cosine',
            'optimizer': 'adamw'
        }
        
        trainer = PyTorchNeuralNetworkTrainer(config)
        model = trainer.train(self.X_train_scaled, self.y_train, 
                            self.X_test_scaled, self.y_test)
        
        # Test predictions
        predictions = trainer.predict(model, self.X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        
        # Basic sanity checks
        self.assertLess(rmse, 2.0)  # RMSE should be reasonable
        self.assertGreater(r2, 0.0)  # R² should be positive
        
        # Check training history
        self.assertGreater(len(trainer.training_history), 0)
        
        # Check that loss generally decreases
        first_loss = trainer.training_history[0].train_loss
        last_loss = trainer.training_history[-1].train_loss
        self.assertLess(last_loss, first_loss * 1.1)  # Allow some tolerance


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCaliforniaHousingDataset,
        TestHousingNeuralNetwork,
        TestEarlyStopping,
        TestTrainingMetrics,
        TestPyTorchNeuralNetworkTrainer,
        TestGPUModelTrainerIntegration,
        TestRealDataIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")