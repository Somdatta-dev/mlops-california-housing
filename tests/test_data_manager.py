"""
Unit tests for data management components.
Tests DataManager, CaliforniaHousingData, and related functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import pickle

# Import components to test
import sys
sys.path.append('src')

from data_manager import (
    DataManager, 
    CaliforniaHousingData, 
    DataQualityReport, 
    PreprocessingConfig
)


class TestCaliforniaHousingData:
    """Test cases for CaliforniaHousingData Pydantic model."""
    
    def test_valid_data(self):
        """Test validation with valid data."""
        valid_data = {
            'MedInc': 5.0,
            'HouseAge': 25.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -118.0,
            'target': 2.5
        }
        
        housing_data = CaliforniaHousingData(**valid_data)
        assert housing_data.MedInc == 5.0
        assert housing_data.target == 2.5
    
    def test_invalid_income_range(self):
        """Test validation with invalid income range."""
        invalid_data = {
            'MedInc': 20.0,  # Too high
            'HouseAge': 25.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -118.0
        }
        
        with pytest.raises(ValueError):
            CaliforniaHousingData(**invalid_data)
    
    def test_invalid_house_age(self):
        """Test validation with invalid house age."""
        invalid_data = {
            'MedInc': 5.0,
            'HouseAge': 0.0,  # Too low
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -118.0
        }
        
        with pytest.raises(ValueError):
            CaliforniaHousingData(**invalid_data)
    
    def test_bedroom_room_validation(self):
        """Test custom validation for bedroom/room ratio."""
        invalid_data = {
            'MedInc': 5.0,
            'HouseAge': 25.0,
            'AveRooms': 4.0,
            'AveBedrms': 5.0,  # More bedrooms than rooms
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -118.0
        }
        
        with pytest.raises(ValueError, match="Average bedrooms cannot exceed average rooms"):
            CaliforniaHousingData(**invalid_data)
    
    def test_geographic_boundaries(self):
        """Test geographic boundary validation."""
        # Test latitude boundary
        data_lat = {
            'MedInc': 5.0,
            'HouseAge': 25.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 32.0,  # Below minimum
            'Longitude': -118.0
        }
        
        with pytest.raises(ValueError):
            CaliforniaHousingData(**data_lat)
        
        # Test longitude boundary
        data_lon = {
            'MedInc': 5.0,
            'HouseAge': 25.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -110.0  # Above maximum
        }
        
        with pytest.raises(ValueError):
            CaliforniaHousingData(**data_lon)
    
    def test_optional_target(self):
        """Test that target field is optional."""
        data_without_target = {
            'MedInc': 5.0,
            'HouseAge': 25.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -118.0
        }
        
        housing_data = CaliforniaHousingData(**data_without_target)
        assert housing_data.target is None


class TestPreprocessingConfig:
    """Test cases for PreprocessingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.scaler_type == "standard"
        assert config.handle_outliers is True
        assert config.feature_engineering is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PreprocessingConfig(
            test_size=0.3,
            random_state=123,
            scaler_type="robust",
            handle_outliers=False
        )
        
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.scaler_type == "robust"
        assert config.handle_outliers is False


class TestDataManager:
    """Test cases for DataManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample California Housing data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        features_df = pd.DataFrame({
            'MedInc': np.random.uniform(1, 10, n_samples),
            'HouseAge': np.random.uniform(1, 50, n_samples),
            'AveRooms': np.random.uniform(3, 8, n_samples),
            'AveBedrms': np.random.uniform(0.8, 2.0, n_samples),
            'Population': np.random.uniform(100, 5000, n_samples),
            'AveOccup': np.random.uniform(2, 6, n_samples),
            'Latitude': np.random.uniform(33, 41, n_samples),
            'Longitude': np.random.uniform(-124, -115, n_samples)
        })
        
        targets_series = pd.Series(np.random.uniform(0.5, 5.0, n_samples))
        
        return features_df, targets_series
    
    @pytest.fixture
    def data_manager(self, temp_dir):
        """Create DataManager instance with temporary directory."""
        config = PreprocessingConfig(test_size=0.2, random_state=42)
        return DataManager(data_dir=temp_dir, config=config)
    
    def test_initialization(self, data_manager, temp_dir):
        """Test DataManager initialization."""
        assert data_manager.data_dir == Path(temp_dir)
        assert data_manager.raw_dir.exists()
        assert data_manager.processed_dir.exists()
        assert data_manager.interim_dir.exists()
        assert data_manager.config.test_size == 0.2
    
    @patch.dict('os.environ', {'DVC_REMOTE_URL': 'gdrive://test-folder-id'})
    @patch('subprocess.run')
    def test_setup_dvc_remote_success(self, mock_subprocess, data_manager):
        """Test successful DVC remote setup."""
        mock_subprocess.return_value = MagicMock()
        
        result = data_manager.setup_dvc_remote()
        
        assert result is True
        assert mock_subprocess.call_count >= 1
    
    @patch.dict('os.environ', {}, clear=True)
    def test_setup_dvc_remote_no_url(self, data_manager):
        """Test DVC remote setup without URL."""
        result = data_manager.setup_dvc_remote()
        assert result is False
    
    def test_download_raw_data(self, data_manager, sample_data):
        """Test raw data download with real data."""
        # Test with real California Housing dataset
        result_features, result_targets = data_manager.download_raw_data()
        
        # Check that files were created
        assert data_manager.features_file.exists()
        assert data_manager.targets_file.exists()
        assert data_manager.metadata_file.exists()
        
        # Check basic data properties
        assert len(result_features) > 0
        assert len(result_targets) > 0
        assert len(result_features) == len(result_targets)
        assert len(result_features.columns) == 8  # California Housing has 8 features
    
    def test_validate_data_quality(self, data_manager, sample_data):
        """Test data quality validation."""
        features_df, targets_series = sample_data
        
        report = data_manager.validate_data_quality(features_df, targets_series)
        
        assert isinstance(report, DataQualityReport)
        assert report.total_samples == len(features_df)
        assert report.total_features == len(features_df.columns)
        assert isinstance(report.is_valid, bool)
        assert isinstance(report.validation_errors, list)
        assert isinstance(report.validation_warnings, list)
        assert data_manager.quality_report_file.exists()
    
    def test_validate_data_quality_with_issues(self, data_manager):
        """Test data quality validation with data issues."""
        # Create problematic data with actual duplicates
        features_df = pd.DataFrame({
            'MedInc': [5.0, np.nan, 3.0, 5.0],  # Missing value and duplicate
            'HouseAge': [25.0, 30.0, 25.0, 25.0],  # Duplicate row
            'AveRooms': [6.0, 5.0, 6.0, 6.0],
            'AveBedrms': [1.2, 1.0, 1.2, 1.2],
            'Population': [3000.0, 2500.0, 3000.0, 3000.0],
            'AveOccup': [3.0, 2.8, 3.0, 3.0],
            'Latitude': [34.0, 35.0, 34.0, 34.0],
            'Longitude': [-118.0, -119.0, -118.0, -118.0]
        })
        
        targets_series = pd.Series([2.5, 3.0, 2.5, 2.5])
        
        report = data_manager.validate_data_quality(features_df, targets_series)
        
        assert len(report.validation_warnings) > 0
        assert report.missing_values['MedInc'] == 1
        assert report.duplicates == 1  # Should detect the duplicate between rows 0 and 3
    
    def test_engineer_features(self, data_manager, sample_data):
        """Test feature engineering."""
        features_df, _ = sample_data
        
        engineered_df = data_manager.engineer_features(features_df)
        
        # Check that new features were created
        expected_new_features = [
            'RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold',
            'DistanceFromCenter', 'IncomePerRoom', 'IncomePerPerson',
            'HouseAgeCategory', 'PopulationDensity'
        ]
        
        for feature in expected_new_features:
            assert feature in engineered_df.columns
        
        assert engineered_df.shape[1] > features_df.shape[1]
    
    def test_handle_outliers_iqr(self, data_manager, sample_data):
        """Test outlier handling with IQR method."""
        features_df, _ = sample_data
        
        # Add some outliers
        features_df.loc[0, 'MedInc'] = 50.0  # Extreme outlier
        
        cleaned_df = data_manager.handle_outliers(features_df, method="iqr")
        
        # Check that outlier was capped
        assert cleaned_df.loc[0, 'MedInc'] < 50.0
        assert cleaned_df.shape == features_df.shape  # No rows removed
    
    def test_handle_outliers_disabled(self, data_manager, sample_data):
        """Test outlier handling when disabled."""
        data_manager.config.handle_outliers = False
        features_df, _ = sample_data
        
        # Add outlier
        features_df.loc[0, 'MedInc'] = 50.0
        
        result_df = data_manager.handle_outliers(features_df)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(result_df, features_df)
    
    def test_preprocess_data(self, data_manager, sample_data):
        """Test complete preprocessing pipeline."""
        features_df, targets_series = sample_data
        
        result = data_manager.preprocess_data(features_df, targets_series)
        
        # Check that all required keys are present
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'scaler', 'feature_names', 'metadata']
        for key in required_keys:
            assert key in result
        
        # Check data shapes
        assert len(result['X_train']) + len(result['X_val']) + len(result['X_test']) <= len(features_df)
        assert len(result['y_train']) == len(result['X_train'])
        assert len(result['y_val']) == len(result['X_val'])
        assert len(result['y_test']) == len(result['X_test'])
        
        # Check that files were saved
        assert (data_manager.processed_dir / "X_train.csv").exists()
        assert (data_manager.processed_dir / "y_train.csv").exists()
        assert data_manager.scaler_file.exists()
        
        # Check scaler functionality
        scaler = result['scaler']
        test_data = result['X_train'].iloc[:5]
        scaled_data = scaler.transform(test_data)
        assert scaled_data.shape == test_data.shape
    
    def test_get_data_summary(self, data_manager):
        """Test data summary generation."""
        summary = data_manager.get_data_summary()
        
        required_keys = ['data_manager_config', 'dvc_remote_url', 'data_directories', 'file_status']
        for key in required_keys:
            assert key in summary
        
        assert 'raw' in summary['data_directories']
        assert 'processed' in summary['data_directories']
        assert 'interim' in summary['data_directories']
    
    @patch('subprocess.run')
    def test_track_with_dvc_success(self, mock_subprocess, data_manager, temp_dir):
        """Test successful DVC file tracking."""
        mock_subprocess.return_value = MagicMock()
        
        # Create a test file
        test_file = Path(temp_dir) / "test_file.csv"
        test_file.write_text("test,data\n1,2\n")
        
        result = data_manager.track_with_dvc(test_file)
        
        assert result is True
        assert mock_subprocess.call_count >= 1
    
    @patch('subprocess.run')
    def test_track_with_dvc_failure(self, mock_subprocess, data_manager, temp_dir):
        """Test DVC file tracking failure."""
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, 'dvc')
        
        test_file = Path(temp_dir) / "test_file.csv"
        test_file.write_text("test,data\n1,2\n")
        
        result = data_manager.track_with_dvc(test_file)
        
        assert result is False


class TestDataQualityReport:
    """Test cases for DataQualityReport dataclass."""
    
    def test_data_quality_report_creation(self):
        """Test DataQualityReport creation and serialization."""
        report = DataQualityReport(
            is_valid=True,
            total_samples=1000,
            total_features=8,
            missing_values={'MedInc': 0, 'HouseAge': 5},
            outliers={'MedInc': 10, 'HouseAge': 15},
            duplicates=2,
            data_types={'MedInc': 'float64', 'HouseAge': 'float64'},
            statistics={'MedInc': {'mean': 5.0, 'std': 2.0}},
            validation_errors=[],
            validation_warnings=['Some warning'],
            timestamp='2024-01-01T00:00:00'
        )
        
        assert report.is_valid is True
        assert report.total_samples == 1000
        assert report.missing_values['HouseAge'] == 5
        assert len(report.validation_warnings) == 1
        
        # Test serialization
        from dataclasses import asdict
        report_dict = asdict(report)
        assert isinstance(report_dict, dict)
        assert report_dict['is_valid'] is True


class TestIntegration:
    """Integration tests for data management components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_pipeline(self, temp_dir):
        """Test complete data management pipeline with real data."""
        # Initialize DataManager
        config = PreprocessingConfig(
            test_size=0.2,
            random_state=42,
            feature_engineering=True,
            handle_outliers=True
        )
        data_manager = DataManager(data_dir=temp_dir, config=config)
        
        # Run full pipeline
        # 1. Download data (uses real California Housing dataset)
        features, targets = data_manager.download_raw_data()
        
        # 2. Validate quality
        quality_report = data_manager.validate_data_quality(features, targets)
        
        # 3. Preprocess data
        processed_data = data_manager.preprocess_data(features, targets)
        
        # 4. Get summary
        summary = data_manager.get_data_summary()
        
        # Verify results
        assert quality_report.is_valid
        assert quality_report.total_samples > 0  # Should have some data
        assert quality_report.total_features == 8  # California Housing has 8 features
        
        assert 'X_train' in processed_data
        assert 'scaler' in processed_data
        
        assert summary['file_status']['raw_features'] is True
        assert summary['file_status']['quality_report'] is True
        
        # Verify file structure
        assert (data_manager.raw_dir / "california_housing_features.csv").exists()
        assert (data_manager.processed_dir / "X_train.csv").exists()
        assert (data_manager.processed_dir / "preprocessing_metadata.json").exists()
        
        # Test data integrity
        train_data = pd.read_csv(data_manager.processed_dir / "X_train.csv")
        assert len(train_data.columns) > 8  # Feature engineering added columns
        
        # Test scaler loading
        with open(data_manager.scaler_file, 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        # Test that scaler works
        sample_data = train_data.iloc[:5]
        transformed = loaded_scaler.transform(sample_data)
        assert transformed.shape == sample_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])