"""
Core Data Management Implementation for MLOps Platform.
Provides comprehensive data management with DVC integration, validation, and preprocessing.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import logging
import json
import subprocess
import pickle
from dataclasses import dataclass, asdict
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaliforniaHousingData(BaseModel):
    """
    Pydantic model for California Housing dataset with comprehensive validation.
    """
    MedInc: float = Field(
        ..., 
        ge=0.0, 
        le=15.0, 
        description="Median income in block group (in tens of thousands of dollars)"
    )
    HouseAge: float = Field(
        ..., 
        ge=1.0, 
        le=52.0, 
        description="Median house age in block group (in years)"
    )
    AveRooms: float = Field(
        ..., 
        ge=1.0, 
        le=20.0, 
        description="Average number of rooms per household"
    )
    AveBedrms: float = Field(
        ..., 
        ge=0.0, 
        le=5.0, 
        description="Average number of bedrooms per household"
    )
    Population: float = Field(
        ..., 
        ge=3.0, 
        le=35682.0, 
        description="Block group population"
    )
    AveOccup: float = Field(
        ..., 
        ge=0.5, 
        le=1243.0, 
        description="Average number of household members"
    )
    Latitude: float = Field(
        ..., 
        ge=32.54, 
        le=41.95, 
        description="Block group latitude"
    )
    Longitude: float = Field(
        ..., 
        ge=-124.35, 
        le=-114.31, 
        description="Block group longitude"
    )
    target: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description="Median house value (target variable, in hundreds of thousands of dollars)"
    )
    
    @field_validator('AveBedrms')
    @classmethod
    def validate_bedrooms_ratio(cls, v, info):
        """Validate that average bedrooms is reasonable compared to average rooms."""
        if info.data and 'AveRooms' in info.data and info.data['AveRooms'] is not None:
            if v > info.data['AveRooms']:
                raise ValueError("Average bedrooms cannot exceed average rooms")
            if v > info.data['AveRooms'] * 0.8:  # More than 80% bedrooms seems unrealistic
                logger.warning(f"High bedroom ratio: {v}/{info.data['AveRooms']}")
        return v
    
    @field_validator('AveOccup')
    @classmethod
    def validate_occupancy(cls, v, info):
        """Validate occupancy makes sense given population and rooms."""
        if info.data and 'Population' in info.data and 'AveRooms' in info.data:
            if info.data['Population'] is not None and info.data['AveRooms'] is not None:
                # Rough check: occupancy should be reasonable
                if v > 50:  # Very high occupancy
                    logger.warning(f"Very high occupancy detected: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_geographic_consistency(self):
        """Validate that latitude and longitude are consistent with California."""
        lat = self.Latitude
        lon = self.Longitude
        
        if lat is not None and lon is not None:
            # Additional checks for California boundaries
            if lat < 32.5 or lat > 42.0:
                logger.warning(f"Latitude {lat} is outside typical California range")
            if lon < -124.5 or lon > -114.0:
                logger.warning(f"Longitude {lon} is outside typical California range")
        
        return self


@dataclass
class DataQualityReport:
    """Data quality report structure."""
    is_valid: bool
    total_samples: int
    total_features: int
    missing_values: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int
    data_types: Dict[str, str]
    statistics: Dict[str, Dict[str, float]]
    validation_errors: List[str]
    validation_warnings: List[str]
    timestamp: str


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline."""
    test_size: float = 0.2
    random_state: int = 42
    scaler_type: str = "standard"  # "standard", "robust", "minmax"
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore"
    outlier_threshold: float = 3.0
    imputation_strategy: str = "median"  # "mean", "median", "most_frequent"
    feature_engineering: bool = True
    validation_split: float = 0.2


class DataManager:
    """
    Comprehensive data manager with DVC integration and environment-based configuration.
    """
    
    def __init__(self, data_dir: str = "data", config: Optional[PreprocessingConfig] = None):
        """
        Initialize the DataManager.
        
        Args:
            data_dir: Base directory for data storage
            config: Preprocessing configuration
        """
        # Load environment variables
        load_dotenv()
        
        # Setup directories
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.interim_dir = self.data_dir / "interim"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.interim_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or PreprocessingConfig()
        
        # DVC configuration
        self.dvc_remote_url = os.getenv('DVC_REMOTE_URL')
        self.dvc_remote_name = "gdrive"
        
        # File paths
        self.features_file = self.raw_dir / "california_housing_features.csv"
        self.targets_file = self.raw_dir / "california_housing_targets.csv"
        self.metadata_file = self.raw_dir / "dataset_metadata.json"
        self.quality_report_file = self.processed_dir / "data_quality_report.json"
        
        # Processed data files
        self.processed_features_file = self.processed_dir / "processed_features.csv"
        self.processed_targets_file = self.processed_dir / "processed_targets.csv"
        self.scaler_file = self.processed_dir / "scaler.pkl"
        
        logger.info(f"DataManager initialized with data directory: {self.data_dir}")
    
    def setup_dvc_remote(self) -> bool:
        """
        Configure DVC remote storage using environment variables.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.dvc_remote_url:
            logger.error("DVC_REMOTE_URL not found in environment variables!")
            return False
        
        try:
            # Check if DVC is initialized
            if not (Path.cwd() / ".dvc").exists():
                logger.info("Initializing DVC...")
                subprocess.run(["dvc", "init"], check=True)
            
            # Configure remote
            logger.info(f"Configuring DVC remote: {self.dvc_remote_url}")
            subprocess.run([
                "dvc", "remote", "add", "-d", self.dvc_remote_name, self.dvc_remote_url
            ], check=True)
            
            # Configure Google Drive specific settings if needed
            if self.dvc_remote_url.startswith("gdrive://"):
                use_service_account = os.getenv('GDRIVE_USE_SERVICE_ACCOUNT', 'false').lower() == 'true'
                if use_service_account:
                    subprocess.run([
                        "dvc", "remote", "modify", self.dvc_remote_name, 
                        "gdrive_use_service_account", "true"
                    ], check=True)
            
            logger.info("DVC remote configured successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure DVC remote: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error configuring DVC: {e}")
            return False
    
    def download_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download California Housing dataset from sklearn.
        
        Returns:
            Tuple of (features_df, targets_series)
        """
        logger.info("Downloading California Housing dataset...")
        
        # Fetch dataset
        housing_data = fetch_california_housing(as_frame=True)
        features_df = housing_data.data
        targets_series = housing_data.target
        
        # Save raw data
        features_df.to_csv(self.features_file, index=False)
        targets_series.to_csv(self.targets_file, index=False, header=["MedHouseVal"])
        
        # Save metadata
        metadata = {
            "dataset_name": "California Housing",
            "source": "sklearn.datasets.fetch_california_housing",
            "download_timestamp": datetime.now().isoformat(),
            "n_samples": len(features_df),
            "n_features": len(features_df.columns),
            "feature_names": list(features_df.columns),
            "target_name": "MedHouseVal",
            "description": housing_data.DESCR,
            "data_shape": features_df.shape,
            "target_shape": targets_series.shape
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Raw data saved: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        return features_df, targets_series
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load raw data from files or download if not exists.
        
        Returns:
            Tuple of (features_df, targets_series)
        """
        if not self.features_file.exists() or not self.targets_file.exists():
            logger.info("Raw data not found, downloading...")
            return self.download_raw_data()
        
        logger.info("Loading raw data from files...")
        features_df = pd.read_csv(self.features_file)
        targets_series = pd.read_csv(self.targets_file)["MedHouseVal"]
        
        return features_df, targets_series
    
    def validate_data_quality(self, features_df: pd.DataFrame, targets_series: pd.Series) -> DataQualityReport:
        """
        Comprehensive data quality validation and reporting.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            DataQualityReport object
        """
        logger.info("Performing data quality validation...")
        
        errors = []
        warnings = []
        
        # Basic validation
        if len(features_df) != len(targets_series):
            errors.append(f"Features and targets length mismatch: {len(features_df)} vs {len(targets_series)}")
        
        # Check for missing values
        missing_features = features_df.isnull().sum().to_dict()
        missing_targets = int(targets_series.isnull().sum())
        
        if sum(missing_features.values()) > 0:
            warnings.append(f"Missing values in features: {missing_features}")
        
        if missing_targets > 0:
            warnings.append(f"Missing values in targets: {missing_targets}")
        
        # Check for duplicates (excluding NaN values for proper duplicate detection)
        duplicates = features_df.dropna().duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate rows")
        
        # Outlier detection using IQR method
        outliers = {}
        for column in features_df.select_dtypes(include=[np.number]).columns:
            Q1 = features_df[column].quantile(0.25)
            Q3 = features_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((features_df[column] < lower_bound) | (features_df[column] > upper_bound)).sum()
            outliers[column] = int(outlier_count)
        
        # Target outliers
        target_Q1 = targets_series.quantile(0.25)
        target_Q3 = targets_series.quantile(0.75)
        target_IQR = target_Q3 - target_Q1
        target_outliers = ((targets_series < target_Q1 - 1.5 * target_IQR) | 
                          (targets_series > target_Q3 + 1.5 * target_IQR)).sum()
        outliers['target'] = int(target_outliers)
        
        # Generate statistics
        statistics = {}
        for column in features_df.columns:
            statistics[column] = {
                'mean': float(features_df[column].mean()),
                'std': float(features_df[column].std()),
                'min': float(features_df[column].min()),
                'max': float(features_df[column].max()),
                'median': float(features_df[column].median()),
                'skewness': float(features_df[column].skew()),
                'kurtosis': float(features_df[column].kurtosis())
            }
        
        statistics['target'] = {
            'mean': float(targets_series.mean()),
            'std': float(targets_series.std()),
            'min': float(targets_series.min()),
            'max': float(targets_series.max()),
            'median': float(targets_series.median()),
            'skewness': float(targets_series.skew()),
            'kurtosis': float(targets_series.kurtosis())
        }
        
        # Validate using Pydantic model
        pydantic_errors = []
        sample_size = min(1000, len(features_df))  # Validate sample for performance
        sample_indices = np.random.choice(len(features_df), sample_size, replace=False)
        
        for idx in sample_indices:
            try:
                row_data = features_df.iloc[idx].to_dict()
                row_data['target'] = targets_series.iloc[idx]
                CaliforniaHousingData(**row_data)
            except Exception as e:
                pydantic_errors.append(f"Row {idx}: {str(e)}")
        
        if pydantic_errors:
            warnings.extend(pydantic_errors[:10])  # Limit to first 10 errors
            if len(pydantic_errors) > 10:
                warnings.append(f"... and {len(pydantic_errors) - 10} more validation errors")
        
        # Create report
        report = DataQualityReport(
            is_valid=len(errors) == 0,
            total_samples=len(features_df),
            total_features=len(features_df.columns),
            missing_values=missing_features,
            outliers=outliers,
            duplicates=int(duplicates),
            data_types={col: str(dtype) for col, dtype in features_df.dtypes.items()},
            statistics=statistics,
            validation_errors=errors,
            validation_warnings=warnings,
            timestamp=datetime.now().isoformat()
        )
        
        # Save report
        with open(self.quality_report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info(f"Data quality validation completed. Valid: {report.is_valid}")
        return report
    
    def engineer_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Performing feature engineering...")
        
        df = features_df.copy()
        
        if self.config.feature_engineering:
            # Create new features
            df['RoomsPerHousehold'] = df['AveRooms']
            df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
            df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']
            
            # Geographic features
            df['DistanceFromCenter'] = np.sqrt((df['Latitude'] - df['Latitude'].mean())**2 + 
                                             (df['Longitude'] - df['Longitude'].mean())**2)
            
            # Income-related features
            df['IncomePerRoom'] = df['MedInc'] / df['AveRooms']
            df['IncomePerPerson'] = df['MedInc'] / df['AveOccup']
            
            # Age-related features
            df['HouseAgeCategory'] = pd.cut(df['HouseAge'], 
                                          bins=[0, 10, 20, 30, 40, 100], 
                                          labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old'])
            df['HouseAgeCategory'] = df['HouseAgeCategory'].cat.codes
            
            # Population density
            df['PopulationDensity'] = df['Population'] / (df['AveRooms'] * df['AveOccup'])
            
            logger.info(f"Feature engineering completed. New shape: {df.shape}")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the dataset.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ("iqr" or "zscore")
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        if not self.config.handle_outliers:
            return df
        
        logger.info(f"Handling outliers using {method} method...")
        
        df_clean = df.copy()
        outlier_counts = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > threshold
            else:
                continue
            
            outlier_count = outliers.sum()
            outlier_counts[column] = outlier_count
            
            if outlier_count > 0:
                # Cap outliers instead of removing them
                df_clean.loc[df[column] < lower_bound, column] = lower_bound
                df_clean.loc[df[column] > upper_bound, column] = upper_bound
        
        logger.info(f"Outliers handled: {outlier_counts}")
        return df_clean    

    def preprocess_data(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Dict[str, Any]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            features_df: Input features DataFrame
            targets_series: Input targets Series
            
        Returns:
            Dictionary containing processed data splits and metadata
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Feature engineering
        features_engineered = self.engineer_features(features_df)
        
        # Handle outliers
        features_clean = self.handle_outliers(features_engineered)
        
        # Handle missing values
        if features_clean.isnull().sum().sum() > 0:
            logger.info("Handling missing values...")
            imputer = SimpleImputer(strategy=self.config.imputation_strategy)
            features_imputed = pd.DataFrame(
                imputer.fit_transform(features_clean),
                columns=features_clean.columns,
                index=features_clean.index
            )
        else:
            features_imputed = features_clean
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_imputed, targets_series,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=None  # Regression task
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )
        
        # Scale features
        if self.config.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.config.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()  # Default
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Save processed data
        X_train_scaled.to_csv(self.processed_dir / "X_train.csv", index=False)
        X_val_scaled.to_csv(self.processed_dir / "X_val.csv", index=False)
        X_test_scaled.to_csv(self.processed_dir / "X_test.csv", index=False)
        y_train.to_csv(self.processed_dir / "y_train.csv", index=False, header=["target"])
        y_val.to_csv(self.processed_dir / "y_val.csv", index=False, header=["target"])
        y_test.to_csv(self.processed_dir / "y_test.csv", index=False, header=["target"])
        
        # Save scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Create preprocessing metadata
        preprocessing_metadata = {
            "preprocessing_config": asdict(self.config),
            "original_shape": features_df.shape,
            "processed_shape": features_imputed.shape,
            "train_shape": X_train_scaled.shape,
            "val_shape": X_val_scaled.shape,
            "test_shape": X_test_scaled.shape,
            "feature_names": list(X_train_scaled.columns),
            "scaler_type": self.config.scaler_type,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.processed_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(preprocessing_metadata, f, indent=2)
        
        result = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaler": scaler,
            "feature_names": list(X_train_scaled.columns),
            "metadata": preprocessing_metadata
        }
        
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return result
    
    def track_with_dvc(self, file_path: Union[str, Path]) -> bool:
        """
        Track a file with DVC.
        
        Args:
            file_path: Path to file to track
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Tracking file with DVC: {file_path}")
            subprocess.run(["dvc", "add", str(file_path)], check=True)
            
            # Add .dvc file to git
            dvc_file = str(file_path) + ".dvc"
            if Path(dvc_file).exists():
                subprocess.run(["git", "add", dvc_file], check=True)
            
            logger.info(f"File tracked successfully: {file_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to track file with DVC: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary with data summary information
        """
        summary = {
            "data_manager_config": asdict(self.config),
            "dvc_remote_url": self.dvc_remote_url,
            "data_directories": {
                "raw": str(self.raw_dir),
                "processed": str(self.processed_dir),
                "interim": str(self.interim_dir)
            },
            "file_status": {
                "raw_features": self.features_file.exists(),
                "raw_targets": self.targets_file.exists(),
                "metadata": self.metadata_file.exists(),
                "quality_report": self.quality_report_file.exists(),
                "processed_data": (self.processed_dir / "X_train.csv").exists()
            }
        }
        
        # Add metadata if available
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                summary["dataset_metadata"] = json.load(f)
        
        # Add quality report if available
        if self.quality_report_file.exists():
            with open(self.quality_report_file, 'r') as f:
                summary["quality_report"] = json.load(f)
        
        return summary


def main():
    """
    Main function to demonstrate DataManager functionality.
    """
    logger.info("Starting DataManager demonstration...")
    
    # Initialize DataManager
    config = PreprocessingConfig(
        test_size=0.2,
        random_state=42,
        scaler_type="standard",
        handle_outliers=True,
        feature_engineering=True
    )
    
    data_manager = DataManager(config=config)
    
    # Setup DVC remote
    data_manager.setup_dvc_remote()
    
    # Load raw data
    features_df, targets_series = data_manager.load_raw_data()
    
    # Validate data quality
    quality_report = data_manager.validate_data_quality(features_df, targets_series)
    
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    print(f"Valid: {quality_report.is_valid}")
    print(f"Samples: {quality_report.total_samples:,}")
    print(f"Features: {quality_report.total_features}")
    print(f"Duplicates: {quality_report.duplicates}")
    print(f"Errors: {len(quality_report.validation_errors)}")
    print(f"Warnings: {len(quality_report.validation_warnings)}")
    
    # Preprocess data
    processed_data = data_manager.preprocess_data(features_df, targets_series)
    
    print("\n" + "="*60)
    print("PREPROCESSING RESULTS")
    print("="*60)
    print(f"Original shape: {features_df.shape}")
    print(f"Processed features: {processed_data['X_train'].shape[1]}")
    print(f"Train samples: {processed_data['X_train'].shape[0]:,}")
    print(f"Validation samples: {processed_data['X_val'].shape[0]:,}")
    print(f"Test samples: {processed_data['X_test'].shape[0]:,}")
    
    # Get summary
    summary = data_manager.get_data_summary()
    print(f"\nData files created: {sum(summary['file_status'].values())}/5")
    
    logger.info("DataManager demonstration completed!")


if __name__ == "__main__":
    main()