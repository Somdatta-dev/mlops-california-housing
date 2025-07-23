"""
Data validation utilities for California Housing dataset.
Ensures data quality and consistency with comprehensive validation checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Data class to store validation results.
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    timestamp: str


class CaliforniaHousingValidator:
    """
    Validator for California Housing dataset with comprehensive quality checks.
    """
    
    # Expected feature names and their constraints
    EXPECTED_FEATURES = {
        'MedInc': {'min': 0.0, 'max': 15.0, 'type': 'float'},
        'HouseAge': {'min': 1.0, 'max': 52.0, 'type': 'float'},
        'AveRooms': {'min': 1.0, 'max': 20.0, 'type': 'float'},
        'AveBedrms': {'min': 0.0, 'max': 5.0, 'type': 'float'},
        'Population': {'min': 3.0, 'max': 35682.0, 'type': 'float'},
        'AveOccup': {'min': 0.5, 'max': 1243.0, 'type': 'float'},
        'Latitude': {'min': 32.54, 'max': 41.95, 'type': 'float'},
        'Longitude': {'min': -124.35, 'max': -114.31, 'type': 'float'}
    }
    
    TARGET_CONSTRAINTS = {
        'MedHouseVal': {'min': 0.0, 'max': 10.0, 'type': 'float'}
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_history = []
    
    def validate_schema(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Validate the schema of features and targets.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check feature columns
        expected_features = set(self.EXPECTED_FEATURES.keys())
        actual_features = set(features_df.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")
        
        if extra_features:
            warnings.append(f"Unexpected features found: {extra_features}")
        
        # Check data types
        for feature, constraints in self.EXPECTED_FEATURES.items():
            if feature in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[feature]):
                    errors.append(f"Feature '{feature}' should be numeric, got {features_df[feature].dtype}")
        
        # Check target
        if not pd.api.types.is_numeric_dtype(targets_series):
            errors.append(f"Target should be numeric, got {targets_series.dtype}")
        
        # Check shapes match
        if len(features_df) != len(targets_series):
            errors.append(f"Features and targets length mismatch: {len(features_df)} vs {len(targets_series)}")
        
        return errors, warnings
    
    def validate_data_quality(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Validate data quality including missing values, duplicates, and outliers.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check for missing values
        missing_features = features_df.isnull().sum()
        missing_targets = targets_series.isnull().sum()
        
        if missing_features.sum() > 0:
            errors.append(f"Missing values in features: {missing_features[missing_features > 0].to_dict()}")
        
        if missing_targets > 0:
            errors.append(f"Missing values in targets: {missing_targets}")
        
        # Check for duplicates
        duplicate_rows = features_df.duplicated().sum()
        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate rows in features")
        
        # Check for infinite values
        inf_features = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        inf_targets = np.isinf(targets_series).sum()
        
        if inf_features > 0:
            errors.append(f"Found {inf_features} infinite values in features")
        
        if inf_targets > 0:
            errors.append(f"Found {inf_targets} infinite values in targets")
        
        return errors, warnings
    
    def validate_value_ranges(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Validate that values are within expected ranges.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Validate feature ranges
        for feature, constraints in self.EXPECTED_FEATURES.items():
            if feature in features_df.columns:
                min_val = features_df[feature].min()
                max_val = features_df[feature].max()
                
                if min_val < constraints['min']:
                    warnings.append(f"Feature '{feature}' has values below expected minimum: {min_val} < {constraints['min']}")
                
                if max_val > constraints['max']:
                    warnings.append(f"Feature '{feature}' has values above expected maximum: {max_val} > {constraints['max']}")
        
        # Validate target range
        target_min = targets_series.min()
        target_max = targets_series.max()
        
        if target_min < self.TARGET_CONSTRAINTS['MedHouseVal']['min']:
            warnings.append(f"Target has values below expected minimum: {target_min} < {self.TARGET_CONSTRAINTS['MedHouseVal']['min']}")
        
        if target_max > self.TARGET_CONSTRAINTS['MedHouseVal']['max']:
            warnings.append(f"Target has values above expected maximum: {target_max} > {self.TARGET_CONSTRAINTS['MedHouseVal']['max']}")
        
        return errors, warnings
    
    def validate_statistical_properties(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Validate statistical properties of the data.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check for constant features
        constant_features = []
        for feature in features_df.columns:
            if features_df[feature].nunique() <= 1:
                constant_features.append(feature)
        
        if constant_features:
            warnings.append(f"Constant features detected: {constant_features}")
        
        # Check for highly correlated features (correlation > 0.95)
        correlation_matrix = features_df.corr()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            warnings.append(f"Highly correlated feature pairs (>0.95): {high_corr_pairs}")
        
        # Check target distribution
        target_skewness = targets_series.skew()
        if abs(target_skewness) > 2:
            warnings.append(f"Target distribution is highly skewed: {target_skewness:.3f}")
        
        return errors, warnings
    
    def generate_statistics(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for the dataset.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'dataset_shape': {
                'n_samples': len(features_df),
                'n_features': len(features_df.columns)
            },
            'feature_statistics': {},
            'target_statistics': {},
            'data_quality': {
                'missing_values_features': features_df.isnull().sum().to_dict(),
                'missing_values_target': int(targets_series.isnull().sum()),
                'duplicate_rows': int(features_df.duplicated().sum()),
                'infinite_values_features': int(np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()),
                'infinite_values_target': int(np.isinf(targets_series).sum())
            }
        }
        
        # Feature statistics
        for feature in features_df.columns:
            stats['feature_statistics'][feature] = {
                'mean': float(features_df[feature].mean()),
                'std': float(features_df[feature].std()),
                'min': float(features_df[feature].min()),
                'max': float(features_df[feature].max()),
                'median': float(features_df[feature].median()),
                'skewness': float(features_df[feature].skew()),
                'kurtosis': float(features_df[feature].kurtosis()),
                'unique_values': int(features_df[feature].nunique())
            }
        
        # Target statistics
        stats['target_statistics'] = {
            'mean': float(targets_series.mean()),
            'std': float(targets_series.std()),
            'min': float(targets_series.min()),
            'max': float(targets_series.max()),
            'median': float(targets_series.median()),
            'skewness': float(targets_series.skew()),
            'kurtosis': float(targets_series.kurtosis()),
            'unique_values': int(targets_series.nunique())
        }
        
        return stats
    
    def validate_dataset(self, features_df: pd.DataFrame, targets_series: pd.Series) -> ValidationResult:
        """
        Perform comprehensive validation of the dataset.
        
        Args:
            features_df: Features DataFrame
            targets_series: Targets Series
            
        Returns:
            ValidationResult object
        """
        logger.info("Starting comprehensive dataset validation...")
        
        all_errors = []
        all_warnings = []
        
        # Schema validation
        errors, warnings = self.validate_schema(features_df, targets_series)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Data quality validation
        errors, warnings = self.validate_data_quality(features_df, targets_series)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Value range validation
        errors, warnings = self.validate_value_ranges(features_df, targets_series)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Statistical properties validation
        errors, warnings = self.validate_statistical_properties(features_df, targets_series)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Generate statistics
        statistics = self.generate_statistics(features_df, targets_series)
        
        # Create validation result
        result = ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            statistics=statistics,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in history
        self.validation_history.append(result)
        
        # Log results
        if result.is_valid:
            logger.info("✅ Dataset validation PASSED!")
        else:
            logger.error("❌ Dataset validation FAILED!")
        
        if all_errors:
            logger.error(f"Errors found: {len(all_errors)}")
            for error in all_errors:
                logger.error(f"  - {error}")
        
        if all_warnings:
            logger.warning(f"Warnings found: {len(all_warnings)}")
            for warning in all_warnings:
                logger.warning(f"  - {warning}")
        
        return result
    
    def save_validation_report(self, result: ValidationResult, output_path: str):
        """
        Save validation report to JSON file.
        
        Args:
            result: ValidationResult object
            output_path: Path to save the report
        """
        report = {
            'validation_result': {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'timestamp': result.timestamp
            },
            'statistics': result.statistics
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_path}")


def main():
    """
    Main function to validate the California Housing dataset.
    """
    logger.info("Starting California Housing dataset validation...")
    
    # Load dataset
    from data_loader import CaliforniaHousingDataLoader
    
    loader = CaliforniaHousingDataLoader()
    features_df, targets_series = loader.load_dataset()
    
    # Initialize validator
    validator = CaliforniaHousingValidator()
    
    # Validate dataset
    result = validator.validate_dataset(features_df, targets_series)
    
    # Save validation report
    report_path = "data/raw/validation_report.json"
    validator.save_validation_report(result, report_path)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET VALIDATION SUMMARY")
    print("="*60)
    print(f"Validation Status: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Dataset Shape: {result.statistics['dataset_shape']['n_samples']:,} samples, {result.statistics['dataset_shape']['n_features']} features")
    print(f"Validation Timestamp: {result.timestamp}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")
    
    logger.info("Dataset validation completed!")
    
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    exit(main())