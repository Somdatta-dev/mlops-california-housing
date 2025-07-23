"""
Data loading script for California Housing dataset.
Downloads and stores the dataset in data/raw/ directory with DVC tracking.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaliforniaHousingDataLoader:
    """
    Data loader for California Housing dataset with DVC integration.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.features_file = self.data_dir / "california_housing_features.csv"
        self.targets_file = self.data_dir / "california_housing_targets.csv"
        self.metadata_file = self.data_dir / "dataset_metadata.json"
        
    def download_and_save_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download California Housing dataset from sklearn and save to CSV files.
        
        Returns:
            Tuple of (features_df, targets_series)
        """
        logger.info("Downloading California Housing dataset from sklearn...")
        
        # Fetch the dataset
        housing_data = fetch_california_housing(as_frame=True)
        
        # Extract features and targets
        features_df = housing_data.data
        targets_series = housing_data.target
        
        # Add some metadata
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
        
        # Save features to CSV
        logger.info(f"Saving features to {self.features_file}")
        features_df.to_csv(self.features_file, index=False)
        
        # Save targets to CSV
        logger.info(f"Saving targets to {self.targets_file}")
        targets_series.to_csv(self.targets_file, index=False, header=["MedHouseVal"])
        
        # Save metadata
        logger.info(f"Saving metadata to {self.metadata_file}")
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved successfully!")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Targets shape: {targets_series.shape}")
        
        return features_df, targets_series
    
    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the dataset from saved CSV files.
        
        Returns:
            Tuple of (features_df, targets_series)
        """
        if not self.features_file.exists() or not self.targets_file.exists():
            logger.warning("Dataset files not found. Downloading...")
            return self.download_and_save_dataset()
        
        logger.info("Loading dataset from saved files...")
        features_df = pd.read_csv(self.features_file)
        targets_series = pd.read_csv(self.targets_file)["MedHouseVal"]
        
        return features_df, targets_series
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information and statistics.
        
        Returns:
            Dictionary with dataset information
        """
        if not self.metadata_file.exists():
            logger.warning("Metadata file not found. Loading dataset first...")
            self.download_and_save_dataset()
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load data for additional statistics
        features_df, targets_series = self.load_dataset()
        
        # Add statistical information
        metadata.update({
            "features_stats": {
                "mean": features_df.mean().to_dict(),
                "std": features_df.std().to_dict(),
                "min": features_df.min().to_dict(),
                "max": features_df.max().to_dict(),
                "missing_values": features_df.isnull().sum().to_dict()
            },
            "target_stats": {
                "mean": float(targets_series.mean()),
                "std": float(targets_series.std()),
                "min": float(targets_series.min()),
                "max": float(targets_series.max()),
                "missing_values": int(targets_series.isnull().sum())
            }
        })
        
        return metadata


def main():
    """
    Main function to download and prepare the California Housing dataset.
    """
    logger.info("Starting California Housing dataset preparation...")
    
    # Initialize data loader
    loader = CaliforniaHousingDataLoader()
    
    # Download and save dataset
    features_df, targets_series = loader.download_and_save_dataset()
    
    # Display basic information
    print("\n" + "="*50)
    print("CALIFORNIA HOUSING DATASET SUMMARY")
    print("="*50)
    print(f"Features shape: {features_df.shape}")
    print(f"Targets shape: {targets_series.shape}")
    print(f"\nFeature columns: {list(features_df.columns)}")
    print(f"\nFirst few rows of features:")
    print(features_df.head())
    print(f"\nFirst few target values:")
    print(targets_series.head())
    
    # Get and display dataset info
    info = loader.get_dataset_info()
    print(f"\nDataset downloaded on: {info['download_timestamp']}")
    print(f"Total samples: {info['n_samples']:,}")
    print(f"Total features: {info['n_features']}")
    
    logger.info("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()