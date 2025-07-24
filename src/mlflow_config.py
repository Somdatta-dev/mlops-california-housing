"""
MLflow Configuration and Experiment Management

This module provides configuration management and utilities for MLflow experiment tracking,
model registry integration, and experiment lifecycle management for the MLOps platform.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MLflowConfig(BaseModel):
    """Configuration class for MLflow tracking and model registry."""
    
    tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        description="MLflow tracking server URI"
    )
    experiment_name: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "california-housing-prediction"),
        description="Name of the MLflow experiment"
    )
    artifact_location: Optional[str] = Field(
        default=None,
        description="Custom artifact storage location"
    )
    registry_uri: Optional[str] = Field(
        default=None,
        description="Model registry URI (defaults to tracking_uri)"
    )
    s3_endpoint_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        description="S3-compatible endpoint URL for artifact storage"
    )
    
    @field_validator('tracking_uri')
    @classmethod
    def validate_tracking_uri(cls, v):
        """Validate tracking URI format."""
        if not v:
            raise ValueError("Tracking URI cannot be empty")
        return v
    
    @field_validator('experiment_name')
    @classmethod
    def validate_experiment_name(cls, v):
        """Validate experiment name."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Experiment name cannot be empty")
        return v.strip()


@dataclass
class ExperimentMetrics:
    """Data class for experiment metrics."""
    rmse: float
    mae: float
    r2_score: float
    training_time: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    model_size_mb: Optional[float] = None


@dataclass
class ModelArtifacts:
    """Data class for model artifacts."""
    model_path: str
    feature_importance_plot: Optional[str] = None
    training_curves_plot: Optional[str] = None
    confusion_matrix_plot: Optional[str] = None
    model_summary: Optional[str] = None


class MLflowExperimentManager:
    """
    Manages MLflow experiments, runs, and model registry operations.
    
    This class provides high-level utilities for experiment tracking,
    model logging, and model registry management with comprehensive
    cross-platform support and fallback mechanisms.
    """
    
    def __init__(self, config: MLflowConfig):
        """
        Initialize the MLflow experiment manager with fallback support.
        
        Args:
            config: MLflow configuration object
        """
        self.config = config
        self.client = None
        self.experiment_id = None
        self.fallback_mode = False
        self._setup_mlflow_with_fallback()
    
    def _setup_mlflow_with_fallback(self) -> None:
        """Set up MLflow with comprehensive fallback mechanisms for cross-platform compatibility."""
        import platform
        import tempfile
        
        # List of fallback tracking URIs to try in order of preference
        fallback_uris = []
        
        # Primary URI from config
        fallback_uris.append(self.config.tracking_uri)
        
        # Generate platform-specific fallback URIs
        fallback_uris.extend(self._generate_fallback_uris())
        
        last_error = None
        
        for uri in fallback_uris:
            try:
                logger.info(f"Attempting MLflow setup with URI: {uri}")
                
                # Create a temporary config with the current URI
                temp_config = MLflowConfig(
                    tracking_uri=uri,
                    experiment_name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    registry_uri=self.config.registry_uri,
                    s3_endpoint_url=self.config.s3_endpoint_url
                )
                
                # Update current config
                self.config = temp_config
                
                # Try to set up MLflow with this URI
                self._setup_mlflow()
                
                # If we get here, setup was successful
                if uri != fallback_uris[0]:
                    logger.warning(f"Using fallback URI: {uri}")
                    self.fallback_mode = True
                
                return
                
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to setup MLflow with URI {uri}: {e}")
                continue
        
        # If all fallback attempts failed, raise the last error
        logger.error(f"All MLflow setup attempts failed. Last error: {last_error}")
        raise last_error
    
    def _generate_fallback_uris(self) -> List[str]:
        """Generate platform-specific fallback URIs."""
        import platform
        import tempfile
        
        fallback_uris = []
        
        # Get temporary directory for fallback storage
        temp_dir = tempfile.gettempdir()
        
        # Platform-specific URI generation
        system = platform.system()
        
        if system == "Windows":
            # Windows-specific fallbacks
            fallback_uris.extend([
                f"sqlite:///{temp_dir}\\mlflow.db",
                f"sqlite:///{temp_dir}/mlflow.db",  # Forward slash variant
                f"file:///{temp_dir.replace(chr(92), '/')}/mlruns",  # File URI with forward slashes
                f"{temp_dir}\\mlruns",  # Local path with backslashes
                f"{temp_dir}/mlruns",   # Local path with forward slashes
            ])
        else:
            # Unix-like systems (Linux, macOS)
            fallback_uris.extend([
                f"sqlite:///{temp_dir}/mlflow.db",
                f"file://{temp_dir}/mlruns",
                f"{temp_dir}/mlruns",
            ])
        
        # Universal fallbacks (work on all platforms)
        fallback_uris.extend([
            "sqlite:///:memory:",  # In-memory SQLite (last resort)
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_uris = []
        for uri in fallback_uris:
            if uri not in seen:
                seen.add(uri)
                unique_uris.append(uri)
        
        return unique_uris
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking and create/get experiment."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Set registry URI if provided
            if self.config.registry_uri:
                mlflow.set_registry_uri(self.config.registry_uri)
            
            # Configure S3 endpoint if provided
            if self.config.s3_endpoint_url:
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.s3_endpoint_url
            
            # Initialize client
            self.client = MlflowClient()
            
            # Create or get experiment
            self.experiment_id = self._create_or_get_experiment()
            
            logger.info(f"MLflow setup complete. Experiment ID: {self.experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def _create_or_get_experiment(self) -> str:
        """Create experiment if it doesn't exist, otherwise get existing experiment ID."""
        try:
            experiment = self.client.get_experiment_by_name(self.config.experiment_name)
            if experiment:
                logger.info(f"Using existing experiment: {self.config.experiment_name}")
                return experiment.experiment_id
        except MlflowException:
            pass
        
        # Create new experiment
        experiment_id = self.client.create_experiment(
            name=self.config.experiment_name,
            artifact_location=self.config.artifact_location
        )
        logger.info(f"Created new experiment: {self.config.experiment_name}")
        return experiment_id
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run
            
        Returns:
            Run ID of the started run
        """
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags or {}
        )
        
        # Set the active run
        mlflow.start_run(run_id=run.info.run_id)
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            for key, value in params.items():
                # Convert complex objects to strings
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                elif not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                
                mlflow.log_param(key, value)
            
            logger.debug(f"Logged {len(params)} parameters")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_metrics(self, metrics: ExperimentMetrics, step: Optional[int] = None) -> None:
        """
        Log metrics to the current run.
        
        Args:
            metrics: ExperimentMetrics object containing metrics to log
            step: Optional step number for the metrics
        """
        try:
            metrics_dict = {
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2_score": metrics.r2_score,
                "training_time": metrics.training_time
            }
            
            # Add optional GPU metrics
            if metrics.gpu_utilization is not None:
                metrics_dict["gpu_utilization"] = metrics.gpu_utilization
            if metrics.gpu_memory_used is not None:
                metrics_dict["gpu_memory_used"] = metrics.gpu_memory_used
            if metrics.model_size_mb is not None:
                metrics_dict["model_size_mb"] = metrics.model_size_mb
            
            for key, value in metrics_dict.items():
                mlflow.log_metric(key, value, step=step)
            
            logger.debug(f"Logged {len(metrics_dict)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_artifacts(self, artifacts: ModelArtifacts) -> None:
        """
        Log artifacts to the current run.
        
        Args:
            artifacts: ModelArtifacts object containing artifact paths
        """
        try:
            artifact_count = 0
            
            # Log feature importance plot
            if artifacts.feature_importance_plot and os.path.exists(artifacts.feature_importance_plot):
                mlflow.log_artifact(artifacts.feature_importance_plot, "plots")
                artifact_count += 1
            
            # Log training curves plot
            if artifacts.training_curves_plot and os.path.exists(artifacts.training_curves_plot):
                mlflow.log_artifact(artifacts.training_curves_plot, "plots")
                artifact_count += 1
            
            # Log confusion matrix plot
            if artifacts.confusion_matrix_plot and os.path.exists(artifacts.confusion_matrix_plot):
                mlflow.log_artifact(artifacts.confusion_matrix_plot, "plots")
                artifact_count += 1
            
            # Log model summary
            if artifacts.model_summary and os.path.exists(artifacts.model_summary):
                mlflow.log_artifact(artifacts.model_summary, "model_info")
                artifact_count += 1
            
            logger.debug(f"Logged {artifact_count} artifacts")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            raise
    
    def log_model(self, model: Any, model_type: str, signature=None, input_example=None) -> None:
        """
        Log a model to MLflow with appropriate flavor.
        
        Args:
            model: The trained model object
            model_type: Type of model ('sklearn', 'pytorch', 'xgboost', 'lightgbm')
            signature: Optional model signature
            input_example: Optional input example
        """
        try:
            model_type = model_type.lower()
            
            if model_type == 'sklearn' or model_type == 'cuml':
                mlflow.sklearn.log_model(
                    model, 
                    "model", 
                    signature=signature, 
                    input_example=input_example
                )
            elif model_type == 'pytorch':
                mlflow.pytorch.log_model(
                    model, 
                    "model", 
                    signature=signature, 
                    input_example=input_example
                )
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(
                    model, 
                    "model", 
                    signature=signature, 
                    input_example=input_example
                )
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(
                    model, 
                    "model", 
                    signature=signature, 
                    input_example=input_example
                )
            else:
                # Fallback to generic model logging
                mlflow.log_artifact(str(model), "model")
                logger.warning(f"Unknown model type {model_type}, using generic logging")
            
            logger.info(f"Logged {model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
            raise
    
    def get_experiment_runs(self, max_results: int = 100) -> List[Run]:
        """
        Get runs from the current experiment.
        
        Args:
            max_results: Maximum number of runs to return
            
        Returns:
            List of Run objects
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            return runs
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            raise
    
    def get_best_run(self, metric_name: str = "rmse", ascending: bool = True) -> Optional[Run]:
        """
        Get the best run based on a specific metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether to sort in ascending order (True for minimizing metrics)
            
        Returns:
            Best Run object or None if no runs found
        """
        try:
            order_direction = "ASC" if ascending else "DESC"
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric_name} {order_direction}"],
                max_results=1
            )
            
            if runs:
                logger.info(f"Found best run with {metric_name}: {runs[0].data.metrics.get(metric_name)}")
                return runs[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            raise
    
    def register_model(self, run_id: str, model_name: str, stage: str = "Staging") -> str:
        """
        Register a model in the MLflow Model Registry.
        
        Args:
            run_id: ID of the run containing the model
            model_name: Name for the registered model
            stage: Initial stage for the model version
            
        Returns:
            Model version number
        """
        try:
            # Create model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition to specified stage
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=stage
                )
            
            logger.info(f"Registered model {model_name} version {model_version.version} in stage {stage}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_model_version(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """
        Get the latest model version for a specific stage.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to retrieve
            
        Returns:
            Model version number or None if not found
        """
        try:
            model_versions = self.client.get_latest_versions(
                name=model_name,
                stages=[stage]
            )
            
            if model_versions:
                return model_versions[0].version
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def load_model(self, model_name: str, stage: str = "Production"):
        """
        Load a model from the Model Registry.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to load
            
        Returns:
            Loaded model object
        """
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Loaded model {model_name} from stage {stage}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def cleanup_old_runs(self, keep_last_n: int = 50) -> int:
        """
        Clean up old experiment runs, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent runs to keep
            
        Returns:
            Number of runs deleted
        """
        try:
            runs = self.get_experiment_runs(max_results=1000)
            
            if len(runs) <= keep_last_n:
                logger.info(f"Only {len(runs)} runs found, no cleanup needed")
                return 0
            
            runs_to_delete = runs[keep_last_n:]
            deleted_count = 0
            
            for run in runs_to_delete:
                try:
                    self.client.delete_run(run.info.run_id)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete run {run.info.run_id}: {e}")
            
            logger.info(f"Deleted {deleted_count} old runs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            return 0


def create_mlflow_manager(config: Optional[MLflowConfig] = None) -> MLflowExperimentManager:
    """
    Factory function to create an MLflow experiment manager.
    
    Args:
        config: Optional MLflow configuration. If None, uses default config.
        
    Returns:
        MLflowExperimentManager instance
    """
    if config is None:
        config = MLflowConfig()
    
    return MLflowExperimentManager(config)