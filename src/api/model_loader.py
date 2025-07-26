"""
Model Loading Utilities with MLflow Integration

This module provides utilities for loading models from MLflow Model Registry
with fallback mechanisms, caching, and comprehensive error handling.
"""

import os
import time
import logging
import threading
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from .config import APIConfig, ModelConfig
from .metrics import PrometheusMetrics
from src.mlflow_config import MLflowExperimentManager, MLflowConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    version: str
    stage: str
    model_type: str
    features: List[str]
    performance_metrics: Dict[str, float]
    load_time: datetime
    model_uri: str
    run_id: Optional[str] = None


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelCache:
    """
    Thread-safe model cache with TTL support.
    
    This class provides caching for loaded models to avoid repeated
    loading from MLflow Model Registry.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize model cache.
        
        Args:
            ttl_seconds: Time-to-live for cached models in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get model from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached model or None if not found/expired
        """
        with self._lock:
            if key in self._cache:
                model, load_time = self._cache[key]
                if datetime.now() - load_time < timedelta(seconds=self.ttl_seconds):
                    logger.debug(f"Cache hit for model key: {key}")
                    return model
                else:
                    # Remove expired entry
                    del self._cache[key]
                    logger.debug(f"Cache expired for model key: {key}")
            
            return None
    
    def put(self, key: str, model: Any) -> None:
        """
        Put model in cache.
        
        Args:
            key: Cache key
            model: Model to cache
        """
        with self._lock:
            self._cache[key] = (model, datetime.now())
            logger.debug(f"Cached model with key: {key}")
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            logger.info("Model cache cleared")
    
    def size(self) -> int:
        """Get number of cached models."""
        with self._lock:
            return len(self._cache)


class ModelLoader:
    """
    Model loader with MLflow integration and comprehensive fallback mechanisms.
    
    This class handles loading models from MLflow Model Registry with
    caching, fallback stages, and performance monitoring.
    """
    
    def __init__(self, api_config: APIConfig, model_config: ModelConfig, 
                 metrics: Optional[PrometheusMetrics] = None):
        """
        Initialize model loader.
        
        Args:
            api_config: API configuration
            model_config: Model configuration
            metrics: Optional Prometheus metrics collector
        """
        self.api_config = api_config
        self.model_config = model_config
        self.metrics = metrics
        
        # Initialize MLflow
        self.mlflow_config = MLflowConfig(
            tracking_uri=api_config.mlflow_tracking_uri,
            registry_uri=api_config.mlflow_registry_uri
        )
        self.mlflow_manager = MLflowExperimentManager(self.mlflow_config)
        self.client = self.mlflow_manager.client
        
        # Initialize cache
        self.cache = ModelCache(ttl_seconds=3600)  # 1 hour TTL
        
        # Current model info
        self._current_model: Optional[Any] = None
        self._current_model_info: Optional[ModelInfo] = None
        self._model_lock = threading.RLock()
        
        logger.info("Model loader initialized")
    
    def _get_cache_key(self, model_name: str, stage: str) -> str:
        """Generate cache key for model."""
        return f"{model_name}:{stage}"
    
    def _get_model_info_from_registry(self, model_name: str, stage: str) -> Optional[Dict[str, Any]]:
        """
        Get model information from MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            model_versions = self.client.get_latest_versions(
                name=model_name,
                stages=[stage]
            )
            
            if not model_versions:
                return None
            
            model_version = model_versions[0]
            
            # Get run information
            run = self.client.get_run(model_version.run_id)
            
            return {
                'name': model_name,
                'version': model_version.version,
                'stage': stage,
                'run_id': model_version.run_id,
                'model_uri': f"models:/{model_name}/{stage}",
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags
            }
            
        except MlflowException as e:
            logger.error(f"Failed to get model info from registry: {e}")
            return None
    
    def _load_model_from_uri(self, model_uri: str) -> Any:
        """
        Load model from MLflow URI.
        
        Args:
            model_uri: MLflow model URI
            
        Returns:
            Loaded model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            start_time = time.time()
            
            # Load model using pyfunc for universal interface
            model = mlflow.pyfunc.load_model(model_uri)
            
            load_duration = time.time() - start_time
            
            if self.metrics:
                self.metrics.record_model_load_time(load_duration)
            
            logger.info(f"Model loaded from {model_uri} in {load_duration:.2f}s")
            return model
            
        except Exception as e:
            error_msg = f"Failed to load model from {model_uri}: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def _validate_model(self, model: Any, model_info: Dict[str, Any]) -> bool:
        """
        Validate loaded model.
        
        Args:
            model: Loaded model
            model_info: Model information
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Create sample input for validation
            sample_input = pd.DataFrame([{
                feature: 1.0 for feature in self.model_config.feature_names
            }])
            
            # Try to make a prediction
            prediction = model.predict(sample_input)
            
            # Check if prediction is reasonable
            if isinstance(prediction, np.ndarray):
                prediction_value = float(prediction[0])
            else:
                prediction_value = float(prediction)
            
            # Validate prediction range (California housing prices typically 0.5-5.0)
            if not (0.1 <= prediction_value <= 10.0):
                logger.warning(f"Model prediction {prediction_value} outside expected range")
                return False
            
            # Check model performance metrics if available
            metrics = model_info.get('metrics', {})
            thresholds = self.model_config.performance_thresholds
            
            if 'r2_score' in metrics and metrics['r2_score'] < thresholds['min_r2_score']:
                logger.warning(f"Model R² score {metrics['r2_score']} below threshold {thresholds['min_r2_score']}")
                return False
            
            if 'rmse' in metrics and metrics['rmse'] > thresholds['max_rmse']:
                logger.warning(f"Model RMSE {metrics['rmse']} above threshold {thresholds['max_rmse']}")
                return False
            
            logger.info("Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def load_model(self, model_name: Optional[str] = None, stage: Optional[str] = None,
                   use_cache: bool = True, validate: bool = True) -> Tuple[Any, ModelInfo]:
        """
        Load model from MLflow Model Registry with fallback mechanisms.
        
        Args:
            model_name: Name of the model to load (defaults to config)
            stage: Stage of the model to load (defaults to config)
            use_cache: Whether to use cached models
            validate: Whether to validate the loaded model
            
        Returns:
            Tuple of (model, model_info)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        model_name = model_name or self.api_config.model_name
        stage = stage or self.api_config.model_stage
        
        logger.info(f"Loading model {model_name} from stage {stage}")
        
        if self.metrics:
            self.metrics.set_model_status('loading')
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(model_name, stage)
            if use_cache:
                cached_model = self.cache.get(cache_key)
                if cached_model is not None:
                    model, model_info = cached_model
                    logger.info(f"Using cached model {model_name}:{stage}")
                    
                    if self.metrics:
                        self.metrics.set_model_status('ready')
                        self.metrics.set_model_info(
                            model_info.name, model_info.version, model_info.stage,
                            model_info.model_type, model_info.features
                        )
                    
                    return model, model_info
            
            # Try to load from primary stage
            stages_to_try = [stage]
            
            # Add fallback stage if different from primary
            if stage != self.api_config.model_fallback_stage:
                stages_to_try.append(self.api_config.model_fallback_stage)
            
            # Add additional fallback stages
            additional_stages = ['Staging', 'Production', 'None']
            for fallback_stage in additional_stages:
                if fallback_stage not in stages_to_try:
                    stages_to_try.append(fallback_stage)
            
            last_error = None
            
            for current_stage in stages_to_try:
                try:
                    logger.info(f"Attempting to load model from stage: {current_stage}")
                    
                    # Get model info from registry
                    model_info_dict = self._get_model_info_from_registry(model_name, current_stage)
                    if not model_info_dict:
                        logger.warning(f"No model found in stage {current_stage}")
                        continue
                    
                    # Load model
                    model = self._load_model_from_uri(model_info_dict['model_uri'])
                    
                    # Validate model if requested
                    if validate and not self._validate_model(model, model_info_dict):
                        logger.warning(f"Model validation failed for stage {current_stage}")
                        continue
                    
                    # Create model info object
                    model_info = ModelInfo(
                        name=model_info_dict['name'],
                        version=model_info_dict['version'],
                        stage=current_stage,
                        model_type=model_info_dict.get('tags', {}).get('model_type', 'unknown'),
                        features=self.model_config.feature_names,
                        performance_metrics=model_info_dict['metrics'],
                        load_time=datetime.now(),
                        model_uri=model_info_dict['model_uri'],
                        run_id=model_info_dict['run_id']
                    )
                    
                    # Cache the model
                    if use_cache:
                        self.cache.put(cache_key, (model, model_info))
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.set_model_status('ready')
                        self.metrics.set_model_info(
                            model_info.name, model_info.version, model_info.stage,
                            model_info.model_type, model_info.features
                        )
                    
                    if current_stage != stage:
                        logger.warning(f"Using fallback model from stage {current_stage} instead of {stage}")
                    
                    logger.info(f"Successfully loaded model {model_name} v{model_info.version} from {current_stage}")
                    return model, model_info
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load model from stage {current_stage}: {e}")
                    continue
            
            # If we get here, all attempts failed
            error_msg = f"Failed to load model {model_name} from any stage. Last error: {last_error}"
            logger.error(error_msg)
            
            if self.metrics:
                self.metrics.set_model_status('error')
                self.metrics.record_error('model_load_failed', 'model_loader')
            
            raise ModelLoadError(error_msg)
            
        except Exception as e:
            if self.metrics:
                self.metrics.set_model_status('error')
                self.metrics.record_error('model_load_failed', 'model_loader')
            
            if isinstance(e, ModelLoadError):
                raise
            else:
                raise ModelLoadError(f"Unexpected error loading model: {e}") from e
    
    def get_current_model(self) -> Tuple[Optional[Any], Optional[ModelInfo]]:
        """
        Get currently loaded model.
        
        Returns:
            Tuple of (model, model_info) or (None, None) if no model loaded
        """
        with self._model_lock:
            return self._current_model, self._current_model_info
    
    def set_current_model(self, model: Any, model_info: ModelInfo) -> None:
        """
        Set current model.
        
        Args:
            model: Model object
            model_info: Model information
        """
        with self._model_lock:
            self._current_model = model
            self._current_model_info = model_info
            logger.info(f"Set current model to {model_info.name} v{model_info.version}")
    
    def reload_model(self, force: bool = False) -> Tuple[Any, ModelInfo]:
        """
        Reload current model.
        
        Args:
            force: Whether to force reload even if current model exists
            
        Returns:
            Tuple of (model, model_info)
        """
        with self._model_lock:
            if not force and self._current_model is not None:
                logger.info("Current model exists, skipping reload")
                return self._current_model, self._current_model_info
            
            # Clear cache to force fresh load
            if force:
                self.cache.clear()
            
            # Load model
            model, model_info = self.load_model(use_cache=not force)
            
            # Set as current model
            self.set_current_model(model, model_info)
            
            return model, model_info
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in the registry.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = []
            
            # Get all registered models
            registered_models = self.client.list_registered_models()
            
            for registered_model in registered_models:
                model_name = registered_model.name
                
                # Get all versions for this model
                model_versions = self.client.get_latest_versions(model_name)
                
                for version in model_versions:
                    try:
                        run = self.client.get_run(version.run_id)
                        
                        models.append({
                            'name': model_name,
                            'version': version.version,
                            'stage': version.current_stage,
                            'run_id': version.run_id,
                            'metrics': run.data.metrics,
                            'tags': run.data.tags,
                            'creation_time': version.creation_timestamp,
                            'last_updated': version.last_updated_timestamp
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get info for model {model_name} v{version.version}: {e}")
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return []
    
    def get_model_performance(self, model_name: str, stage: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific model.
        
        Args:
            model_name: Name of the model
            stage: Model stage
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            model_info = self._get_model_info_from_registry(model_name, stage)
            if model_info:
                return model_info.get('metrics', {})
            return {}
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return {}


# Global model loader instance
_model_loader_instance: Optional[ModelLoader] = None


def get_model_loader(api_config: Optional[APIConfig] = None, 
                    model_config: Optional[ModelConfig] = None,
                    metrics: Optional[PrometheusMetrics] = None) -> ModelLoader:
    """
    Get global model loader instance.
    
    Args:
        api_config: Optional API configuration
        model_config: Optional model configuration
        metrics: Optional Prometheus metrics
        
    Returns:
        ModelLoader instance
    """
    global _model_loader_instance
    
    if _model_loader_instance is None:
        from .config import get_api_config, get_model_config
        
        api_config = api_config or get_api_config()
        model_config = model_config or get_model_config()
        
        _model_loader_instance = ModelLoader(api_config, model_config, metrics)
    
    return _model_loader_instance


def initialize_model_loader(api_config: APIConfig, model_config: ModelConfig,
                          metrics: Optional[PrometheusMetrics] = None,
                          load_model_on_init: bool = True) -> ModelLoader:
    """
    Initialize global model loader instance.
    
    Args:
        api_config: API configuration
        model_config: Model configuration
        metrics: Optional Prometheus metrics
        load_model_on_init: Whether to load model immediately
        
    Returns:
        ModelLoader instance
    """
    global _model_loader_instance
    
    _model_loader_instance = ModelLoader(api_config, model_config, metrics)
    
    if load_model_on_init:
        try:
            model, model_info = _model_loader_instance.load_model()
            _model_loader_instance.set_current_model(model, model_info)
            logger.info("Model loaded successfully during initialization")
        except Exception as e:
            logger.error(f"Failed to load model during initialization: {e}")
            # Don't raise here - let the service start and handle model loading errors gracefully
    
    return _model_loader_instance