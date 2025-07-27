"""
Prediction API Endpoints

This module provides FastAPI endpoints for single and batch predictions
with comprehensive input validation, error handling, and logging.
"""

import time
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import (
    HousingPredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, ModelInfo, PredictionError, ValidationErrorResponse,
    PredictionStatus, ModelStage, ValidationErrorType
)
from .database import DatabaseManager, PredictionLogData, get_database_manager
from .model_loader import ModelLoader, get_model_loader
from .metrics import PrometheusMetrics, get_metrics
from .config import APIConfig, get_api_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])


def get_client_info(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract client information from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Tuple of (user_agent, ip_address)
    """
    user_agent = request.headers.get("user-agent")
    
    # Get IP address, considering proxy headers
    ip_address = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip() or
        request.headers.get("x-real-ip") or
        (request.client.host if request.client else None)
    )
    
    return user_agent, ip_address


def create_prediction_response(
    prediction_value: float,
    model_info: Any,
    processing_time_ms: float,
    request_id: str,
    confidence_interval: Optional[Tuple[float, float]] = None,
    confidence_score: Optional[float] = None,
    warnings: Optional[List[str]] = None
) -> PredictionResponse:
    """
    Create prediction response object.
    
    Args:
        prediction_value: Predicted value
        model_info: Model information object
        processing_time_ms: Processing time in milliseconds
        request_id: Request identifier
        confidence_interval: Optional confidence interval
        confidence_score: Optional confidence score
        warnings: Optional validation warnings
        
    Returns:
        PredictionResponse object
    """
    # Extract model performance metrics
    performance_metrics = {}
    if hasattr(model_info, 'performance_metrics'):
        performance_metrics = model_info.performance_metrics
    
    # Create model info dictionary
    model_info_dict = {
        "algorithm": getattr(model_info, 'model_type', 'unknown'),
        "training_date": getattr(model_info, 'load_time', datetime.utcnow()).isoformat(),
        "performance_metrics": performance_metrics
    }
    
    return PredictionResponse(
        prediction=prediction_value,
        model_version=getattr(model_info, 'version', 'unknown'),
        model_stage=ModelStage(getattr(model_info, 'stage', 'None')),
        confidence_interval=confidence_interval,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
        request_id=request_id,
        timestamp=datetime.utcnow(),
        features_used=len(getattr(model_info, 'features', [])),
        model_info=model_info_dict,
        warnings=warnings
    )


def make_prediction(
    model: Any,
    features: Dict[str, float],
    feature_names: List[str]
) -> Tuple[float, Optional[Tuple[float, float]], Optional[float]]:
    """
    Make prediction using the loaded model.
    
    Args:
        model: Loaded model object
        features: Feature dictionary
        feature_names: Expected feature names
        
    Returns:
        Tuple of (prediction, confidence_interval, confidence_score)
        
    Raises:
        ValueError: If prediction fails
    """
    try:
        # Create DataFrame with features in correct order
        feature_data = pd.DataFrame([{
            name: features[name] for name in feature_names
        }])
        
        # Make prediction
        prediction = model.predict(feature_data)
        
        # Extract prediction value
        if isinstance(prediction, np.ndarray):
            prediction_value = float(prediction[0])
        elif isinstance(prediction, (list, tuple)):
            prediction_value = float(prediction[0])
        else:
            prediction_value = float(prediction)
        
        # Validate prediction range
        if not (0.01 <= prediction_value <= 20.0):
            logger.warning(f"Prediction {prediction_value} outside expected range")
        
        # TODO: Add confidence interval calculation if model supports it
        confidence_interval = None
        confidence_score = None
        
        return prediction_value, confidence_interval, confidence_score
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError(f"Model prediction failed: {str(e)}")


@router.post("/", response_model=PredictionResponse)
async def predict_single(
    request: HousingPredictionRequest,
    http_request: Request
) -> PredictionResponse:
    """
    Make a single housing price prediction.
    
    This endpoint accepts housing features and returns a price prediction
    with comprehensive validation, error handling, and logging.
    
    Args:
        request: Housing prediction request
        http_request: FastAPI HTTP request object
        model_loader: Model loader dependency
        database_manager: Database manager dependency
        metrics: Metrics collector dependency
        config: API configuration dependency
        
    Returns:
        Prediction response with price and metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    logger.info(f"Processing single prediction request: {request_id}")
    
    # Initialize variables for exception handling
    model_loader = None
    database_manager = None
    metrics = None
    config = None
    model_info = None
    features = {}
    user_agent = None
    ip_address = None
    
    try:
        # Get dependencies from app state
        model_loader = getattr(http_request.app.state, 'model_loader', None)
        database_manager = getattr(http_request.app.state, 'database_manager', None)
        metrics = getattr(http_request.app.state, 'metrics', None)
        config = getattr(http_request.app.state, 'api_config', None)
        
        # Fallback to global instances if app state is not available (e.g., during testing)
        if model_loader is None:
            model_loader = get_model_loader()
        if database_manager is None:
            database_manager = get_database_manager()
        if metrics is None:
            metrics = get_metrics()
        if config is None:
            config = get_api_config()
        
        # Get client information
        user_agent, ip_address = get_client_info(http_request)
        
        # Get current model
        model, model_info = model_loader.get_current_model()
        
        if model is None or model_info is None:
            logger.error("No model available for prediction")
            if metrics:
                metrics.record_error("model_not_available", "predict_single")
            raise HTTPException(
                status_code=503,
                detail="Model service is currently unavailable"
            )
        
        # Use specified model version if provided
        if request.model_version and request.model_version != model_info.version:
            try:
                model, model_info = model_loader.load_model(
                    model_name=config.model_name,
                    stage=request.model_version,
                    use_cache=True
                )
            except Exception as e:
                logger.warning(f"Failed to load requested model version {request.model_version}: {e}")
                # Continue with current model
        
        # Prepare features
        features = {
            "MedInc": request.MedInc,
            "HouseAge": request.HouseAge,
            "AveRooms": request.AveRooms,
            "AveBedrms": request.AveBedrms,
            "Population": request.Population,
            "AveOccup": request.AveOccup,
            "Latitude": request.Latitude,
            "Longitude": request.Longitude
        }
        
        # Make prediction
        prediction_value, confidence_interval, confidence_score = make_prediction(
            model, features, model_info.features
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = create_prediction_response(
            prediction_value=prediction_value,
            model_info=model_info,
            processing_time_ms=processing_time_ms,
            request_id=request_id,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score
        )
        
        # Log prediction to database
        prediction_log_data = PredictionLogData(
            request_id=request_id,
            model_version=model_info.version,
            model_stage=model_info.stage,
            input_features=features,
            prediction=prediction_value,
            confidence_lower=confidence_interval[0] if confidence_interval else None,
            confidence_upper=confidence_interval[1] if confidence_interval else None,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            user_agent=user_agent,
            ip_address=ip_address,
            status="success"
        )
        
        # Log to database (non-blocking)
        try:
            database_manager.log_prediction(prediction_log_data)
        except Exception as e:
            logger.error(f"Failed to log prediction to database: {e}")
            # Don't fail the request if logging fails
        
        # Record metrics
        if metrics:
            metrics.record_prediction(processing_time_ms)
        
        logger.info(f"Single prediction completed: {request_id}, value: {prediction_value:.4f}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle prediction errors
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.error(f"Prediction error for request {request_id}: {e}")
        
        # Log error to database
        try:
            error_log_data = PredictionLogData(
                request_id=request_id,
                model_version=model_info.version if model_info else "unknown",
                model_stage=model_info.stage if model_info else "unknown",
                input_features=features if 'features' in locals() else {},
                prediction=0.0,  # Placeholder for failed prediction
                processing_time_ms=processing_time_ms,
                user_agent=user_agent,
                ip_address=ip_address,
                status="error",
                error_message=str(e)
            )
            database_manager.log_prediction(error_log_data)
        except Exception as log_error:
            logger.error(f"Failed to log error to database: {log_error}")
        
        # Record error metrics
        if metrics:
            metrics.record_error("prediction_failed", "predict_single")
        
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        # Handle unexpected errors
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.error(f"Unexpected error in single prediction {request_id}: {e}")
        
        # Record error metrics
        if metrics:
            metrics.record_error("internal_error", "predict_single")
        
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction"
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    http_request: Request
) -> BatchPredictionResponse:
    """
    Make batch housing price predictions.
    
    This endpoint accepts multiple housing feature sets and returns
    predictions for each with comprehensive error handling and logging.
    
    Args:
        request: Batch prediction request
        http_request: FastAPI HTTP request object
        model_loader: Model loader dependency
        database_manager: Database manager dependency
        metrics: Metrics collector dependency
        config: API configuration dependency
        
    Returns:
        Batch prediction response with results and statistics
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    batch_id = request.batch_id or str(uuid.uuid4())
    
    logger.info(f"Processing batch prediction request: {batch_id}, size: {len(request.predictions)}")
    
    # Initialize variables for exception handling
    model_loader = None
    database_manager = None
    metrics = None
    config = None
    model_info = None
    user_agent = None
    ip_address = None
    
    try:
        # Get dependencies from app state
        model_loader = getattr(http_request.app.state, 'model_loader', None)
        database_manager = getattr(http_request.app.state, 'database_manager', None)
        metrics = getattr(http_request.app.state, 'metrics', None)
        config = getattr(http_request.app.state, 'api_config', None)
        
        # Fallback to global instances if app state is not available (e.g., during testing)
        if model_loader is None:
            model_loader = get_model_loader()
        if database_manager is None:
            database_manager = get_database_manager()
        if metrics is None:
            metrics = get_metrics()
        if config is None:
            config = get_api_config()
        
        # Get client information
        user_agent, ip_address = get_client_info(http_request)
        
        # Get current model
        model, model_info = model_loader.get_current_model()
        
        if model is None or model_info is None:
            logger.error("No model available for batch prediction")
            if metrics:
                metrics.record_error("model_not_available", "predict_batch")
            raise HTTPException(
                status_code=503,
                detail="Model service is currently unavailable"
            )
        
        # Use specified model version if provided
        if request.model_version and request.model_version != model_info.version:
            try:
                model, model_info = model_loader.load_model(
                    model_name=config.model_name,
                    stage=request.model_version,
                    use_cache=True
                )
            except Exception as e:
                logger.warning(f"Failed to load requested model version {request.model_version}: {e}")
                # Continue with current model
        
        # Process predictions
        predictions = []
        successful_predictions = 0
        failed_predictions = 0
        errors_summary = {}
        warnings = []
        
        for i, pred_request in enumerate(request.predictions):
            pred_start_time = time.time()
            pred_request_id = f"{batch_id}_{i}"
            
            try:
                # Prepare features
                features = {
                    "MedInc": pred_request.MedInc,
                    "HouseAge": pred_request.HouseAge,
                    "AveRooms": pred_request.AveRooms,
                    "AveBedrms": pred_request.AveBedrms,
                    "Population": pred_request.Population,
                    "AveOccup": pred_request.AveOccup,
                    "Latitude": pred_request.Latitude,
                    "Longitude": pred_request.Longitude
                }
                
                # Make prediction
                prediction_value, confidence_interval, confidence_score = make_prediction(
                    model, features, model_info.features
                )
                
                # Calculate processing time for this prediction
                pred_processing_time_ms = (time.time() - pred_start_time) * 1000
                
                # Create response
                pred_response = create_prediction_response(
                    prediction_value=prediction_value,
                    model_info=model_info,
                    processing_time_ms=pred_processing_time_ms,
                    request_id=pred_request_id,
                    confidence_interval=confidence_interval if request.return_confidence else None,
                    confidence_score=confidence_score if request.return_confidence else None
                )
                
                predictions.append(pred_response)
                successful_predictions += 1
                
                # Log prediction to database
                prediction_log_data = PredictionLogData(
                    request_id=pred_request_id,
                    model_version=model_info.version,
                    model_stage=model_info.stage,
                    input_features=features,
                    prediction=prediction_value,
                    confidence_lower=confidence_interval[0] if confidence_interval else None,
                    confidence_upper=confidence_interval[1] if confidence_interval else None,
                    confidence_score=confidence_score,
                    processing_time_ms=pred_processing_time_ms,
                    user_agent=user_agent,
                    ip_address=ip_address,
                    batch_id=batch_id,
                    status="success"
                )
                
                # Log to database (non-blocking)
                try:
                    database_manager.log_prediction(prediction_log_data)
                except Exception as e:
                    logger.error(f"Failed to log prediction {pred_request_id} to database: {e}")
                
            except Exception as e:
                # Handle individual prediction error
                pred_processing_time_ms = (time.time() - pred_start_time) * 1000
                failed_predictions += 1
                
                error_type = type(e).__name__
                errors_summary[error_type] = errors_summary.get(error_type, 0) + 1
                
                logger.warning(f"Prediction failed for item {i} in batch {batch_id}: {e}")
                
                # Create error response
                error_response = PredictionError(
                    error_type=ValidationErrorType.FIELD_CONSTRAINT,
                    error_code="PREDICTION_FAILED",
                    message=str(e),
                    request_id=pred_request_id,
                    timestamp=datetime.utcnow()
                )
                
                predictions.append(error_response)
                
                # Log error to database
                try:
                    error_log_data = PredictionLogData(
                        request_id=pred_request_id,
                        model_version=model_info.version,
                        model_stage=model_info.stage,
                        input_features=features if 'features' in locals() else {},
                        prediction=0.0,  # Placeholder for failed prediction
                        processing_time_ms=pred_processing_time_ms,
                        user_agent=user_agent,
                        ip_address=ip_address,
                        batch_id=batch_id,
                        status="error",
                        error_message=str(e)
                    )
                    database_manager.log_prediction(error_log_data)
                except Exception as log_error:
                    logger.error(f"Failed to log error to database: {log_error}")
        
        # Calculate total processing time
        total_processing_time_ms = (time.time() - start_time) * 1000
        average_processing_time_ms = total_processing_time_ms / len(request.predictions) if request.predictions else 0.0
        
        # Determine overall status
        if failed_predictions == 0:
            status = PredictionStatus.SUCCESS
        elif successful_predictions == 0:
            status = PredictionStatus.ERROR
        else:
            status = PredictionStatus.PARTIAL_SUCCESS
        
        # Create batch response
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_predictions=len(request.predictions),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            total_processing_time_ms=total_processing_time_ms,
            average_processing_time_ms=average_processing_time_ms,
            status=status,
            timestamp=datetime.utcnow(),
            warnings=warnings if warnings else None,
            errors_summary=errors_summary if errors_summary else None
        )
        
        # Record metrics
        if metrics:
            metrics.record_batch_prediction(
                batch_size=len(request.predictions),
                successful_predictions=successful_predictions,
                failed_predictions=failed_predictions,
                total_processing_time_ms=total_processing_time_ms
            )
        
        logger.info(
            f"Batch prediction completed: {batch_id}, "
            f"successful: {successful_predictions}, failed: {failed_predictions}"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        total_processing_time_ms = (time.time() - start_time) * 1000
        
        logger.error(f"Unexpected error in batch prediction {batch_id}: {e}")
        
        # Record error metrics
        if metrics:
            metrics.record_error("internal_error", "predict_batch")
        
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction"
        )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info(
    request: Request
) -> ModelInfo:
    """
    Get information about the currently loaded model.
    
    This endpoint returns comprehensive metadata about the model
    including performance metrics, features, and technical details.
    
    Args:
        model_loader: Model loader dependency
        metrics: Metrics collector dependency
        
    Returns:
        Model information and metadata
        
    Raises:
        HTTPException: If no model is available
    """
    logger.info("Getting model information")
    
    # Initialize variables for exception handling
    model_loader = None
    metrics = None
    model_info = None
    
    try:
        # Get dependencies from app state
        model_loader = getattr(request.app.state, 'model_loader', None)
        metrics = getattr(request.app.state, 'metrics', None)
        
        # Fallback to global instances if app state is not available (e.g., during testing)
        if model_loader is None:
            model_loader = get_model_loader()
        if metrics is None:
            metrics = get_metrics()
        
        # Get current model
        model, model_info = model_loader.get_current_model()
        
        if model is None or model_info is None:
            logger.error("No model available for info request")
            if metrics:
                metrics.record_error("model_not_available", "get_model_info")
            raise HTTPException(
                status_code=503,
                detail="No model is currently loaded"
            )
        
        # Calculate model size (approximate)
        model_size_mb = None
        try:
            # This is a rough estimate - actual implementation would depend on model type
            model_size_mb = 2.5  # Placeholder value
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
        
        # Create model info response
        response = ModelInfo(
            name=model_info.name,
            version=model_info.version,
            stage=ModelStage(model_info.stage),
            algorithm=model_info.model_type,
            framework="mlflow",  # Since we're using MLflow pyfunc
            training_date=model_info.load_time,
            features=model_info.features,
            performance_metrics=model_info.performance_metrics,
            model_size_mb=model_size_mb,
            gpu_accelerated=True,  # Based on our GPU-accelerated training
            last_updated=model_info.load_time,
            description=f"California Housing price prediction model using {model_info.model_type}",
            tags={
                "model_uri": model_info.model_uri,
                "run_id": model_info.run_id
            }
        )
        
        logger.info(f"Model info retrieved: {model_info.name} v{model_info.version}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error getting model info: {e}")
        
        # Record error metrics
        if metrics:
            metrics.record_error("internal_error", "get_model_info")
        
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while retrieving model information"
        )