"""
Database API Endpoints

This module provides FastAPI endpoints for database operations including
prediction history browsing, filtering, search, and data export functionality.
"""

import logging
import time
import csv
import json
import io
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .database import DatabaseManager, PredictionLog, get_database_manager
from .metrics import PrometheusMetrics, get_metrics
from .config import APIConfig, get_api_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["database"])


class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history."""
    
    id: int
    request_id: str
    model_version: str
    model_stage: str
    input_features: Dict[str, Any]
    prediction: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    confidence_score: Optional[float] = None
    processing_time_ms: float
    timestamp: datetime
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    batch_id: Optional[str] = None
    status: str
    error_message: Optional[str] = None


class PredictionHistoryListResponse(BaseModel):
    """Response model for paginated prediction history."""
    
    predictions: List[PredictionHistoryResponse]
    total_count: int
    page: int
    limit: int
    has_next: bool
    has_previous: bool


class PredictionStatsResponse(BaseModel):
    """Response model for prediction statistics."""
    
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    success_rate: float
    average_processing_time_ms: float
    date_range: Dict[str, Optional[str]]


class FilterParams(BaseModel):
    """Filter parameters for prediction history."""
    
    model_version: Optional[str] = None
    status: Optional[str] = None
    batch_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search_term: Optional[str] = None


def convert_prediction_log_to_response(log: PredictionLog) -> PredictionHistoryResponse:
    """Convert PredictionLog database model to response model."""
    return PredictionHistoryResponse(
        id=log.id,
        request_id=log.request_id,
        model_version=log.model_version,
        model_stage=log.model_stage,
        input_features=log.input_features,
        prediction=log.prediction,
        confidence_lower=log.confidence_lower,
        confidence_upper=log.confidence_upper,
        confidence_score=log.confidence_score,
        processing_time_ms=log.processing_time_ms,
        timestamp=log.timestamp,
        user_agent=log.user_agent,
        ip_address=log.ip_address,
        batch_id=log.batch_id,
        status=log.status,
        error_message=log.error_message
    )


@router.get("/predictions", response_model=PredictionHistoryListResponse)
async def get_prediction_history(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=1000, description="Number of records per page"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    status: Optional[str] = Query(None, description="Filter by status (success/error)"),
    batch_id: Optional[str] = Query(None, description="Filter by batch ID"),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format)"),
    search_term: Optional[str] = Query(None, description="Search term for request ID")
) -> PredictionHistoryListResponse:
    """
    Get paginated prediction history with filtering and search capabilities.
    
    This endpoint provides comprehensive access to prediction logs with support for:
    - Pagination with configurable page size
    - Filtering by model version, status, batch ID, and date range
    - Search functionality for request IDs
    - Sorting by timestamp (newest first)
    
    Args:
        request: FastAPI request object
        page: Page number (1-based)
        limit: Number of records per page
        model_version: Optional model version filter
        status: Optional status filter (success/error)
        batch_id: Optional batch ID filter
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
        search_term: Optional search term for request ID
        
    Returns:
        Paginated prediction history with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    logger.info(f"Getting prediction history: page={page}, limit={limit}")
    
    try:
        # Get dependencies
        database_manager = getattr(request.app.state, 'database_manager', None)
        metrics = getattr(request.app.state, 'metrics', None)
        
        if database_manager is None:
            database_manager = get_database_manager()
        if metrics is None:
            metrics = get_metrics()
        
        # Parse date filters
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid start_date format: {start_date}. Use ISO format."
                )
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid end_date format: {end_date}. Use ISO format."
                )
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get predictions with filtering
        predictions = database_manager.get_predictions_filtered(
            limit=limit,
            offset=offset,
            model_version=model_version,
            status=status,
            batch_id=batch_id,
            start_time=start_datetime,
            end_time=end_datetime,
            search_term=search_term
        )
        
        # Get total count for pagination
        total_count = database_manager.get_predictions_count(
            model_version=model_version,
            status=status,
            batch_id=batch_id,
            start_time=start_datetime,
            end_time=end_datetime,
            search_term=search_term
        )
        
        # Convert to response models
        prediction_responses = [
            convert_prediction_log_to_response(pred) for pred in predictions
        ]
        
        # Calculate pagination metadata
        has_next = (offset + limit) < total_count
        has_previous = page > 1
        
        # Create response
        response = PredictionHistoryListResponse(
            predictions=prediction_responses,
            total_count=total_count,
            page=page,
            limit=limit,
            has_next=has_next,
            has_previous=has_previous
        )
        
        # Record metrics
        processing_time_ms = (time.time() - start_time) * 1000
        if metrics:
            metrics.record_database_query("get_prediction_history", processing_time_ms)
        
        logger.info(f"Retrieved {len(predictions)} predictions (page {page}/{(total_count + limit - 1) // limit})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve prediction history"
        )


@router.get("/predictions/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format)")
) -> PredictionStatsResponse:
    """
    Get prediction statistics and metrics.
    
    This endpoint provides comprehensive statistics about predictions including:
    - Total prediction counts
    - Success/failure rates
    - Average processing times
    - Date range information
    
    Args:
        request: FastAPI request object
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
        
    Returns:
        Prediction statistics and metrics
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    logger.info("Getting prediction statistics")
    
    try:
        # Get dependencies
        database_manager = getattr(request.app.state, 'database_manager', None)
        metrics = getattr(request.app.state, 'metrics', None)
        
        if database_manager is None:
            database_manager = get_database_manager()
        if metrics is None:
            metrics = get_metrics()
        
        # Parse date filters
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid start_date format: {start_date}. Use ISO format."
                )
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid end_date format: {end_date}. Use ISO format."
                )
        
        # Get statistics
        stats = database_manager.get_prediction_stats(
            start_time=start_datetime,
            end_time=end_datetime
        )
        
        # Create response
        response = PredictionStatsResponse(
            total_predictions=stats.get("total_predictions", 0),
            successful_predictions=stats.get("successful_predictions", 0),
            failed_predictions=stats.get("failed_predictions", 0),
            success_rate=stats.get("success_rate", 0.0),
            average_processing_time_ms=stats.get("average_processing_time_ms", 0.0),
            date_range={
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        # Record metrics
        processing_time_ms = (time.time() - start_time) * 1000
        if metrics:
            metrics.record_database_query("get_prediction_stats", processing_time_ms)
        
        logger.info(f"Retrieved prediction stats: {stats.get('total_predictions', 0)} total predictions")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve prediction statistics"
        )


@router.get("/predictions/export")
async def export_predictions(
    request: Request,
    format: str = Query("csv", regex="^(csv|json)$", description="Export format (csv or json)"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    status: Optional[str] = Query(None, description="Filter by status (success/error)"),
    batch_id: Optional[str] = Query(None, description="Filter by batch ID"),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format)"),
    search_term: Optional[str] = Query(None, description="Search term for request ID"),
    limit: int = Query(10000, ge=1, le=50000, description="Maximum number of records to export")
) -> StreamingResponse:
    """
    Export prediction data in CSV or JSON format.
    
    This endpoint allows exporting prediction history with the same filtering
    capabilities as the history endpoint. Supports both CSV and JSON formats
    with proper content headers for download.
    
    Args:
        request: FastAPI request object
        format: Export format (csv or json)
        model_version: Optional model version filter
        status: Optional status filter (success/error)
        batch_id: Optional batch ID filter
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
        search_term: Optional search term for request ID
        limit: Maximum number of records to export
        
    Returns:
        StreamingResponse with exported data
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    logger.info(f"Exporting predictions: format={format}, limit={limit}")
    
    try:
        # Get dependencies
        database_manager = getattr(request.app.state, 'database_manager', None)
        metrics = getattr(request.app.state, 'metrics', None)
        
        if database_manager is None:
            database_manager = get_database_manager()
        if metrics is None:
            metrics = get_metrics()
        
        # Parse date filters
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid start_date format: {start_date}. Use ISO format."
                )
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid end_date format: {end_date}. Use ISO format."
                )
        
        # Get predictions with filtering
        predictions = database_manager.get_predictions_filtered(
            limit=limit,
            offset=0,
            model_version=model_version,
            status=status,
            batch_id=batch_id,
            start_time=start_datetime,
            end_time=end_datetime,
            search_term=search_term
        )
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_export_{timestamp}.{format}"
        
        if format == "csv":
            # Generate CSV content
            def generate_csv():
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "id", "request_id", "model_version", "model_stage", "prediction",
                    "confidence_lower", "confidence_upper", "confidence_score",
                    "processing_time_ms", "timestamp", "user_agent", "ip_address",
                    "batch_id", "status", "error_message",
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"
                ])
                
                # Write data rows
                for pred in predictions:
                    features = pred.input_features or {}
                    writer.writerow([
                        pred.id,
                        pred.request_id,
                        pred.model_version,
                        pred.model_stage,
                        pred.prediction,
                        pred.confidence_lower,
                        pred.confidence_upper,
                        pred.confidence_score,
                        pred.processing_time_ms,
                        pred.timestamp.isoformat() if pred.timestamp else None,
                        pred.user_agent,
                        pred.ip_address,
                        pred.batch_id,
                        pred.status,
                        pred.error_message,
                        features.get("MedInc"),
                        features.get("HouseAge"),
                        features.get("AveRooms"),
                        features.get("AveBedrms"),
                        features.get("Population"),
                        features.get("AveOccup"),
                        features.get("Latitude"),
                        features.get("Longitude")
                    ])
                
                content = output.getvalue()
                output.close()
                return content
            
            content = generate_csv()
            media_type = "text/csv"
            
        else:  # JSON format
            # Convert predictions to JSON-serializable format
            predictions_data = []
            for pred in predictions:
                pred_dict = {
                    "id": pred.id,
                    "request_id": pred.request_id,
                    "model_version": pred.model_version,
                    "model_stage": pred.model_stage,
                    "input_features": pred.input_features,
                    "prediction": pred.prediction,
                    "confidence_lower": pred.confidence_lower,
                    "confidence_upper": pred.confidence_upper,
                    "confidence_score": pred.confidence_score,
                    "processing_time_ms": pred.processing_time_ms,
                    "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
                    "user_agent": pred.user_agent,
                    "ip_address": pred.ip_address,
                    "batch_id": pred.batch_id,
                    "status": pred.status,
                    "error_message": pred.error_message
                }
                predictions_data.append(pred_dict)
            
            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_records": len(predictions_data),
                    "filters": {
                        "model_version": model_version,
                        "status": status,
                        "batch_id": batch_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "search_term": search_term
                    }
                },
                "predictions": predictions_data
            }
            
            content = json.dumps(export_data, indent=2)
            media_type = "application/json"
        
        # Record metrics
        processing_time_ms = (time.time() - start_time) * 1000
        if metrics:
            metrics.record_database_export(format, len(predictions), processing_time_ms)
        
        logger.info(f"Exported {len(predictions)} predictions in {format} format")
        
        # Return streaming response
        return StreamingResponse(
            io.StringIO(content),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(content.encode('utf-8')))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to export prediction data"
        )


@router.get("/predictions/trends")
async def get_prediction_trends(
    request: Request,
    days: int = Query(7, ge=1, le=365, description="Number of days for trend analysis"),
    interval: str = Query("hour", regex="^(hour|day)$", description="Time interval for aggregation")
) -> Dict[str, Any]:
    """
    Get prediction trends and patterns over time.
    
    This endpoint provides time-series data for prediction trends including:
    - Prediction volume over time
    - Success/failure rates over time
    - Average processing times over time
    - Model version usage over time
    
    Args:
        request: FastAPI request object
        days: Number of days to analyze (1-365)
        interval: Time interval for aggregation (hour or day)
        
    Returns:
        Dictionary with trend data and visualizations
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    logger.info(f"Getting prediction trends: days={days}, interval={interval}")
    
    try:
        # Get dependencies
        database_manager = getattr(request.app.state, 'database_manager', None)
        metrics = getattr(request.app.state, 'metrics', None)
        
        if database_manager is None:
            database_manager = get_database_manager()
        if metrics is None:
            metrics = get_metrics()
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get trend data
        trends = database_manager.get_prediction_trends(
            start_time=start_date,
            end_time=end_date,
            interval=interval
        )
        
        # Record metrics
        processing_time_ms = (time.time() - start_time) * 1000
        if metrics:
            metrics.record_database_query("get_prediction_trends", processing_time_ms)
        
        logger.info(f"Retrieved prediction trends for {days} days")
        
        return {
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
                "interval": interval
            },
            "trends": trends
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction trends: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve prediction trends"
        )