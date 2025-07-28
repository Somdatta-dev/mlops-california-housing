"""
FastAPI Main Application

This module provides the main FastAPI application with comprehensive configuration,
middleware, error handling, and service initialization.
"""

import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import APIConfig, ModelConfig, get_api_config, get_model_config, setup_logging
from .metrics import PrometheusMetrics, initialize_metrics, get_metrics
from .model_loader import ModelLoader, initialize_model_loader, get_model_loader
from .database import DatabaseManager, initialize_database_manager, get_database_manager
from .health import router as health_router
from .predictions import router as predictions_router
from .database_endpoints import router as database_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting MLOps California Housing Prediction API")
    
    try:
        # Get configurations
        api_config = get_api_config()
        model_config = get_model_config()
        
        # Setup logging
        setup_logging(api_config)
        
        # Initialize metrics
        metrics = initialize_metrics(
            start_server=api_config.enable_prometheus,
            server_port=api_config.prometheus_port,
            start_monitoring=True,
            monitoring_interval=5.0
        )
        
        # Set system info in metrics
        import platform
        metrics.system_info.info({
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'api_version': api_config.version
        })
        
        # Initialize database manager and run migrations
        from .database_init import initialize_database
        
        logger.info("Initializing database...")
        db_init_success = initialize_database(
            config=api_config,
            run_migrations=True,
            create_sample_data=False
        )
        
        if not db_init_success:
            logger.error("Database initialization failed")
            raise RuntimeError("Database initialization failed")
        
        database_manager = initialize_database_manager(api_config)
        
        # Initialize model loader
        model_loader = initialize_model_loader(
            api_config=api_config,
            model_config=model_config,
            metrics=metrics,
            load_model_on_init=True
        )
        
        # Store instances in app state
        app.state.api_config = api_config
        app.state.model_config = model_config
        app.state.metrics = metrics
        app.state.model_loader = model_loader
        app.state.database_manager = database_manager
        
        logger.info("API startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down MLOps California Housing Prediction API")
    
    try:
        # Stop background monitoring
        if hasattr(app.state, 'metrics'):
            app.state.metrics.stop_background_monitoring()
        
        logger.info("API shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Get initial config for app metadata
    config = get_api_config()
    
    # Create FastAPI app
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        openapi_url="/openapi.json" if config.debug else None
    )
    
    # Add middleware
    setup_middleware(app, config)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Add routers
    setup_routers(app)
    
    return app


def setup_middleware(app: FastAPI, config: APIConfig) -> None:
    """
    Set up middleware for the FastAPI application.
    
    Args:
        app: FastAPI application
        config: API configuration
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (for production)
    if not config.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add request processing time to response headers."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            # Record metrics
            if hasattr(app.state, 'metrics') and app.state.metrics:
                app.state.metrics.record_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=response.status_code,
                    duration=process_time
                )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Record error metrics
            if hasattr(app.state, 'metrics') and app.state.metrics:
                app.state.metrics.record_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=500,
                    duration=process_time
                )
                app.state.metrics.record_error(
                    error_type=type(e).__name__,
                    endpoint=str(request.url.path)
                )
            
            raise
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} in {process_time:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)} in {process_time:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": process_time
                }
            )
            
            raise


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Set up exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI application
    """
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
                "detail": exc.detail
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(
            f"Validation error: {exc.errors()}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "validation_errors": exc.errors()
            }
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid request data",
                "details": exc.errors(),
                "path": str(request.url.path),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        logger.error(
            f"Starlette exception: {exc.status_code} - {exc.detail}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
                "detail": exc.detail
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "server_error",
                "message": exc.detail or "Internal server error",
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        error_id = f"error_{int(time.time())}"
        
        logger.error(
            f"Unhandled exception [{error_id}]: {str(exc)}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "error_id": error_id,
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc()
            }
        )
        
        # Record error metrics
        if hasattr(app.state, 'metrics') and app.state.metrics:
            app.state.metrics.record_error(
                error_type=type(exc).__name__,
                endpoint=str(request.url.path)
            )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "error_id": error_id,
                "path": str(request.url.path),
                "timestamp": time.time()
            }
        )


def setup_routers(app: FastAPI) -> None:
    """
    Set up API routers.
    
    Args:
        app: FastAPI application
    """
    # Health check router
    app.include_router(health_router)
    
    # Predictions router
    app.include_router(predictions_router)
    
    # Database router
    app.include_router(database_router)
    
    # Root endpoint
    @app.get("/", response_class=PlainTextResponse)
    async def root():
        """Root endpoint."""
        return "MLOps California Housing Prediction API"
    
    # Metrics endpoint (if Prometheus is not running separate server)
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint(request: Request):
        """Prometheus metrics endpoint."""
        if hasattr(app.state, 'metrics'):
            return app.state.metrics.get_metrics()
        else:
            return "# No metrics available\n"
    
    # API info endpoint
    @app.get("/info")
    async def api_info(request: Request) -> Dict[str, Any]:
        """Get API information."""
        config = app.state.api_config if hasattr(app.state, 'api_config') else get_api_config()
        
        return {
            "name": config.title,
            "version": config.version,
            "description": config.description,
            "debug": config.debug,
            "timestamp": time.time(),
            "endpoints": {
                "health": "/health",
                "detailed_health": "/health/detailed",
                "model_health": "/health/model",
                "system_health": "/health/system",
                "gpu_health": "/health/gpu",
                "metrics": "/metrics",
                "docs": "/docs" if config.debug else None,
                "redoc": "/redoc" if config.debug else None
            }
        }


# Create the FastAPI app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    config = get_api_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level.lower(),
        access_log=True
    )