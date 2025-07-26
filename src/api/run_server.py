#!/usr/bin/env python3
"""
FastAPI Server Runner

This script provides a convenient way to run the FastAPI server with
proper configuration and error handling.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import uvicorn
from api.config import get_api_config, setup_logging


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Run MLOps FastAPI Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", default=None, help="Log level")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_api_config()
    
    # Override config with command line arguments
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.reload:
        config.reload = True
    if args.debug:
        config.debug = True
    if args.log_level:
        config.log_level = args.log_level.upper()
    
    # Setup logging
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MLOps FastAPI Server on {config.host}:{config.port}")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"Reload mode: {config.reload}")
    logger.info(f"Log level: {config.log_level}")
    
    try:
        # Run the server
        uvicorn.run(
            "api.main:app",
            host=config.host,
            port=config.port,
            reload=config.reload,
            log_level=config.log_level.lower(),
            access_log=True,
            workers=1 if config.reload else None
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()