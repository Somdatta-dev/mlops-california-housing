"""
Script to configure DVC remote storage using environment variables.
Sets up Google Drive as the remote storage for DVC.
"""

import os
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dvc_remote():
    """
    Configure DVC remote storage using environment variables.
    """
    # Load environment variables
    load_dotenv()
    
    # Get DVC remote URL from environment
    dvc_remote_url = os.getenv('DVC_REMOTE_URL')
    
    if not dvc_remote_url:
        logger.error("DVC_REMOTE_URL not found in environment variables!")
        logger.info("Please set DVC_REMOTE_URL in your .env file")
        logger.info("Example: DVC_REMOTE_URL=gdrive://your-google-drive-folder-id-here")
        return False
    
    if dvc_remote_url == "gdrive://your-google-drive-folder-id-here":
        logger.warning("DVC_REMOTE_URL is still set to placeholder value!")
        logger.info("For this demo, we'll configure a local remote storage instead")
        
        # Create a local remote storage directory for demo purposes
        local_remote_dir = Path("../dvc_remote_storage")
        local_remote_dir.mkdir(exist_ok=True)
        dvc_remote_url = str(local_remote_dir.absolute())
        
        logger.info(f"Using local remote storage: {dvc_remote_url}")
    
    try:
        # Configure DVC remote
        logger.info("Configuring DVC remote storage...")
        
        # Add remote storage
        result = subprocess.run([
            "dvc", "remote", "add", "-d", "gdrive", dvc_remote_url
        ], capture_output=True, text=True, check=True)
        
        logger.info("DVC remote storage configured successfully!")
        
        # If using Google Drive, set additional configuration
        if dvc_remote_url.startswith("gdrive://"):
            # Configure Google Drive specific settings
            use_service_account = os.getenv('GDRIVE_USE_SERVICE_ACCOUNT', 'false').lower() == 'true'
            
            if use_service_account:
                logger.info("Configuring Google Drive service account...")
                subprocess.run([
                    "dvc", "remote", "modify", "gdrive", "gdrive_use_service_account", "true"
                ], check=True)
            
            logger.info("Google Drive remote configured!")
            logger.info("Note: You may need to authenticate with Google Drive on first use")
        
        # Show current remote configuration
        logger.info("Current DVC remote configuration:")
        result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure DVC remote: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def test_dvc_remote():
    """
    Test DVC remote configuration by checking status.
    """
    try:
        logger.info("Testing DVC remote configuration...")
        
        # Check DVC status
        result = subprocess.run(["dvc", "status"], capture_output=True, text=True, check=True)
        logger.info("DVC status check passed!")
        
        # Show DVC remote list
        result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
        if result.stdout:
            logger.info("Configured remotes:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC remote test failed: {e}")
        return False


def main():
    """
    Main function to set up DVC remote storage.
    """
    logger.info("Setting up DVC remote storage...")
    
    # Setup DVC remote
    if setup_dvc_remote():
        logger.info("DVC remote setup completed successfully!")
        
        # Test the configuration
        if test_dvc_remote():
            logger.info("DVC remote configuration test passed!")
        else:
            logger.warning("DVC remote configuration test failed!")
    else:
        logger.error("Failed to setup DVC remote storage!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())