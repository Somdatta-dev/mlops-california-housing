"""
Script to authenticate with Google Drive for DVC.
"""

import subprocess
import sys
import time

def authenticate_gdrive():
    """
    Authenticate with Google Drive using OAuth.
    """
    print("🔐 Starting Google Drive authentication...")
    print("This will open your browser for authentication.")
    print("Please complete the authentication in your browser.")
    print()
    
    try:
        # Try to push data, which will trigger authentication
        print("Attempting to push data to Google Drive...")
        result = subprocess.run(
            ["dvc", "push"], 
            capture_output=False,  # Don't capture output so user can see the auth URL
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Authentication successful! Data pushed to Google Drive.")
            print("Your files are now available at:")
            print("https://drive.google.com/drive/folders/1cTtogain8t53Ztzx2Dog0DJ9tUnscmyM")
            return True
        else:
            print("❌ Authentication failed or was interrupted.")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Authentication timed out.")
        print("Please try again and complete the authentication quickly.")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Authentication cancelled by user.")
        return False
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        return False

def check_auth_status():
    """
    Check if authentication is working by testing DVC status.
    """
    try:
        result = subprocess.run(["dvc", "status"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ DVC is working correctly.")
            return True
        else:
            print(f"❌ DVC status check failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking DVC status: {e}")
        return False

def main():
    """
    Main function to handle Google Drive authentication.
    """
    print("🚀 GOOGLE DRIVE AUTHENTICATION FOR DVC")
    print("=" * 50)
    print()
    
    # Check current status
    print("Checking current DVC status...")
    if check_auth_status():
        print("DVC is working. Proceeding with authentication...")
    else:
        print("DVC status check failed. Continuing anyway...")
    
    print()
    
    # Attempt authentication
    success = authenticate_gdrive()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("Your data has been pushed to Google Drive.")
        print("You can now use 'dvc push' and 'dvc pull' to sync data.")
    else:
        print("\n❌ AUTHENTICATION FAILED")
        print("Please try the following:")
        print("1. Make sure you have access to the Google Drive folder")
        print("2. Check that the OAuth credentials are correct")
        print("3. Try running this script again")
        print("4. If issues persist, check the Google Drive setup guide")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())