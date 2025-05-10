#!/usr/bin/env python
"""
Installation script for AV Catalog Converter
"""
import os
import sys
import subprocess
import platform

def main():
    """Main installation function"""
    print("Installing AV Catalog Converter...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return 1
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install package in development mode
    print("Installing package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Install platform-specific dependencies
    system = platform.system().lower()
    if system == "windows":
        print("Installing Windows-specific dependencies...")
        # Uncomment if needed:
        # subprocess.check_call([sys.executable, "-m", "pip", "install", "python-magic-bin"])
    else:
        print("Installing Unix-specific dependencies...")
        # Uncomment if needed:
        # subprocess.check_call([sys.executable, "-m", "pip", "install", "python-magic"])
    
    print("\nInstallation complete!")
    print("\nTo run the application:")
    print("  - As a command-line tool: av-catalog-converter --input file.csv --output result.csv")
    print("  - As an API server: av-catalog-converter --api --port 8080")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
