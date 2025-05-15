#!/bin/bash
# Script to install missing dependencies for the AV Catalog Converter

echo "Installing missing dependencies..."

# Install pytesseract and its dependencies
pip install pytesseract

# Ensure correct NumPy version
pip uninstall -y numpy
pip install numpy==1.24.3

# Install scipy
pip install scipy>=1.10.0

# Reinstall pandas with the correct NumPy version
pip install pandas==1.5.3

# Install other dependencies that might be missing
pip install -r requirements.txt

echo "Dependencies installed successfully!"
