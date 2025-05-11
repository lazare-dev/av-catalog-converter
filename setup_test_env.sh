#!/bin/bash
# Script to set up the test environment

# Exit on error
set -e

echo "Setting up test environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install specific versions of numpy and pandas first to avoid conflicts
echo "Installing numpy and pandas..."
pip install numpy==1.24.3 pandas==1.5.3

# Install test dependencies
echo "Installing test dependencies..."
pip install -r test-requirements.txt

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Test environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run tests, run: python -m pytest tests/"
