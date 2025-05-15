#!/bin/bash
# Script to build the frontend for AV Catalog Converter

echo "Building frontend for AV Catalog Converter..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js to build the frontend."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install npm to build the frontend."
    exit 1
fi

# Create web/frontend/build directory if it doesn't exist
mkdir -p web/frontend/build

# Install dependencies
echo "Installing dependencies..."
npm install

# Build the frontend
echo "Building frontend..."
npm run build

# Copy the build to web/frontend/build
echo "Copying build to web/frontend/build..."
cp -r build/* web/frontend/build/

echo "Frontend build completed successfully!"
