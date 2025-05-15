#!/bin/bash
# Script to run frontend tests

echo "Running frontend tests..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js to run frontend tests."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install npm to run frontend tests."
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Run tests
echo "Running Jest tests..."
npm test

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All frontend tests passed!"
else
    echo "❌ Some frontend tests failed."
    exit 1
fi
