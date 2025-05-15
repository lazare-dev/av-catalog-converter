#!/bin/bash
# Script to run a specific test in the Docker container

# Check if a test path was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <test_path>"
    echo "Example: $0 tests/unit/services/category/test_category_extractor.py"
    exit 1
fi

TEST_PATH="$1"

# Create a container for running tests
echo "Creating Docker container for testing..."
docker run -d --name av-catalog-converter-test av-catalog-converter --test-only

# Run the specific test
echo "Running test: $TEST_PATH"
docker exec -it av-catalog-converter-test python -m pytest "$TEST_PATH" -v

# Clean up
echo "Cleaning up..."
docker rm -f av-catalog-converter-test

echo "Test completed."
