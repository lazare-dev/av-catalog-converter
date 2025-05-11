#!/bin/bash
# Entrypoint script for Docker container

set -e

# Run tests first with parallel execution and coverage
echo "Running comprehensive tests..."
python -m pytest tests/ -v --cov=. --cov-report=term --cov-report=html -n auto --timeout=300

# Check if tests passed
if [ $? -ne 0 ]; then
    echo "Tests failed! Please fix the issues before continuing."
    exit 1
fi

echo "Tests passed successfully!"

# Print coverage summary
echo "Coverage summary:"
python -m coverage report --skip-covered

# Start the application
echo "Starting AV Catalog Converter API..."
exec python app.py "$@"
