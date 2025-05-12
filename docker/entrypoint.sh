#!/bin/bash
# Entrypoint script for Docker container

set -e

# Check if we should skip tests
if [ "$SKIP_TESTS" = "true" ]; then
    echo "Skipping tests as SKIP_TESTS=true"
else
    # Run tests with parallel execution and coverage
    echo "Running comprehensive tests..."

    # Try to run tests with better error handling
    if ! python -m pytest tests/ -v --cov=. --cov-report=term --cov-report=html -n auto --timeout=300; then
        echo "Tests failed! Checking if we should continue anyway..."

        # Check if we should continue despite test failures
        if [ "$CONTINUE_ON_TEST_FAILURE" = "true" ]; then
            echo "Continuing despite test failures as CONTINUE_ON_TEST_FAILURE=true"
        else
            echo "Tests failed! Please fix the issues before continuing."
            echo "You can set CONTINUE_ON_TEST_FAILURE=true to continue anyway."
            exit 1
        fi
    else
        echo "Tests passed successfully!"

        # Print coverage summary
        echo "Coverage summary:"
        python -m coverage report --skip-covered
    fi
fi

# Start the application
echo "Starting AV Catalog Converter API..."
exec python app.py "$@"
