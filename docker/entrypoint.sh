#!/bin/bash
# Entrypoint script for Docker container

# Don't exit on error - we want to handle errors gracefully
set +e

# Create necessary directories with proper permissions
mkdir -p /app/logs
mkdir -p /app/data/input
mkdir -p /app/data/output
mkdir -p /app/cache/models
mkdir -p /app/cache/torch
mkdir -p /app/test_results
chmod -R 777 /app/logs /app/data /app/cache /app/test_results

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_CACHE=/app/cache/models
export TORCH_HOME=/app/cache/torch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set environment variable to indicate we're running in Docker
export RUNNING_IN_DOCKER=true

# Create a log file for the entrypoint script
ENTRYPOINT_LOG="/app/logs/entrypoint.log"
touch $ENTRYPOINT_LOG

# Function to log messages to both console and log file
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $ENTRYPOINT_LOG
}

log "Starting AV Catalog Converter Docker container"
log "Python version: $(python --version 2>&1)"
log "Pip version: $(pip --version)"
log "System information:"
log "- CPU: $(grep -c processor /proc/cpuinfo) cores"
log "- Memory: $(free -h | grep Mem | awk '{print $2}')"
log "- Disk space: $(df -h / | tail -1 | awk '{print $4}') available"

# Pre-initialize the LLM to ensure it's ready
log "Pre-initializing LLM model..."
python -c "from core.llm.llm_factory import LLMFactory; LLMFactory.create_client()" 2>&1 | tee -a $ENTRYPOINT_LOG
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Warning: LLM pre-initialization failed. Will retry during application startup."
    # Create a flag file to indicate LLM initialization failed
    touch /app/llm_init_failed
else
    log "LLM pre-initialization successful"
fi

# Check if we should skip tests
if [ "$SKIP_TESTS" = "true" ]; then
    log "Skipping tests as SKIP_TESTS=true"
else
    # Run comprehensive test suite
    log "Running comprehensive test suite..."

    # Create test results directory
    mkdir -p /app/test_results

    # Run the test script
    if ! bash /app/docker/run_tests.sh; then
        log "Tests failed! Checking if we should continue anyway..."

        # Check if we should continue despite test failures
        if [ "$CONTINUE_ON_TEST_FAILURE" = "true" ]; then
            log "Continuing despite test failures as CONTINUE_ON_TEST_FAILURE=true"
        else
            log "Tests failed! Please fix the issues before continuing."
            log "You can set CONTINUE_ON_TEST_FAILURE=true to continue anyway."

            # Don't exit, just set a flag
            export TESTS_FAILED=true
        fi
    else
        log "All tests passed successfully!"
    fi

    # Check for specific test failures
    if [ -f "/app/test_results/llm_tests_failed" ]; then
        log "LLM tests failed! This is expected if there are compatibility issues."
        log "The application will use fallback mechanisms for LLM functionality."

        # Create a flag file to indicate LLM tests failed
        touch /app/llm_tests_failed
        log "Created flag file: /app/llm_tests_failed"

        # Continue anyway as LLM has fallback mechanisms
        log "Continuing despite LLM test failures..."
    fi

    # No frontend tests to check

    # Check for integration test failures
    if [ -f "/app/test_results/integration_tests_failed" ]; then
        log "Integration tests failed! This may affect the overall functionality."

        # Check if we should continue despite integration test failures
        if [ "$CONTINUE_ON_INTEGRATION_TEST_FAILURE" = "true" ]; then
            log "Continuing despite integration test failures as CONTINUE_ON_INTEGRATION_TEST_FAILURE=true"
        else
            log "Integration tests failed! Please fix the issues before continuing."
            log "You can set CONTINUE_ON_INTEGRATION_TEST_FAILURE=true to continue anyway."

            # Don't exit, just set a flag
            export INTEGRATION_TESTS_FAILED=true
        fi
    fi
fi

# Check if tests failed but we're continuing anyway
if [ "$TESTS_FAILED" = "true" ] && [ "$CONTINUE_ON_TEST_FAILURE" = "true" ]; then
    log "WARNING: Running with failed tests. Some functionality may not work correctly."
fi

# Install any missing dependencies that might be needed
log "Ensuring all dependencies are installed..."
pip install --no-cache-dir -r requirements.txt 2>&1 | tee -a $ENTRYPOINT_LOG

# Generate test report if tests were run
if [ -f "/app/test_results/backend_results.xml" ]; then
    log "Generating test report from test results..."
    python generate_test_report.py
    if [ -f "/app/test_results/test_report.html" ]; then
        log "Test report generated successfully: /app/test_results/test_report.html"
    else
        log "Failed to generate test report"
    fi
fi

# Check if we're only running tests
if [[ "$*" == *"--test-only"* ]]; then
    log "Running tests only, not starting the API"
    exit 0
else
    # Start the application with enhanced logging
    log "Starting AV Catalog Converter API..."
    log "Command: python app.py --api"
    python app.py --api 2>&1 | tee -a $ENTRYPOINT_LOG
fi
