#!/bin/bash
# Script to run backend tests in the Docker container

set -e  # Exit immediately if a command exits with a non-zero status

echo "===== Starting AV Catalog Converter Test Suite ====="
echo "$(date)"
echo "=================================================="

# Directory setup
BACKEND_DIR="/app"
LOG_DIR="/app/logs"
TEST_RESULTS_DIR="/app/test_results"

# Create directories if they don't exist
mkdir -p $LOG_DIR
mkdir -p $TEST_RESULTS_DIR

# Log file
LOG_FILE="$LOG_DIR/test_run_$(date +%Y%m%d_%H%M%S).log"
BACKEND_RESULTS="$TEST_RESULTS_DIR/backend_results.xml"

# Function to log messages
log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a $LOG_FILE
}

# Function to run backend tests
run_backend_tests() {
  log "Running backend tests..."
  cd $BACKEND_DIR

  # Create flag files to track specific test failures
  LLM_TESTS_FAILED_FLAG="$TEST_RESULTS_DIR/llm_tests_failed"
  PARSER_TESTS_FAILED_FLAG="$TEST_RESULTS_DIR/parser_tests_failed"
  CATEGORY_TESTS_FAILED_FLAG="$TEST_RESULTS_DIR/category_tests_failed"

  # Remove previous flags if they exist
  rm -f $LLM_TESTS_FAILED_FLAG $PARSER_TESTS_FAILED_FLAG $CATEGORY_TESTS_FAILED_FLAG

  # Run pytest with JUnit XML output
  log "Running all backend tests..."
  python -m pytest tests/ -v --junitxml=$BACKEND_RESULTS 2>&1 | tee -a $LOG_FILE
  MAIN_TEST_RESULT=${PIPESTATUS[0]}

  # Run specific test categories to identify failures
  log "Running LLM tests to identify issues..."
  python -m pytest tests/unit/core/llm/ -v 2>&1 | tee -a $LOG_FILE || touch $LLM_TESTS_FAILED_FLAG

  log "Running parser tests to identify issues..."
  python -m pytest tests/unit/core/file_parser/ -v 2>&1 | tee -a $LOG_FILE || touch $PARSER_TESTS_FAILED_FLAG

  log "Running category tests to identify issues..."
  python -m pytest tests/unit/services/category/ -v 2>&1 | tee -a $LOG_FILE || touch $CATEGORY_TESTS_FAILED_FLAG

  # Check exit code
  if [ $MAIN_TEST_RESULT -eq 0 ]; then
    log "✅ Backend tests passed!"
    return 0
  else
    log "❌ Backend tests failed!"

    # Log specific failures
    if [ -f $LLM_TESTS_FAILED_FLAG ]; then
      log "❌ LLM tests failed!"
    fi

    if [ -f $PARSER_TESTS_FAILED_FLAG ]; then
      log "❌ Parser tests failed!"
    fi

    if [ -f $CATEGORY_TESTS_FAILED_FLAG ]; then
      log "❌ Category tests failed!"
    fi

    return 1
  fi
}

# No frontend or integration tests

# Function to generate test report
generate_test_report() {
  log "Generating test report..."

  # Ensure test_results directory exists
  mkdir -p $TEST_RESULTS_DIR

  # Generate the HTML report
  python generate_test_report.py

  # Check if report was generated
  if [ -f "$TEST_RESULTS_DIR/test_report.html" ]; then
    log "✅ Test report generated successfully: $TEST_RESULTS_DIR/test_report.html"
  else
    log "❌ Failed to generate test report"
  fi
}

# Main execution
log "Starting test suite execution"

# Run backend tests
BACKEND_RESULT=0
run_backend_tests || BACKEND_RESULT=1

# Summary
log "=================================================="
log "Test Suite Summary:"
log "Backend Tests: $([ $BACKEND_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
log "=================================================="

# Always generate test report, regardless of test results
generate_test_report

# Overall result
if [ $BACKEND_RESULT -eq 0 ]; then
  log "✅ All tests passed!"
  exit 0
else
  log "❌ Some tests failed!"
  exit 1
fi
