#!/bin/bash
# Script to run all tests (backend and frontend) in the Docker container

set -e  # Exit immediately if a command exits with a non-zero status

echo "===== Starting AV Catalog Converter Test Suite ====="
echo "$(date)"
echo "=================================================="

# Directory setup
BACKEND_DIR="/app"
FRONTEND_DIR="/app/web/frontend"
LOG_DIR="/app/logs"
TEST_RESULTS_DIR="/app/test_results"

# Create directories if they don't exist
mkdir -p $LOG_DIR
mkdir -p $TEST_RESULTS_DIR

# Log file
LOG_FILE="$LOG_DIR/test_run_$(date +%Y%m%d_%H%M%S).log"
BACKEND_RESULTS="$TEST_RESULTS_DIR/backend_results.xml"
FRONTEND_RESULTS="$TEST_RESULTS_DIR/frontend_results.xml"

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

# Function to run frontend tests
run_frontend_tests() {
  log "Running frontend tests..."

  # Check if frontend directory exists
  if [ ! -d "$FRONTEND_DIR" ]; then
    log "Frontend directory not found. Skipping frontend tests."
    return 0
  fi

  cd $FRONTEND_DIR

  # Install dependencies if needed
  if [ ! -d "node_modules" ]; then
    log "Installing frontend dependencies..."
    npm install --silent 2>&1 | tee -a $LOG_FILE
  fi

  # Run Jest tests with JUnit reporter
  log "Running Jest tests..."
  npx jest --ci --reporters=default --reporters=jest-junit 2>&1 | tee -a $LOG_FILE

  # Save test results
  mv junit.xml $FRONTEND_RESULTS 2>/dev/null || true

  # Check exit code
  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ Frontend tests passed!"
    return 0
  else
    log "❌ Frontend tests failed!"
    return 1
  fi
}

# Function to run integration tests
run_integration_tests() {
  log "Running integration tests..."
  cd $BACKEND_DIR

  # Start the backend server in the background
  log "Starting backend server for integration tests..."
  python app.py --api --port 8080 &
  SERVER_PID=$!

  # Wait for server to start
  log "Waiting for server to start..."
  sleep 5

  # Run integration tests
  log "Running Cypress integration tests..."
  cd $FRONTEND_DIR

  # Install Cypress if needed
  if [ ! -d "node_modules/cypress" ]; then
    log "Installing Cypress..."
    npm install cypress --silent 2>&1 | tee -a $LOG_FILE
  fi

  # Run Cypress tests
  npx cypress run --reporter junit --reporter-options "mochaFile=$TEST_RESULTS_DIR/integration_results.xml" 2>&1 | tee -a $LOG_FILE

  # Save exit code
  CYPRESS_EXIT_CODE=${PIPESTATUS[0]}

  # Kill the server
  log "Stopping backend server..."
  kill $SERVER_PID

  # Check exit code
  if [ $CYPRESS_EXIT_CODE -eq 0 ]; then
    log "✅ Integration tests passed!"
    return 0
  else
    log "❌ Integration tests failed!"
    return 1
  fi
}

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

# Run frontend tests
FRONTEND_RESULT=0
run_frontend_tests || FRONTEND_RESULT=1

# Run integration tests
INTEGRATION_RESULT=0
run_integration_tests || INTEGRATION_RESULT=1

# Summary
log "=================================================="
log "Test Suite Summary:"
log "Backend Tests: $([ $BACKEND_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
log "Frontend Tests: $([ $FRONTEND_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
log "Integration Tests: $([ $INTEGRATION_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
log "=================================================="

# Always generate test report, regardless of test results
generate_test_report

# Overall result
if [ $BACKEND_RESULT -eq 0 ] && [ $FRONTEND_RESULT -eq 0 ] && [ $INTEGRATION_RESULT -eq 0 ]; then
  log "✅ All tests passed!"
  exit 0
else
  log "❌ Some tests failed!"
  exit 1
fi
