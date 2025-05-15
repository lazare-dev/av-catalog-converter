#!/bin/bash
# Script to build and run the Docker container with tests

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
  local color=$1
  local message=$2
  echo -e "${color}${message}${NC}"
}

# Function to check if Docker is running
check_docker() {
  if ! docker info > /dev/null 2>&1; then
    print_message "$RED" "Error: Docker is not running. Please start Docker and try again."
    exit 1
  fi
}

# Function to create required directories
create_directories() {
  print_message "$BLUE" "Creating required directories..."

  mkdir -p data/input
  mkdir -p data/output
  mkdir -p logs
  mkdir -p cache
  mkdir -p test_results

  print_message "$GREEN" "Directories created successfully."
}

# Function to build and start the container
build_and_start() {
  print_message "$BLUE" "Building and starting the Docker container..."

  # Build and start the container in detached mode
  docker-compose up --build -d

  if [ $? -ne 0 ]; then
    print_message "$RED" "Error: Failed to build and start the Docker container."
    exit 1
  fi

  print_message "$GREEN" "Docker container started successfully."
}

# Function to follow logs until tests complete
follow_logs() {
  print_message "$BLUE" "Following logs until tests complete..."

  # Follow logs until we see a marker indicating tests are complete
  docker-compose logs -f app | while read line; do
    echo "$line"

    # Check if the line contains markers indicating tests are complete
    if [[ "$line" == *"All tests passed successfully!"* ]] ||
       [[ "$line" == *"Tests failed! Checking if we should continue anyway..."* ]]; then
      # Break out of the loop
      pkill -P $$ docker-compose
      break
    fi
  done

  print_message "$GREEN" "Tests completed."
}

# Function to display test results
display_test_results() {
  print_message "$BLUE" "Displaying test results..."

  # Check if test results exist
  if [ -d "test_results" ] && [ "$(ls -A test_results)" ]; then
    # Display summary of test results
    if [ -f "test_results/backend_results.xml" ]; then
      print_message "$YELLOW" "Backend Test Results:"
      python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('test_results/backend_results.xml')
root = tree.getroot()
tests = int(root.attrib.get('tests', 0))
failures = int(root.attrib.get('failures', 0))
errors = int(root.attrib.get('errors', 0))
skipped = int(root.attrib.get('skipped', 0))
print(f'Tests: {tests}, Failures: {failures}, Errors: {errors}, Skipped: {skipped}')
print(f'Success Rate: {((tests - failures - errors) / tests * 100) if tests > 0 else 0:.2f}%')
      "
    else
      print_message "$YELLOW" "No backend test results found."
    fi

    if [ -f "test_results/frontend_results.xml" ]; then
      print_message "$YELLOW" "Frontend Test Results:"
      python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('test_results/frontend_results.xml')
root = tree.getroot()
tests = int(root.attrib.get('tests', 0))
failures = int(root.attrib.get('failures', 0))
errors = int(root.attrib.get('errors', 0))
skipped = int(root.attrib.get('skipped', 0))
print(f'Tests: {tests}, Failures: {failures}, Errors: {errors}, Skipped: {skipped}')
print(f'Success Rate: {((tests - failures - errors) / tests * 100) if tests > 0 else 0:.2f}%')
      "
    else
      print_message "$YELLOW" "No frontend test results found."
    fi

    if [ -f "test_results/integration_results.xml" ]; then
      print_message "$YELLOW" "Integration Test Results:"
      python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('test_results/integration_results.xml')
root = tree.getroot()
tests = int(root.attrib.get('tests', 0))
failures = int(root.attrib.get('failures', 0))
errors = int(root.attrib.get('errors', 0))
skipped = int(root.attrib.get('skipped', 0))
print(f'Tests: {tests}, Failures: {failures}, Errors: {errors}, Skipped: {skipped}')
print(f'Success Rate: {((tests - failures - errors) / tests * 100) if tests > 0 else 0:.2f}%')
      "
    else
      print_message "$YELLOW" "No integration test results found."
    fi

    # Generate HTML report
    print_message "$BLUE" "Generating HTML test report..."
    if [ -f "./generate_test_report.py" ]; then
      ./generate_test_report.py

      if [ -f "test_results/test_report.html" ]; then
        print_message "$GREEN" "HTML test report generated: test_results/test_report.html"

        # Try to open the report in the default browser
        if command -v open &> /dev/null; then
          open "test_results/test_report.html"
        elif command -v xdg-open &> /dev/null; then
          xdg-open "test_results/test_report.html"
        elif command -v explorer &> /dev/null; then
          explorer "test_results/test_report.html"
        else
          print_message "$YELLOW" "Please open the HTML report manually: test_results/test_report.html"
        fi
      else
        print_message "$RED" "Failed to generate HTML test report."
      fi
    else
      print_message "$YELLOW" "HTML report generator not found. Skipping HTML report generation."
    fi
  else
    print_message "$RED" "No test results found. Tests may not have completed successfully."
  fi
}

# Function to check container status
check_container_status() {
  print_message "$BLUE" "Checking container status..."

  # Check if container is running
  if docker-compose ps | grep -q "Up"; then
    print_message "$GREEN" "Container is running."
    print_message "$BLUE" "API is available at: http://localhost:8080"
  else
    print_message "$RED" "Container is not running. Something went wrong."
    docker-compose logs app | tail -n 50
    exit 1
  fi
}

# Main execution
print_message "$BLUE" "=== AV Catalog Converter with Tests ==="

# Check if Docker is running
check_docker

# Create required directories
create_directories

# Build and start the container
build_and_start

# Follow logs until tests complete
follow_logs

# Display test results
display_test_results

# Check container status
check_container_status

print_message "$GREEN" "Setup complete! The application is running with tests."
