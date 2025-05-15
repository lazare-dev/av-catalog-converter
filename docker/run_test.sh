#!/bin/bash
# Enhanced script to run specific tests in the Docker container
# Supports running tests by path, class, or method name

set -e

# Function to display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] <test_identifier>"
    echo
    echo "Run a specific test or test file in the Docker container."
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -v, --verbose              Run tests with increased verbosity"
    echo "  -k, --keyword PATTERN      Only run tests matching the given substring expression"
    echo "  -m, --marker MARKER        Only run tests with the given marker"
    echo "  -r, --report               Generate an HTML test report"
    echo "  --no-cleanup               Don't remove the container after running tests"
    echo
    echo "Examples:"
    echo "  $0 tests/unit/core/llm/test_phi_client.py                       # Run all tests in a file"
    echo "  $0 tests/unit/core/llm/test_phi_client.py::TestPhiClient        # Run all tests in a class"
    echo "  $0 \"tests/unit/core/llm/test_phi_client.py::TestPhiClient::test_method\"  # Run a specific test method"
    echo "  $0 -k \"health_check\"                                           # Run all tests with 'health_check' in the name"
    echo "  $0 \"tests.integration.test_api_with_optimizations.TestAPIWithOptimizations::test_health_check_with_llm_stats\"  # Run by full test path"
    echo
}

# Default values
VERBOSE=""
KEYWORD=""
MARKER=""
GENERATE_REPORT=false
CLEANUP=true
CONTAINER_NAME="av-catalog-converter-test-$(date +%s)"  # Unique container name

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        -k|--keyword)
            KEYWORD="-k \"$2\""
            # When using -k, we don't need a test identifier
            # We'll run all tests matching the keyword
            TEST_IDENTIFIER="tests/"
            shift 2
            ;;
        -m|--marker)
            MARKER="-m \"$2\""
            # When using -m, we don't need a test identifier
            # We'll run all tests with the marker
            TEST_IDENTIFIER="tests/"
            shift 2
            ;;
        -r|--report)
            GENERATE_REPORT=true
            shift
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        *)
            TEST_IDENTIFIER="$1"
            shift
            ;;
    esac
done

# Check if a test identifier was provided
if [ -z "$TEST_IDENTIFIER" ]; then
    echo "Error: No test identifier provided."
    show_usage
    exit 1
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running or not accessible."
        exit 1
    fi
}

# Function to check if the image exists
check_image() {
    if ! docker image inspect av-catalog-converter > /dev/null 2>&1; then
        echo "Error: Docker image 'av-catalog-converter' not found."
        echo "Please build the image first with: docker build -t av-catalog-converter ."
        exit 1
    fi
}

# Function to run the test
run_test() {
    local test_cmd="python -m pytest"

    # Build the test command
    test_cmd="$test_cmd $TEST_IDENTIFIER $VERBOSE $KEYWORD $MARKER"

    # Remove extra quotes from the command
    test_cmd=$(echo "$test_cmd" | sed 's/"\([^"]*\)"/\1/g')

    # Create a container for running tests
    echo "Creating Docker container for testing..."
    docker run -d --name "$CONTAINER_NAME" av-catalog-converter --test-only

    # Run the test
    echo "Running test: $TEST_IDENTIFIER"
    echo "Command: $test_cmd"
    docker exec -it "$CONTAINER_NAME" bash -c "$test_cmd"
    TEST_EXIT_CODE=$?

    # Generate report if requested
    if [ "$GENERATE_REPORT" = true ]; then
        echo "Generating test report..."
        docker exec -it "$CONTAINER_NAME" python generate_test_report.py

        # Copy the report from the container
        echo "Copying test report from container..."
        docker cp "$CONTAINER_NAME:/app/test_results/test_report.html" ./test_results/
        echo "Test report saved to: $(pwd)/test_results/test_report.html"
    fi

    # Clean up
    if [ "$CLEANUP" = true ]; then
        echo "Cleaning up..."
        docker rm -f "$CONTAINER_NAME"
    else
        echo "Container '$CONTAINER_NAME' is still running. Remember to clean up with:"
        echo "docker rm -f $CONTAINER_NAME"
    fi

    return $TEST_EXIT_CODE
}

# Main execution
echo "=== AV Catalog Converter Test Runner ==="
check_docker
check_image
run_test
TEST_RESULT=$?

echo "Test completed with exit code: $TEST_RESULT"
exit $TEST_RESULT
