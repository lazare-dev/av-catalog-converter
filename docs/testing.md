# Testing Guide for AV Catalog Converter

This document provides comprehensive information about running tests for the AV Catalog Converter project.

## Table of Contents

1. [Running Tests](#running-tests)
   - [Running All Tests](#running-all-tests)
   - [Running Specific Tests](#running-specific-tests)
   - [Running Tests with Reports](#running-tests-with-reports)
2. [Test Reports](#test-reports)
   - [Automatic Report Generation](#automatic-report-generation)
   - [Manual Report Generation](#manual-report-generation)
3. [Docker Testing](#docker-testing)
   - [Running Tests in Docker](#running-tests-in-docker)
   - [Getting Test Reports from Docker](#getting-test-reports-from-docker)
4. [Troubleshooting](#troubleshooting)

## Running Tests

### Running All Tests

To run all tests in the project:

```bash
python -m pytest tests/
```

### Running Specific Tests

To run specific test categories:

```bash
# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run tests for parallel processing
python -m pytest -m parallel

# Run tests for rate limiting
python -m pytest -m rate_limiting

# Run tests for caching
python -m pytest -m caching
```

### Running Tests with Reports

To run tests and automatically generate an HTML report, use the `run_tests_with_report.py` script:

```bash
# Run all tests and generate a report
./run_tests_with_report.py

# Run specific tests and generate a report
./run_tests_with_report.py tests/unit/core/llm/

# Run with increased verbosity
./run_tests_with_report.py -v

# Run with additional pytest arguments
./run_tests_with_report.py --pytest-args "-k test_name -xvs"

# Run tests without generating a report
./run_tests_with_report.py --no-report
```

## Test Reports

### Automatic Report Generation

Test reports are automatically generated in the following scenarios:

1. When running the full test suite in Docker
2. When using the `run_tests_with_report.py` script
3. When running tests through the Docker entrypoint

The reports are generated even if tests fail, ensuring you always have visibility into test results.

### Manual Report Generation

To manually generate a test report from existing test results:

```bash
python generate_test_report.py
```

This will create an HTML report at `test_results/test_report.html` based on the XML test results in the `test_results` directory.

## Docker Testing

### Running Tests in Docker

To run tests in the Docker container:

```bash
# Run all tests
docker-compose exec app python -m pytest

# Run specific tests
docker-compose exec app python -m pytest tests/unit/core/llm/

# Run tests with the report script
docker-compose exec app ./run_tests_with_report.py
```

### Getting Test Reports from Docker

Test reports are generated inside the Docker container in the `/app/test_results` directory, which is mounted to the local `test_results` directory. You can access the reports directly from your local filesystem.

Alternatively, you can copy the reports from the container:

```bash
docker cp av-catalog-converter:/app/test_results/test_report.html ./test_report.html
```

## Troubleshooting

If you encounter issues with test reports not being generated:

1. Ensure the `test_results` directory exists
2. Check that pytest is configured to output XML results with `--junitxml=test_results/backend_results.xml`
3. Verify that the `generate_test_report.py` script is executable
4. Check for any error messages in the console output

If tests are failing but you still need reports:

1. Use the `run_tests_with_report.py` script, which generates reports regardless of test results
2. Manually run `python generate_test_report.py` after test execution
