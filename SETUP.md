# AV Catalog Converter Setup Guide

This guide provides detailed instructions for setting up and running the AV Catalog Converter project.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Git (for version control)

## Setup Options

You can run the AV Catalog Converter in two ways:

1. **Local Development Environment**: Run directly on your machine using a virtual environment
2. **Docker Container**: Run in a containerized environment

## Option 1: Local Development Environment

### Step 1: Create a Virtual Environment

Run the provided setup script to create a virtual environment with the correct dependencies:

```bash
# Make the script executable
chmod +x setup_venv.sh

# Run the script
./setup_venv.sh
```

Or manually create a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install specific versions of numpy and pandas first to avoid conflicts
pip install numpy==1.24.3 pandas==1.5.3

# Install the rest of the dependencies
pip install -r requirements.txt
```

### Step 2: Run Tests

After setting up the virtual environment, you can run the tests:

```bash
# Activate the virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
python run_tests.py --all

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration
```

### Step 3: Run the Application

To run the application in API mode:

```bash
# Activate the virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
python app.py --api --port 8080
```

## Option 2: Docker Container

### Step 1: Build and Run the Docker Container

```bash
# Build and start the container
docker-compose up --build
```

By default, the Docker container will:
- Skip running tests on startup (controlled by `SKIP_TESTS=true` in docker-compose.yml)
- Continue even if tests fail (controlled by `CONTINUE_ON_TEST_FAILURE=true` in docker-compose.yml)

### Step 2: Run Tests in the Docker Container

If you want to run tests in the Docker container:

```bash
# Run tests in the container
docker exec -it av-catalog-converter python run_tests.py --all
```

## Troubleshooting

### NumPy Version Conflicts

If you encounter NumPy version conflicts, ensure you're using the correct version:

```bash
pip install numpy==1.24.3
```

### Docker Build Issues

If you encounter GPG signature verification errors during Docker build:

1. Try rebuilding with the updated Dockerfile that includes GPG key handling
2. If issues persist, try:

```bash
# Pull the base image explicitly
docker pull python:3.9-slim

# Then build
docker-compose build --no-cache
```

### Test Execution Issues

If tests fail to run:

1. Check that you're using the correct versions of dependencies
2. Try running with the `--force` flag:

```bash
python run_tests.py --all --force
```

3. Skip environment checks:

```bash
python run_tests.py --all --skip-env-check
```

## Configuration

### Environment Variables

The following environment variables can be set in the Docker container:

- `SKIP_TESTS`: Set to `true` to skip running tests on startup
- `CONTINUE_ON_TEST_FAILURE`: Set to `true` to continue even if tests fail
- `LOG_LEVEL`: Set the logging level (default: `INFO`)
- `TRANSFORMERS_CACHE`: Set the cache directory for transformer models
- `TORCH_HOME`: Set the cache directory for PyTorch models

## Additional Resources

- See `README.md` for general project information
- See `docs/` for detailed documentation
- See `tests/README.md` for information about the test suite
