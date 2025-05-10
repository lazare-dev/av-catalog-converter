# Docker Setup for AV Catalog Converter

This directory contains Docker-related files and instructions for containerizing the AV Catalog Converter application.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Quick Start

1. Build and start the containers:

```bash
docker-compose up -d
```

2. Check if the containers are running:

```bash
docker-compose ps
```

3. Access the API at http://localhost:8080

4. Stop the containers:

```bash
docker-compose down
```

## Configuration

The Docker setup uses the following configuration:

- The application runs on port 8080 (mapped to host port 8080)
- Input files should be placed in the `data/input` directory
- Output files will be saved to the `data/output` directory
- Logs are stored in the `logs` directory

## Environment Variables

You can customize the application behavior using environment variables:

- `LOG_LEVEL`: Set the logging level (default: INFO)
- `PYTHONPATH`: Set the Python path (default: /app)

## Building the Image Manually

To build the Docker image manually:

```bash
docker build -t av-catalog-converter .
```

To run the container manually:

```bash
docker run -p 8080:8080 -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs av-catalog-converter
```

## Running Tests in Docker

To run tests inside the Docker container:

```bash
docker-compose run --rm app python run_tests.py
```

## Troubleshooting

If you encounter issues with the Docker setup:

1. Check the logs:

```bash
docker-compose logs app
```

2. Ensure the required directories exist and have proper permissions:

```bash
mkdir -p data/input data/output logs
chmod -R 777 data logs
```

3. Verify the application is healthy:

```bash
curl http://localhost:8080/api/health
```
