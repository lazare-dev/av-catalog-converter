#!/bin/bash
# Script to run the AV Catalog Converter in Docker

# Create required directories
mkdir -p data/input data/output logs uploads

# Build and run the Docker container
docker-compose up -d

# Check if the container is running
docker-compose ps

# Show the logs
docker-compose logs -f app
