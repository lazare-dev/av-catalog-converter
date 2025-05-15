#!/bin/bash
# Script to build and run Docker container with tests

echo "Building Docker container for AV Catalog Converter..."
docker build -t av-catalog-converter .

echo "Running Docker container with tests..."
docker run --name av-catalog-converter-test av-catalog-converter --test-only

echo "Copying test results from container..."
docker cp av-catalog-converter-test:/app/test_results ./test_results

echo "Cleaning up..."
docker rm av-catalog-converter-test

echo "Test results are available in the ./test_results directory"
