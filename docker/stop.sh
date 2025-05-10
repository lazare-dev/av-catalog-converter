#!/bin/bash
# Script to stop the AV Catalog Converter Docker containers

# Stop the containers
docker-compose down

# Show the status
docker-compose ps
