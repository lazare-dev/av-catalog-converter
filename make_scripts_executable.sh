#!/bin/bash
# Make scripts executable

echo "Making scripts executable..."

# Make build_frontend.sh executable
chmod +x build_frontend.sh

# Make Docker scripts executable
chmod +x docker/entrypoint.sh
chmod +x docker/run_tests.sh
chmod +x docker/run.sh
chmod +x docker/stop.sh

echo "Scripts are now executable!"
