#!/bin/bash
# Script to set up a virtual environment with the correct dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up virtual environment for AV Catalog Converter...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
else
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install specific versions of numpy and pandas first to avoid conflicts
echo -e "${YELLOW}Installing numpy and pandas...${NC}"
pip install numpy==1.24.3 pandas==1.5.3

# Install the rest of the dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}Virtual environment setup complete!${NC}"
echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
echo -e "${GREEN}source venv/bin/activate${NC}"

# Deactivate virtual environment
deactivate

echo -e "${GREEN}Done!${NC}"
