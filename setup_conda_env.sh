#!/bin/bash
# Script to set up a conda environment for the AV Catalog Converter

echo "Setting up conda environment for AV Catalog Converter..."

# Create a new conda environment
conda create -n av-catalog-converter python=3.9 -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate av-catalog-converter

# Install dependencies in the correct order
conda install -y numpy=1.24.3
conda install -y pandas=1.5.3
conda install -y scipy=1.10.0
conda install -y pytorch=2.0.1 -c pytorch
conda install -y transformers=4.30.2 -c huggingface
conda install -y pytest=7.3.1

# Install other dependencies
pip install -r requirements.txt

echo "Conda environment 'av-catalog-converter' set up successfully!"
echo "To activate the environment, run: conda activate av-catalog-converter"
