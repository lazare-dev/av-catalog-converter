# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    tesseract-ocr \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
# Install specific versions of numpy and pandas first to avoid conflicts
RUN pip install numpy==1.24.3 pandas==1.5.3
# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for LLM support
RUN pip install --no-cache-dir accelerate==0.20.3 bitsandbytes==0.41.1 sentencepiece==0.1.99 protobuf==3.20.3

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/input data/output cache/models cache/torch

# Set permissions
RUN chmod +x run_tests.py
RUN chmod +x docker/entrypoint.sh

# Expose port for API
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default command
CMD ["--api", "--port", "8080"]
