# Use a smaller Python image
FROM python:3.9-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    curl \
    libmagic \
    tesseract-ocr \
    git \
    && rm -rf /var/cache/apk/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    # Install specific versions of numpy and pandas first to avoid conflicts
    pip install numpy==1.24.3 pandas==1.5.3 && \
    # Install the rest of the dependencies
    pip install --no-cache-dir -r requirements.txt && \
    # Install additional dependencies for LLM support
    pip install --no-cache-dir accelerate==0.20.3 sentencepiece==0.1.99 protobuf==3.20.3

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
