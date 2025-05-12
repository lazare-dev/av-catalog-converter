# Use a Python image with CUDA support
FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/cache/models \
    TORCH_HOME=/app/cache/torch \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    tesseract-ocr \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p logs data/input data/output cache/models cache/torch

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in a specific order to avoid conflicts
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 pandas==1.5.3 && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
    pip install --no-cache-dir transformers==4.30.2 accelerate==0.20.3 bitsandbytes==0.39.1 && \
    pip install --no-cache-dir sentencepiece==0.1.99 protobuf==3.20.3 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set permissions
RUN chmod +x docker/entrypoint.sh

# Expose port for API
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default command
CMD ["--api", "--port", "8080"]
