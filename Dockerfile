# Single-stage build for Python backend
# The frontend is built locally and mounted into the container

# Use a Python image for the backend
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
    tesseract-ocr-eng \
    poppler-utils \
    git \
    gnupg \
    libgtk2.0-0 \
    libgtk-3-0 \
    libgbm-dev \
    libnotify-dev \
    libgconf-2-4 \
    libnss3 \
    libxss1 \
    libasound2 \
    libxtst6 \
    xauth \
    xvfb \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend tests
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@10.2.4

# Create necessary directories
RUN mkdir -p logs data/input data/output cache/models cache/torch test_results \
    && chmod -R 777 logs data cache test_results

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in a specific order to avoid conflicts
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir pandas==1.5.3 && \
    pip install --no-cache-dir scipy==1.10.0 && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
    pip install --no-cache-dir transformers==4.30.2 accelerate==0.20.3 bitsandbytes==0.41.1 && \
    pip install --no-cache-dir sentencepiece==0.1.99 protobuf==3.20.3 && \
    pip install --no-cache-dir scikit-learn==1.2.2 && \
    pip install --no-cache-dir pytesseract>=0.3.8 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directory for frontend build
RUN mkdir -p /app/web/frontend/build

# Set permissions for scripts
RUN chmod +x docker/entrypoint.sh docker/run_tests.sh docker/run.sh docker/stop.sh

# Expose port for API
EXPOSE 8080

# Health check to ensure the API is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default command
CMD ["--api", "--port", "8080"]
