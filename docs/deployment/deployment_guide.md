# AV Catalog Converter Deployment Guide

This guide provides detailed instructions for deploying the AV Catalog Converter in various environments, from local development to production.

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [AWS Deployment](#aws-deployment)
  - [Azure Deployment](#azure-deployment)
  - [Google Cloud Deployment](#google-cloud-deployment)
- [Scaling Considerations](#scaling-considerations)
- [Security Considerations](#security-considerations)
- [Monitoring and Logging](#monitoring-and-logging)
- [Backup and Recovery](#backup-and-recovery)

## Docker Deployment

Docker is the recommended deployment method for the AV Catalog Converter, as it provides a consistent environment and simplifies dependency management.

### Prerequisites

- Docker 20.10.0 or higher
- Docker Compose 2.0.0 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB disk space minimum

### Basic Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/av-catalog-converter.git
   cd av-catalog-converter
   ```

2. Build and start the Docker containers:
   ```bash
   docker compose up -d
   ```

3. Access the application:
   - Web UI: http://localhost:3000
   - API: http://localhost:8080

4. Stop the containers:
   ```bash
   docker compose down
   ```

### Production Deployment

For production environments, use the production Docker Compose configuration:

```bash
# Build optimized containers
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

The production configuration includes:
- Optimized build settings
- HTTPS support
- Persistent volume for data storage
- Health checks
- Automatic restarts

### Configuration Options

You can customize the deployment by setting environment variables:

```bash
# Set environment variables
export AV_CATALOG_API_PORT=9000
export AV_CATALOG_UI_PORT=8000
export AV_CATALOG_LOG_LEVEL=INFO

# Deploy with custom configuration
docker compose up -d
```

Common configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `AV_CATALOG_API_PORT` | API server port | 8080 |
| `AV_CATALOG_UI_PORT` | Web UI port | 3000 |
| `AV_CATALOG_LOG_LEVEL` | Logging level | INFO |
| `AV_CATALOG_WORKERS` | Number of worker processes | CPU count |
| `AV_CATALOG_CACHE_SIZE` | Cache size in MB | 512 |
| `AV_CATALOG_LLM_MODEL` | LLM model to use | microsoft/phi-2 |

## Local Deployment

For development or testing purposes, you can deploy the application locally without Docker.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Node.js 14+ and npm (for frontend development)
- Tesseract OCR (optional, for PDF parsing with OCR capabilities)

### Backend Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/av-catalog-converter.git
   cd av-catalog-converter
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the API server:
   ```bash
   python app.py --api --port 8080
   ```

### Frontend Deployment

1. Navigate to the frontend directory:
   ```bash
   cd web/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. For production builds:
   ```bash
   npm run build
   ```

## Cloud Deployment

### AWS Deployment

#### Using Elastic Beanstalk

1. Install the AWS CLI and EB CLI:
   ```bash
   pip install awscli awsebcli
   ```

2. Initialize the EB application:
   ```bash
   eb init -p docker av-catalog-converter
   ```

3. Create an environment and deploy:
   ```bash
   eb create av-catalog-converter-env
   ```

4. For subsequent deployments:
   ```bash
   eb deploy
   ```

#### Using ECS (Elastic Container Service)

1. Create an ECR repository:
   ```bash
   aws ecr create-repository --repository-name av-catalog-converter
   ```

2. Build and push the Docker image:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<region>.amazonaws.com
   docker build -t <your-account-id>.dkr.ecr.<region>.amazonaws.com/av-catalog-converter:latest .
   docker push <your-account-id>.dkr.ecr.<region>.amazonaws.com/av-catalog-converter:latest
   ```

3. Create an ECS cluster, task definition, and service using the AWS Console or CLI.

### Azure Deployment

#### Using Azure Container Instances

1. Create a resource group:
   ```bash
   az group create --name av-catalog-converter-rg --location eastus
   ```

2. Create a container registry:
   ```bash
   az acr create --resource-group av-catalog-converter-rg --name avconverteracr --sku Basic
   ```

3. Build and push the Docker image:
   ```bash
   az acr login --name avconverteracr
   docker build -t avconverteracr.azurecr.io/av-catalog-converter:latest .
   docker push avconverteracr.azurecr.io/av-catalog-converter:latest
   ```

4. Deploy the container:
   ```bash
   az container create --resource-group av-catalog-converter-rg --name av-catalog-converter --image avconverteracr.azurecr.io/av-catalog-converter:latest --dns-name-label av-catalog-converter --ports 80
   ```

### Google Cloud Deployment

#### Using Google Cloud Run

1. Install the Google Cloud SDK:
   ```bash
   # Follow instructions at https://cloud.google.com/sdk/docs/install
   ```

2. Build and push the Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/av-catalog-converter
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy av-catalog-converter --image gcr.io/[PROJECT-ID]/av-catalog-converter --platform managed
   ```

## Scaling Considerations

### Horizontal Scaling

For high-volume processing, consider deploying multiple instances behind a load balancer:

- Use Kubernetes for container orchestration
- Configure auto-scaling based on CPU/memory usage
- Implement a shared cache (e.g., Redis) for improved performance

### Vertical Scaling

For processing large files, consider increasing resources:

- Increase container memory limits
- Use instances with more CPU cores
- Optimize parallel processing settings

## Security Considerations

### API Security

- Implement API authentication for production deployments
- Use HTTPS for all communications
- Implement rate limiting to prevent abuse
- Validate all input data

### Data Security

- Use encrypted storage for sensitive data
- Implement proper access controls
- Regularly backup data
- Consider data retention policies

## Monitoring and Logging

### Logging Configuration

The application uses a centralized logging system. Configure logging levels in `config/settings.yaml`:

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/av-catalog-converter.log
  max_size: 10485760  # 10MB
  backup_count: 5
```

### Monitoring

For production deployments, consider implementing:

- Health check endpoints
- Prometheus metrics
- Grafana dashboards
- Alert notifications

## Backup and Recovery

### Data Backup

Regularly backup:

- Configuration files
- Cache data
- Custom mappings and templates

### Disaster Recovery

1. Document recovery procedures
2. Test recovery processes regularly
3. Maintain backup deployment configurations
4. Implement automated backup solutions

For more information on specific deployment scenarios, please contact support.
