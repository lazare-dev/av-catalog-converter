# AV Catalog Converter

A powerful tool for converting and standardizing audio-visual equipment catalogs from various formats into a standardized format. The AV Catalog Converter uses advanced machine learning techniques to automatically detect and map fields, normalize values, and produce consistent output formats.

![AV Catalog Converter](docs/images/av-catalog-converter-logo.png)

## Features

- **Multi-Format Support**: Parse catalogs from multiple file formats (CSV, Excel, PDF, JSON, XML)
- **Intelligent Field Mapping**: Automatically detect file structure and map fields to standard formats using GPT-2 LLM technology
- **Category Normalization**: Extract and normalize product categories into a standardized hierarchy
- **Value Standardization**: Normalize prices, IDs, descriptions, and other text fields
- **Flexible Export Options**: Export to CSV, Excel, or JSON formats with customizable options
- **Web API**: RESTful API for integration with other systems
- **React UI**: Modern web interface for easy file processing
- **Performance Optimized**: Parallel processing, adaptive caching, and intelligent rate limiting for efficient operation
- **Extensible Architecture**: Easily add new parsers, normalizers, and export formats

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Node.js 14+ and npm (for frontend development)
- Docker and Docker Compose (recommended for production deployment)
- Tesseract OCR (optional, for PDF parsing with OCR capabilities)

### Quick Start with Docker (Recommended)

The easiest way to get started is using Docker, which includes all dependencies and configurations:

```bash
# Clone the repository
git clone https://github.com/yourusername/av-catalog-converter.git
cd av-catalog-converter

# Build and start the Docker containers
docker compose up -d

# Access the web UI at http://localhost:3000
# Access the API at http://localhost:8080
```

### Local Development Setup

For development purposes, you can set up the application locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/av-catalog-converter.git
   cd av-catalog-converter
   ```

2. **Set up a virtual environment**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r test-requirements.txt  # For development/testing
   ```

3. **Install additional dependencies**:

   For PDF parsing with OCR capabilities:
   - **Windows**: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

4. **Set up the frontend**:
   ```bash
   cd web/frontend
   npm install
   ```

5. **Start the development servers**:
   ```bash
   # Start the API server (from the project root)
   python app.py --api --port 8080

   # In another terminal, start the frontend development server
   cd web/frontend
   npm start
   ```

### Production Deployment

For production deployment, we recommend using Docker:

```bash
# Build optimized containers
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Access the application at your configured domain
```

For detailed deployment instructions, including cloud deployment options, see our [Deployment Guide](docs/deployment/deployment_guide.md).

## Usage

### Web Interface

The easiest way to use the AV Catalog Converter is through the web interface:

1. Access the web UI at http://localhost:3000 (or your configured domain)
2. Upload your catalog file
3. Review the automatically detected field mappings
4. Adjust mappings if needed
5. Configure output options
6. Process the file and download the standardized catalog

![Web Interface](docs/images/web-interface.png)

### Command Line Interface

For batch processing or integration into scripts, use the command line interface:

```bash
# Basic usage
python app.py --input path/to/catalog.csv --output path/to/output.csv

# Specify output format
python app.py --input path/to/catalog.xlsx --output path/to/output.json --format json

# Process a directory of files
python app.py --input path/to/catalogs/ --output path/to/outputs/ --format csv

# Enable verbose logging
python app.py --input path/to/catalog.csv --verbose
```

**Available Options**:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input file or directory path | (required) |
| `--output` | `-o` | Output file or directory path | input_name_standardized.csv |
| `--format` | `-f` | Output format: 'csv', 'excel', or 'json' | 'csv' |
| `--verbose` | `-v` | Enable verbose logging | False |
| `--config` | `-c` | Path to configuration file | config/settings.yaml |
| `--api` | `-a` | Run as API server | False |
| `--port` | `-p` | API server port | 8080 |
| `--workers` | `-w` | Number of worker processes | CPU count |
| `--no-cache` | | Disable caching | False |

### API Server

For integration with other applications, use the RESTful API:

```bash
# Start the API server
python app.py --api --port 8080
```

#### Core API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check endpoint |
| `/api/upload` | POST | Upload a file for processing |
| `/api/analyze` | POST | Analyze file structure |
| `/api/map` | GET/POST | Get or update field mappings |
| `/api/preview` | GET | Preview processed data |
| `/api/process` | POST | Process the file with mappings |
| `/api/download` | GET | Download the processed file |
| `/api/status` | GET | Check job status |

For detailed API documentation, see the [API Reference](docs/api_reference.md) or the [OpenAPI Specification](docs/openapi.yaml).

## Supported File Formats

The AV Catalog Converter supports a wide range of input formats:

| Format | Extensions | Features |
|--------|------------|----------|
| **CSV/TSV** | .csv, .tsv, .txt | Auto-detection of delimiters and encodings |
| **Excel** | .xlsx, .xls, .xlsm | Multi-sheet support, formula evaluation |
| **PDF** | .pdf | Table extraction, OCR for scanned documents |
| **JSON** | .json | Nested structure flattening |
| **XML** | .xml | XPath-based extraction |

## Output Format

The standardized output includes the following columns in order:

1. SKU
2. Short Description
3. Long Description
4. Model
5. Category Group
6. Category
7. Manufacturer
8. Manufacturer SKU
9. Image URL
10. Document Name
11. Document URL
12. Unit Of Measure
13. Buy Cost
14. Trade Price
15. MSRP GBP
16. MSRP USD
17. MSRP EUR
18. Discontinued

## Architecture

The AV Catalog Converter is built with a modular, extensible architecture:

![Architecture Diagram](docs/images/architecture.png)

### Project Structure

```
av-catalog-converter/
├── app.py                  # Main application entry point
├── config/                 # Configuration files
├── core/                   # Core functionality
│   ├── file_parser/        # File parsing modules
│   ├── llm/                # Language model clients
│   └── chunking/           # Data chunking utilities
├── services/               # Business logic services
│   ├── structure/          # Data structure analysis
│   ├── mapping/            # Field mapping
│   ├── category/           # Category extraction
│   └── normalization/      # Value normalization
├── utils/                  # Utility functions and helpers
│   ├── caching/            # Caching mechanisms
│   ├── parallel/           # Parallel processing utilities
│   └── rate_limiting/      # Rate limiting for API and LLM calls
├── web/                    # Web components
│   ├── api/                # API controllers and routes
│   └── frontend/           # React-based UI
├── tests/                  # Test cases
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docker/                 # Docker-related files
└── docs/                   # Documentation
```

For detailed architecture documentation, see the [Architecture Guide](docs/architecture/architecture_guide.md).

## Development

### Running Tests

We have comprehensive test coverage for all components of the AV Catalog Converter:

```bash
# Set up the test environment
chmod +x setup_test_env.sh
./setup_test_env.sh
source venv/bin/activate

# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/                # Unit tests only
python -m pytest tests/integration/         # Integration tests only
python -m pytest tests/ -n auto             # Parallel execution
python -m pytest tests/ --cov=.             # With coverage report

# Run tests for specific components
python -m pytest -m parallel                # Parallel processing tests
python -m pytest -m rate_limiting           # Rate limiting tests
python -m pytest -m caching                 # Caching tests
python -m pytest tests/unit/utils/          # All utility tests
```

For convenience, you can use the test runner script:

```bash
python run_tests.py --all                   # Run all tests
python run_tests.py --unit                  # Run unit tests
python run_tests.py --integration           # Run integration tests
python run_tests.py --coverage              # Generate coverage report
```

### Extending the Application

#### Adding a New Parser

1. Create a new parser class in `core/file_parser/` that inherits from `BaseParser`
2. Implement the required methods (`parse()`, `get_headers()`)
3. Add the parser to `ParserFactory` in `core/file_parser/parser_factory.py`
4. Add configuration settings in `config/settings.py`
5. Write unit tests for the new parser in `tests/unit/core/file_parser/`

#### Adding a New Normalizer

1. Create a new normalizer class in `services/normalization/` that inherits from `BaseNormalizer`
2. Implement the required methods (`normalize()`)
3. Register the normalizer in `services/normalization/normalizer_factory.py`
4. Add configuration settings in `config/settings.py`
5. Write unit tests for the new normalizer

#### Customizing LLM Integration

The application currently uses GPT-2 for LLM tasks, which provides a good balance of performance and efficiency. The LLM integration is modular and can be extended:

1. Create a new LLM client in `core/llm/` that inherits from `BaseLLMClient`
2. Implement the required methods (`generate_response()`, `batch_generate()`)
3. Register the client in `core/llm/llm_factory.py`
4. Update configuration settings in `config/settings.py`
5. Write unit tests for the new LLM client

Available LLM clients:
- `GPTClient`: Optimized for OpenAI's GPT-2 model
- `PhiClient`: Support for Microsoft's Phi-2 model (requires more memory)

### Continuous Integration and Deployment

This project uses GitHub Actions for CI/CD:

- **Continuous Integration**: Automatically runs tests, linting, and type checking on every push and pull request
- **Continuous Deployment**: Automatically builds and pushes Docker images on merges to main/master branch

The CI/CD pipeline configuration is in `.github/workflows/ci-cd.yml`.

## Documentation

### Comprehensive Documentation

For detailed documentation on all aspects of the AV Catalog Converter, see:

- [**HTML Documentation**](https://lazare-dev.github.io/av-catalog-converter/docs/html/index.html): Interactive, searchable documentation with copyable code examples
- [**Markdown Documentation**](docs/index.md): Comprehensive markdown documentation

### Component Documentation

- [**Architecture Guide**](docs/architecture/architecture_guide.md): Detailed architecture overview
- [**API Reference**](docs/api_reference.md): Complete API documentation
- [**Frontend Guide**](docs/frontend/frontend_guide.md): Frontend development documentation
- [**Performance Optimization**](docs/performance_optimization.md): Performance tuning guide
- [**Deployment Guide**](docs/deployment/deployment_guide.md): Deployment instructions
- [**Troubleshooting Guide**](docs/troubleshooting.md): Common issues and solutions

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest improvements.

## License

[MIT License](LICENSE)

## Acknowledgements

- This project uses [OpenAI's GPT-2](https://huggingface.co/gpt2) for field mapping and text analysis
- PDF parsing capabilities are provided by [PyPDF2](https://github.com/py-pdf/pypdf) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- The web interface is built with [React](https://reactjs.org/) and [Material-UI](https://mui.com/)
- Special thanks to all contributors who have helped improve this project
