# AV Catalog Converter

A tool for converting and standardizing audio-visual equipment catalogs from various formats into a standardized format.

## Features

- Parse catalogs from multiple file formats (CSV, Excel, PDF, JSON, XML)
- Automatically detect file structure and field mappings
- Extract and normalize product categories
- Standardize values (prices, IDs, text)
- Export to CSV, Excel, or JSON formats
- Web API for integration with other systems

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Docker and Docker Compose (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/av-catalog-converter.git
   cd av-catalog-converter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For PDF parsing with OCR capabilities, you'll need to install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

### Docker Setup

1. Build and start the Docker containers:
   ```bash
   docker compose up -d
   ```

2. Access the API at http://localhost:8080

3. Stop the containers:
   ```bash
   docker compose down
   ```

For more details on Docker setup, see [Docker README](docker/README.md).

## Usage

### Command Line Interface

Process a catalog file:

```
python app.py --input path/to/catalog.csv --output path/to/output.csv --format csv
```

Options:
- `--input`, `-i`: Input file path (required)
- `--output`, `-o`: Output file path (optional, defaults to input filename with "_standardized" suffix)
- `--format`, `-f`: Output format, either 'csv' or 'excel' (default: 'csv')
- `--verbose`, `-v`: Enable verbose logging
- `--api`, `-a`: Run as API server
- `--port`, `-p`: API server port (default: 8080)

### API Server

Start the API server:

```
python app.py --api --port 8080
```

#### API Endpoints

- `GET /api/health`: Health check endpoint
- `POST /api/upload`: Upload and process a catalog file
- `POST /api/analyze`: Analyze a catalog file structure without full processing
- `POST /api/map-fields`: Map fields from input columns to standardized format

## Supported File Formats

- CSV/TSV/Text files
- Excel (XLSX, XLS, XLSM)
- PDF (with table extraction and OCR capabilities)
- JSON
- XML

## Development

### Project Structure

- `app.py`: Main application entry point
- `config/`: Configuration files
- `core/`: Core functionality
  - `file_parser/`: File parsing modules
  - `llm/`: Language model clients
  - `chunking/`: Data chunking utilities
- `services/`: Business logic services
  - `structure/`: Data structure analysis
  - `mapping/`: Field mapping
  - `category/`: Category extraction
  - `normalization/`: Value normalization
- `utils/`: Utility functions and helpers
- `web/`: Web API controllers
- `tests/`: Test cases
- `docker/`: Docker-related files and scripts
- `.github/workflows/`: CI/CD pipeline configuration

### Running Tests

Run all tests:

```bash
python run_tests.py
```

Run only unit tests:

```bash
python run_tests.py --unit
```

Run only integration tests:

```bash
python run_tests.py --integration
```

Generate coverage report:

```bash
python run_tests.py --coverage
```

### Adding a New Parser

1. Create a new parser class in `core/file_parser/` that inherits from `BaseParser`
2. Implement the required methods (`parse()`)
3. Add the parser to `ParserFactory` in `core/file_parser/parser_factory.py`
4. Add configuration settings in `config/settings.py`
5. Write unit tests for the new parser in `tests/unit/core/file_parser/`

### Continuous Integration and Deployment

This project uses GitHub Actions for CI/CD:

- **Continuous Integration**: Automatically runs tests, linting, and type checking on every push and pull request
- **Continuous Deployment**: Automatically builds and pushes Docker images on merges to main/master branch

The CI/CD pipeline configuration is in `.github/workflows/ci-cd.yml`.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Documentation

For more detailed documentation with copyable commands, see the [AV-Catalog-Converter-Documentation.html](AV-Catalog-Converter-Documentation.html) file included in this repository. This HTML documentation includes:

- Comprehensive overview of all components
- Detailed installation and usage instructions
- Copy-paste functionality for all commands
- Troubleshooting tips
- API endpoint documentation with example requests
