# AV Catalog Converter Documentation

## Codebase Overview

The AV (Audio-Visual) Catalog Converter is a comprehensive tool designed to standardize and convert various catalog formats. This document provides a complete guide to understanding and running all components of the system.

### Core Structure

1. **File Parsers**
   - Support for CSV, Excel, JSON, PDF, and XML files
   - Automatic format detection and appropriate parser selection
   - Extensible architecture for adding new parsers

2. **Structure Analysis**
   - Data boundary detection
   - Header detection and analysis
   - Structure pattern recognition

3. **Field Mapping**
   - Direct mapping based on field names
   - Pattern-based mapping for common formats
   - Semantic mapping using machine learning
   - Field definition management

4. **Category Extraction**
   - Hierarchical category analysis
   - Taxonomy mapping to standard categories
   - Category normalization

5. **Value Normalization**
   - Price normalization (currency, format)
   - ID normalization
   - Text normalization
   - Unit conversion

### Architecture

1. **Web API**
   - RESTful API built with Flask
   - Swagger documentation
   - File upload and processing endpoints
   - Analysis and mapping endpoints

2. **CLI Interface**
   - Command-line interface for batch processing
   - Support for various input and output formats
   - Verbose logging options

3. **Modular Design**
   - Separate modules for parsing, analysis, mapping, etc.
   - Factory patterns for component creation
   - Dependency injection for flexibility

### Technologies

- **Python 3.8+**: Core programming language
- **Pandas/NumPy**: Data processing and manipulation
- **Flask**: Web API framework
- **Machine Learning**: Uses transformers and PyTorch for intelligent mapping
- **Docker**: Containerization support
- **Tesseract OCR**: For extracting text from PDFs

## How to Run Everything

### 1. Installation

#### Option A: Direct Installation

```bash
# Install dependencies and the package
python install.py
```

#### Option B: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### Option C: Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### 2. Running the Application

#### As a Web API Server

```bash
# Method 1: Using the installed package
av-catalog-converter --api --port 8080

# Method 2: Running the Python script directly
python app.py --api --port 8080

# Method 3: Using Docker
docker-compose up
```

The API will be available at http://localhost:8080 with Swagger documentation at http://localhost:8080/api/docs/

#### As a Command-Line Tool

```bash
# Method 1: Using the installed package
av-catalog-converter --input path/to/input.csv --output path/to/output.csv --format csv

# Method 2: Running the Python script directly
python app.py --input path/to/input.csv --output path/to/output.csv --format csv
```

### 3. Running Tests

```bash
# Run all tests
python run_tests.py --all

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Generate coverage report
python run_tests.py --all --coverage
```

### 4. Profiling Performance

```bash
# Run the profiler
python profile_app.py

# Analyze the profile results
python analyze_profile.py
```

### 5. Docker Operations

```bash
# Build the Docker image
docker build -t av-catalog-converter .

# Run the container
docker run -p 8080:8080 av-catalog-converter

# Run with volume mounts for data
docker run -p 8080:8080 \
  -v $(pwd)/data/input:/app/data/input \
  -v $(pwd)/data/output:/app/data/output \
  -v $(pwd)/logs:/app/logs \
  av-catalog-converter
```

## API Endpoints

### Health Check
- **URL**: `/api/health`
- **Method**: `GET`
- **Description**: Checks if the API is running properly
- **Response**: Basic information about the application status

### Upload File
- **URL**: `/api/upload`
- **Method**: `POST`
- **Description**: Uploads and processes a catalog file
- **Parameters**:
  - `file`: The file to upload
  - `format`: Output format (csv, excel, json)
- **Response**: The processed file as an attachment

### Analyze File
- **URL**: `/api/analyze`
- **Method**: `POST`
- **Description**: Analyzes a file's structure without full processing
- **Parameters**:
  - `file`: The file to analyze
- **Response**: JSON with file structure analysis

### Map Fields
- **URL**: `/api/map-fields`
- **Method**: `POST`
- **Description**: Maps fields from input columns to standardized format
- **Parameters**:
  - `columns`: Array of column names
  - `sample_data`: Array of sample data objects
- **Response**: JSON with field mappings

## Project Structure

```
av-catalog-converter/
├── app.py                  # Main application entry point
├── config/                 # Configuration files
├── core/                   # Core functionality
│   ├── chunking/           # Data chunking modules
│   ├── file_parser/        # File parsing modules
│   └── llm/                # Language model clients
├── prompts/                # LLM prompts and templates
├── services/               # Business logic services
│   ├── category/           # Category extraction
│   ├── mapping/            # Field mapping
│   ├── normalization/      # Value normalization
│   └── structure/          # Structure analysis
├── tests/                  # Test suite
│   ├── integration/        # Integration tests
│   └── unit/               # Unit tests
├── utils/                  # Utility modules
│   ├── caching/            # Caching utilities
│   ├── error_handling/     # Error handling
│   ├── helpers/            # Helper functions
│   ├── logging/            # Logging utilities
│   ├── parsers/            # Utility parsers
│   └── profiling/          # Performance profiling
├── web/                    # Web API components
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── install.py              # Installation script
├── requirements.txt        # Dependencies
├── run_tests.py            # Test runner
└── setup.py                # Package setup
```

## Key Components

### File Parsers
The system supports multiple file formats through specialized parsers:
- `CSVParser`: Handles CSV files with various delimiters
- `ExcelParser`: Processes Excel files (xls, xlsx)
- `JSONParser`: Parses JSON files and APIs
- `XMLParser`: Handles XML catalog formats
- `PDFParser`: Extracts tables from PDF files

### Structure Analysis
- `StructureAnalyzer`: Determines the overall structure of the data
- `HeaderDetector`: Identifies header rows in tabular data
- `DataBoundaryDetector`: Finds where actual data begins and ends

### Field Mapping
- `FieldMapper`: Orchestrates the mapping process
- `DirectMapper`: Maps fields based on exact or fuzzy name matches
- `PatternMapper`: Uses regex patterns to identify field types
- `SemanticMapper`: Uses ML to understand field semantics

### Category Extraction
- `CategoryExtractor`: Identifies product categories
- `HierarchyAnalyzer`: Analyzes category hierarchies
- `TaxonomyMapper`: Maps to standardized taxonomy

### Value Normalization
- `ValueNormalizer`: Orchestrates normalization of all values
- `PriceNormalizer`: Standardizes price formats and currencies
- `TextNormalizer`: Cleans and standardizes text fields
- `UnitNormalizer`: Converts units to standard formats
- `IDNormalizer`: Standardizes product IDs

## Conclusion

The AV Catalog Converter is a comprehensive solution for standardizing audio-visual equipment catalogs from various formats. With its modular design, it can be extended to support additional formats and standardization requirements. The system can be run as a web service or command-line tool, making it flexible for different use cases.
