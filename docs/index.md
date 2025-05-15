# AV Catalog Converter Documentation

Welcome to the AV Catalog Converter documentation. This documentation provides comprehensive information about the application, its features, and how to use it.

## Overview

The AV Catalog Converter is a powerful tool for standardizing audio-visual equipment catalogs from various manufacturers into a consistent format. It uses advanced machine learning techniques to automatically detect and map fields, normalize values, and produce standardized output formats.

## Key Features

- **Multi-Format Support**: Parse catalogs from multiple file formats (CSV, Excel, PDF, JSON, XML)
- **Intelligent Field Mapping**: Automatically detect file structure and map fields to standard formats using DistilBERT LLM technology
- **Category Normalization**: Extract and normalize product categories into a standardized hierarchy
- **Value Standardization**: Normalize prices, IDs, descriptions, and other text fields
- **Flexible Export Options**: Export to CSV, Excel, or JSON formats with customizable options
- **Web API**: RESTful API for integration with other systems
- **Performance Optimized**: Parallel processing, adaptive caching, and intelligent rate limiting for efficient operation
- **Extensible Architecture**: Easily add new parsers, normalizers, and export formats
- **Comprehensive Logging**: Detailed logging throughout the application for troubleshooting

## Table of Contents

### Getting Started
- [Installation](../README.md#installation)
- [Usage](../README.md#usage)
- [Docker Setup](../docker/README.md)

### API Reference
- [API Documentation](api_reference.md)
- [OpenAPI Specification](openapi.yaml)

### Development
- [Project Structure](../README.md#project-structure)
- [Adding a New Parser](../README.md#adding-a-new-parser)
- [Running Tests](../README.md#running-tests)
- [Continuous Integration and Deployment](../README.md#continuous-integration-and-deployment)

### Performance
- [Performance Optimization Guide](performance_optimization.md)
- [Profiling Tools](performance_optimization.md#profiling-tools)
- [Memory Optimization](performance_optimization.md#memory-optimization)

### LLM Integration
- [LLM Overview](#llm-integration)
- [Field Mapping with LLM](#field-mapping-with-llm)
- [Text Analysis](#text-analysis)

### Examples
- [Command Line Examples](#command-line-examples)
- [API Examples](#api-examples)
- [Integration Examples](#integration-examples)

## Command Line Examples

### Basic Usage

Process a CSV file and output to CSV:

```bash
python app.py --input data/input/catalog.csv --output data/output/standardized.csv
```

Process an Excel file and output to JSON:

```bash
python app.py --input data/input/catalog.xlsx --output data/output/standardized.json --format json
```

Process a PDF file with verbose logging:

```bash
python app.py --input data/input/catalog.pdf --output data/output/standardized.csv --verbose
```

### Batch Processing

Process all files in a directory:

```bash
python app.py --input data/input --output data/output --format csv
```

### API Server

Start the API server:

```bash
python app.py --api --port 8080
```

## API Examples

### Upload and Process a File

Using cURL:

```bash
curl -X POST -F "file=@data/input/catalog.csv" -F "format=csv" http://localhost:8080/api/upload -o standardized_catalog.csv
```

Using Python:

```python
import requests

url = "http://localhost:8080/api/upload"
files = {"file": open("data/input/catalog.csv", "rb")}
data = {"format": "csv"}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open("standardized_catalog.csv", "wb") as f:
        f.write(response.content)
    print("File processed successfully")
else:
    print(f"Error: {response.json().get('error')}")
```

### Analyze a File

Using cURL:

```bash
curl -X POST -F "file=@data/input/catalog.csv" http://localhost:8080/api/analyze
```

Using Python:

```python
import requests
import json

url = "http://localhost:8080/api/analyze"
files = {"file": open("data/input/catalog.csv", "rb")}

response = requests.post(url, files=files)

if response.status_code == 200:
    analysis = response.json()
    print(json.dumps(analysis, indent=2))
else:
    print(f"Error: {response.json().get('error')}")
```

### Map Fields

Using cURL:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"columns":["item_sku","item_name","item_price","item_category"],"sample_data":[{"item_sku":"ABC123","item_name":"HD Camera","item_price":"299.99","item_category":"Video"}]}' http://localhost:8080/api/map-fields
```

Using Python:

```python
import requests
import json

url = "http://localhost:8080/api/map-fields"
data = {
    "columns": ["item_sku", "item_name", "item_price", "item_category"],
    "sample_data": [
        {
            "item_sku": "ABC123",
            "item_name": "HD Camera",
            "item_price": "299.99",
            "item_category": "Video"
        }
    ]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    mappings = response.json()
    print(json.dumps(mappings, indent=2))
else:
    print(f"Error: {response.json().get('error')}")
```

## Integration Examples

### Integration with Python Scripts

```python
from app import process_file

# Process a file
data, error = process_file("data/input/catalog.csv")

if error:
    print(f"Error: {error}")
else:
    # Use the processed data
    print(f"Processed {len(data)} rows with {len(data.columns)} columns")

    # Save to CSV
    data.to_csv("data/output/standardized.csv", index=False)
```

### Integration with Web Applications

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/process-catalog', methods=['POST'])
def process_catalog():
    # Get the file from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Send the file to the AV Catalog Converter API
    converter_url = "http://localhost:8080/api/upload"
    files = {"file": file}
    data = {"format": "json"}

    response = requests.post(converter_url, files=files, data=data)

    if response.status_code == 200:
        # Process the standardized data
        standardized_data = response.json()

        # Do something with the data
        result = process_standardized_data(standardized_data)

        return jsonify(result)
    else:
        return jsonify({'error': response.json().get('error')}), response.status_code

if __name__ == '__main__':
    app.run(port=5000)
```

### Integration with Batch Processing Systems

```python
import os
import subprocess
import logging

def batch_process_catalogs(input_dir, output_dir):
    """Process all catalog files in a directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the input directory
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"standardized_{file}")

        # Process the file
        try:
            subprocess.run([
                "python", "app.py",
                "--input", input_path,
                "--output", output_path,
                "--format", "csv"
            ], check=True)

            logging.info(f"Successfully processed {file}")

        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing {file}: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    batch_process_catalogs("data/input", "data/output")
```

## LLM Integration

The AV Catalog Converter uses DistilBERT with 8-bit quantization for field mapping and text analysis. This provides several benefits:

1. **Efficient Field Mapping**: The LLM can understand the semantic meaning of column names and map them to the standardized schema, even when the names are different.

2. **Text Analysis**: The LLM can analyze product descriptions and extract relevant information such as categories, features, and specifications.

3. **Performance Optimization**: The use of 8-bit quantization reduces memory usage while maintaining accuracy.

4. **Adaptive Caching**: The LLM responses are cached to improve performance for similar requests.

5. **Rate Limiting**: The LLM requests are rate-limited to prevent overloading the system.

### Field Mapping with LLM

The LLM is used to map fields from the input catalog to the standardized schema. For example, if the input catalog has a column named "Item Number", the LLM can recognize that this corresponds to the "SKU" field in the standardized schema.

The field mapping process works as follows:

1. The LLM analyzes the column names and sample data from the input catalog.
2. It compares the column names and data to the standardized schema fields.
3. It assigns a confidence score to each potential mapping.
4. The mappings with the highest confidence scores are used to transform the data.

### Text Analysis

The LLM is also used for text analysis, particularly for extracting information from product descriptions. For example:

1. **Category Extraction**: The LLM can analyze a product description and determine the appropriate category and category group.
2. **Feature Extraction**: The LLM can identify key features and specifications from the description.
3. **Normalization**: The LLM can normalize text fields to ensure consistency.

## Output Format

The standardized output includes the following columns in this specific order:

1. **SKU** (required) - Stock Keeping Unit
2. **Short Description** (required) - Brief product description
3. **Long Description** - Detailed product description
4. **Model** - Product model number
5. **Category Group** - Higher-level product category
6. **Category** - Specific product category
7. **Manufacturer** (required) - Company that manufactures the product
8. **Manufacturer SKU** - Manufacturer's own product identifier
9. **Image URL** - Link to product image
10. **Document Name** - Name of associated document
11. **Document URL** - Link to associated document
12. **Unit Of Measure** - How the product is measured/sold
13. **Buy Cost** - Wholesale/purchase cost
14. **Trade Price** (required) - Price for trade customers
15. **MSRP GBP** - Manufacturer's Suggested Retail Price in British Pounds
16. **MSRP USD** - Manufacturer's Suggested Retail Price in US Dollars
17. **MSRP EUR** - Manufacturer's Suggested Retail Price in Euros
18. **Discontinued** - Whether the product is discontinued
