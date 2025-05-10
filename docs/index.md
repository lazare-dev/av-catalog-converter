# AV Catalog Converter Documentation

Welcome to the AV Catalog Converter documentation. This documentation provides comprehensive information about the application, its features, and how to use it.

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
