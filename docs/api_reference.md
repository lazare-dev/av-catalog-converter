# AV Catalog Converter API Reference

This document provides detailed information about the AV Catalog Converter API endpoints, request/response formats, and examples.

## Base URL

When running locally:
```
http://localhost:8080
```

When deployed:
```
https://your-deployment-url.com
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing an authentication mechanism.

## API Endpoints

### Health Check

Check if the API is running and get basic information about the service.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "app_name": "AV Catalog Converter"
}
```

**Status Codes:**
- `200 OK`: Service is running properly

### Upload and Process File

Upload a catalog file and convert it to a standardized format.

**Endpoint:** `POST /api/upload`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The catalog file to process (required)
  - `format`: Output format - one of `csv`, `excel`, or `json` (optional, default: `csv`)

**Response:**
- Content-Type: Depends on the requested format
- Body: The processed file as an attachment

**Status Codes:**
- `200 OK`: File processed successfully
- `400 Bad Request`: Missing file or invalid format
- `500 Internal Server Error`: Processing error

**Example using cURL:**
```bash
curl -X POST -F "file=@path/to/your/catalog.csv" -F "format=csv" http://localhost:8080/api/upload -o standardized_catalog.csv
```

**Example using Python:**
```python
import requests

url = "http://localhost:8080/api/upload"
files = {"file": open("path/to/your/catalog.csv", "rb")}
data = {"format": "csv"}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open("standardized_catalog.csv", "wb") as f:
        f.write(response.content)
    print("File processed successfully")
else:
    print(f"Error: {response.json().get('error')}")
```

### Analyze File Structure

Analyze a catalog file's structure without fully processing it.

**Endpoint:** `POST /api/analyze`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The catalog file to analyze (required)

**Response:**
- Content-Type: `application/json`
- Body:
```json
{
  "structure": {
    "column_types": {
      "SKU": {"type": "id"},
      "Product Name": {"type": "text"},
      "Price": {"type": "price"},
      "Category": {"type": "category"}
    },
    "row_count": 100,
    "column_count": 4
  },
  "sample_data": [
    {
      "SKU": "ABC123",
      "Product Name": "HD Camera",
      "Price": 299.99,
      "Category": "Video"
    },
    {
      "SKU": "DEF456",
      "Product Name": "Wireless Mic",
      "Price": 149.50,
      "Category": "Audio"
    }
  ],
  "columns": ["SKU", "Product Name", "Price", "Category"]
}
```

**Status Codes:**
- `200 OK`: File analyzed successfully
- `400 Bad Request`: Missing file
- `500 Internal Server Error`: Analysis error

**Example using cURL:**
```bash
curl -X POST -F "file=@path/to/your/catalog.csv" http://localhost:8080/api/analyze
```

**Example using Python:**
```python
import requests

url = "http://localhost:8080/api/analyze"
files = {"file": open("path/to/your/catalog.csv", "rb")}

response = requests.post(url, files=files)

if response.status_code == 200:
    analysis = response.json()
    print(f"File has {analysis['structure']['row_count']} rows and {analysis['structure']['column_count']} columns")
else:
    print(f"Error: {response.json().get('error')}")
```

### Map Fields

Map fields from input columns to standardized format.

**Endpoint:** `POST /api/map-fields`

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "columns": ["item_sku", "item_name", "item_price", "item_category"],
  "sample_data": [
    {
      "item_sku": "ABC123",
      "item_name": "HD Camera",
      "item_price": "299.99",
      "item_category": "Video"
    },
    {
      "item_sku": "DEF456",
      "item_name": "Wireless Mic",
      "item_price": "149.50",
      "item_category": "Audio"
    }
  ]
}
```

**Response:**
- Content-Type: `application/json`
- Body:
```json
{
  "mappings": [
    {
      "source_field": "item_sku",
      "target_field": "SKU",
      "confidence": 0.95
    },
    {
      "source_field": "item_name",
      "target_field": "Product Name",
      "confidence": 0.9
    },
    {
      "source_field": "item_price",
      "target_field": "Price",
      "confidence": 0.85
    },
    {
      "source_field": "item_category",
      "target_field": "Category",
      "confidence": 1.0
    }
  ]
}
```

**Status Codes:**
- `200 OK`: Fields mapped successfully
- `400 Bad Request`: Invalid request data
- `500 Internal Server Error`: Mapping error

**Example using cURL:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"columns":["item_sku","item_name","item_price","item_category"],"sample_data":[{"item_sku":"ABC123","item_name":"HD Camera","item_price":"299.99","item_category":"Video"}]}' http://localhost:8080/api/map-fields
```

**Example using Python:**
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
    print(f"Mapped {len(mappings['mappings'])} fields")
else:
    print(f"Error: {response.json().get('error')}")
```

## Error Handling

All API endpoints return appropriate HTTP status codes and error messages in a consistent format:

```json
{
  "error": "Detailed error message"
}
```

Common error codes:
- `400 Bad Request`: Missing or invalid input
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

## Rate Limiting

Currently, there are no rate limits implemented. For production deployments, consider adding rate limiting to prevent abuse.

## Versioning

The API version is included in the health check response. Future versions may include the version in the URL path (e.g., `/api/v2/upload`).
