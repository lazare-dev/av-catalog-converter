# AV Catalog Converter API Reference

This document provides detailed information about the AV Catalog Converter API endpoints, request/response formats, and examples.

## Overview

The AV Catalog Converter provides a RESTful API for converting various audio-visual equipment catalogs into a standardized format. The API is designed to be easy to use and integrate with other systems.

## Base URL

When running locally:
```
http://localhost:8080/api
```

When deployed:
```
https://your-deployment-url.com/api
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
  "app_name": "AV Catalog Standardizer",
  "timestamp": 1625097600,
  "environment": "production",
  "llm_info": {
    "model_type": "distilbert",
    "is_initialized": true,
    "cache_hits": 0,
    "rate_limited_count": 0,
    "rate_limited_wait_time": 0
  }
}
```

**Status Codes:**
- `200 OK`: Service is running properly
- `500 Internal Server Error`: Service is experiencing issues

### Upload File

Upload a catalog file for processing.

**Endpoint:** `POST /api/upload`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The catalog file to process (required)
  - `format`: Output format - one of `csv`, `excel`, or `json` (optional, default: `csv`)

**Response:**
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "File uploaded and processed successfully",
  "file_info": {
    "filename": "example_catalog.xlsx",
    "size": 1024000,
    "rows": 150,
    "columns": 25
  }
}
```

**Status Codes:**
- `200 OK`: File uploaded successfully
- `400 Bad Request`: Missing file or invalid format
- `413 Payload Too Large`: File too large
- `500 Internal Server Error`: Processing error

**Example using cURL:**
```bash
curl -X POST -F "file=@path/to/your/catalog.csv" -F "format=csv" http://localhost:8080/api/upload
```

**Example using Python:**
```python
import requests

url = "http://localhost:8080/api/upload"
files = {"file": open("path/to/your/catalog.csv", "rb")}
data = {"format": "csv"}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    job_id = result["job_id"]
    print(f"File uploaded successfully. Job ID: {job_id}")
else:
    print(f"Error: {response.json().get('error')}")
```

### Analyze File

Analyze a catalog file structure without full processing.

**Endpoint:** `POST /api/analyze`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The catalog file to analyze (required)
  - `job_id`: Optional job ID for tracking (optional)

**Response:**
```json
{
  "structure": {
    "column_count": 25,
    "row_count": 151,
    "header_row": 0,
    "data_rows": 150,
    "effective_rows": 150,
    "column_types": {
      "Product Code": "string",
      "Description": "string",
      "Price": "numeric"
    }
  },
  "sample_data": [
    {
      "Product Code": "ABC123",
      "Description": "4K Projector",
      "Price": "1299.99"
    },
    {
      "Product Code": "DEF456",
      "Description": "Wireless Speaker",
      "Price": "299.99"
    }
  ],
  "columns": ["Product Code", "Description", "Price"],
  "file_info": {
    "filename": "example_catalog.xlsx",
    "size": 1024000,
    "parser": "ExcelParser",
    "product_count": 150
  },
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- `200 OK`: File analyzed successfully
- `400 Bad Request`: Missing file
- `404 Not Found`: File not found
- `413 Payload Too Large`: File too large
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
    print(f"Product count: {analysis['file_info']['product_count']}")
    print(f"Job ID: {analysis['job_id']}")
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
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "columns": ["Product Code", "Description", "Price"],
  "manual_mappings": {
    "Product Code": "SKU",
    "Description": "Short Description",
    "Price": "Trade Price"
  }
}
```

**Response:**
```json
{
  "mappings": {
    "Product Code": {
      "field": "SKU",
      "confidence": 0.95,
      "source": "manual"
    },
    "Description": {
      "field": "Short Description",
      "confidence": 0.95,
      "source": "manual"
    },
    "Price": {
      "field": "Trade Price",
      "confidence": 0.95,
      "source": "manual"
    }
  },
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- `200 OK`: Fields mapped successfully
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Mapping error

**Example using cURL:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"job_id":"550e8400-e29b-41d4-a716-446655440000","columns":["Product Code","Description","Price"],"manual_mappings":{"Product Code":"SKU","Description":"Short Description","Price":"Trade Price"}}' http://localhost:8080/api/map-fields
```

**Example using Python:**
```python
import requests

url = "http://localhost:8080/api/map-fields"
data = {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "columns": ["Product Code", "Description", "Price"],
    "manual_mappings": {
        "Product Code": "SKU",
        "Description": "Short Description",
        "Price": "Trade Price"
    }
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Mapped {len(result['mappings'])} fields")
    print(f"Job ID: {result['job_id']}")
else:
    print(f"Error: {response.json().get('error')}")
```

### Process File

Process the file with the specified mappings.

**Endpoint:** `POST /api/process/{job_id}`

**URL Parameters:**
- `job_id`: Job ID from previous steps (required)

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "mappings": {
    "Product Code": {
      "field": "SKU",
      "confidence": 0.95,
      "source": "manual"
    },
    "Description": {
      "field": "Short Description",
      "confidence": 0.95,
      "source": "manual"
    },
    "Price": {
      "field": "Trade Price",
      "confidence": 0.95,
      "source": "manual"
    }
  },
  "output_format": "csv"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "File processed successfully",
  "stats": {
    "input_rows": 150,
    "output_rows": 150,
    "processing_time_ms": 1250
  },
  "download_url": "/api/download/550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- `200 OK`: File processed successfully
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Processing error

### Download Processed File

Download the processed file.

**Endpoint:** `GET /api/download/{job_id}`

**URL Parameters:**
- `job_id`: Job ID from previous steps (required)

**Response:**
- Content-Type: Depends on the requested format
- Body: The processed file as an attachment

**Status Codes:**
- `200 OK`: File downloaded successfully
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Download error

### Get Job Status

Check the status of a processing job.

**Endpoint:** `GET /api/status/{job_id}`

**URL Parameters:**
- `job_id`: Job ID to check (required)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "message": "Processing complete",
  "created_at": "2023-01-15T12:30:45Z",
  "updated_at": "2023-01-15T12:31:15Z"
}
```

**Status Codes:**
- `200 OK`: Status retrieved successfully
- `404 Not Found`: Job ID not found

### Preview Processed Data

Preview the processed data before downloading.

**Endpoint:** `GET /api/preview/{job_id}`

**URL Parameters:**
- `job_id`: Job ID from previous steps (required)

**Query Parameters:**
- `limit`: Maximum number of rows to return (optional, default: 10)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_rows": 150,
  "preview_rows": 10,
  "data": [
    {
      "SKU": "ABC123",
      "Short Description": "4K Projector",
      "Long Description": "Professional 4K projector with 5000 lumens",
      "Model": "XG-500",
      "Category Group": "Display",
      "Category": "Projectors",
      "Manufacturer": "Sony",
      "Manufacturer SKU": "VPL-XG500",
      "Image URL": "https://example.com/images/projector.jpg",
      "Document Name": "User Manual",
      "Document URL": "https://example.com/docs/manual.pdf",
      "Unit Of Measure": "Each",
      "Buy Cost": "1000.00",
      "Trade Price": "1299.99",
      "MSRP GBP": "1499.99",
      "MSRP USD": "1999.99",
      "MSRP EUR": "1799.99",
      "Discontinued": "No"
    }
  ]
}
```

**Status Codes:**
- `200 OK`: Preview retrieved successfully
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Preview error

### Cleanup Job Resources

Clean up resources associated with a job.

**Endpoint:** `DELETE /api/cleanup/{job_id}`

**URL Parameters:**
- `job_id`: Job ID to clean up (required)

**Response:**
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Job resources cleaned up successfully"
}
```

**Status Codes:**
- `200 OK`: Resources cleaned up successfully
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Cleanup error

## Error Handling

All API endpoints return appropriate HTTP status codes and error messages in a consistent format:

```json
{
  "error": "Processing failed",
  "details": "Invalid file format",
  "suggestions": [
    "Check that the file format is supported",
    "Ensure the file is not corrupted",
    "Try a different file format"
  ]
}
```

Common error codes:
- `200 OK`: Request successful
- `400 Bad Request`: Missing or invalid input
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: Uploaded file is too large
- `500 Internal Server Error`: Server-side error

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

## Rate Limiting

The API implements rate limiting to prevent overloading the LLM components. The default configuration allows:
- 60 requests per minute
- Burst size of 3000 tokens
- Fallback mechanisms when rate limited

## Versioning

The API version is included in the health check response. Future versions may include the version in the URL path (e.g., `/api/v2/upload`).
