#!/usr/bin/env python3
"""
Test script for API endpoints

This script tests the API endpoints of the AV Catalog Converter application.
It can be used to verify that the API is working correctly.
"""
import argparse
import json
import os
import requests
import sys
import time
from pprint import pprint

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    url = f"{base_url}/api/health"

    try:
        response = requests.get(url)
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Health check successful:")
            pprint(data)
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def test_upload_endpoint(base_url, file_path):
    """Test the upload-file endpoint (uploads without processing)"""
    print("\n=== Testing Upload File Endpoint ===")
    url = f"{base_url}/api/upload-file"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(url, files=files)

        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Upload successful:")
            pprint(data)
            return data.get('job_id')
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception: {str(e)}")
        return None

def test_analyze_endpoint_with_file(base_url, file_path):
    """Test the analyze endpoint with a file upload"""
    print("\n=== Testing Analyze Endpoint with File Upload ===")
    url = f"{base_url}/api/analyze"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(url, files=files)

        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Analysis successful:")
            print(f"Job ID: {data.get('job_id')}")
            print(f"Structure: {data.get('structure', {}).get('status', 'unknown')}")
            return data.get('job_id')
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception: {str(e)}")
        return None

def test_analyze_endpoint_with_job_id(base_url, job_id):
    """Test the analyze endpoint with a job ID"""
    print(f"\n=== Testing Analyze Endpoint with Job ID: {job_id} ===")
    url = f"{base_url}/api/analyze"

    try:
        headers = {'Content-Type': 'application/json'}
        data = {'job_id': job_id}
        response = requests.post(url, headers=headers, json=data)

        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Analysis successful:")
            print(f"Job ID: {data.get('job_id')}")
            print(f"Structure: {data.get('structure', {}).get('status', 'unknown')}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def test_map_fields_endpoint(base_url, job_id):
    """Test the map-fields endpoint"""
    print(f"\n=== Testing Map Fields Endpoint with Job ID: {job_id} ===")
    url = f"{base_url}/api/map-fields"

    try:
        # First, get the mapping data
        mapping_url = f"{base_url}/api/get-mapping-data/{job_id}"
        mapping_response = requests.get(mapping_url)

        if mapping_response.status_code != 200:
            print(f"Error getting mapping data: {mapping_response.status_code}")
            print(mapping_response.text)
            return False

        mapping_data = mapping_response.json()
        columns = mapping_data.get('columns', [])
        sample_data = mapping_data.get('sample_data', {})

        # Convert sample data to the format expected by the map-fields endpoint
        sample_rows = []
        if columns and sample_data:
            # Get the first 5 samples for each column
            sample_count = min(5, len(next(iter(sample_data.values()), [])))

            for i in range(sample_count):
                row = {}
                for col in columns:
                    if col in sample_data and len(sample_data[col]) > i:
                        row[col] = sample_data[col][i]
                    else:
                        row[col] = ""
                sample_rows.append(row)

        # Call the map-fields endpoint
        headers = {'Content-Type': 'application/json'}
        data = {
            'columns': columns,
            'sample_data': sample_rows
        }

        response = requests.post(url, headers=headers, json=data)

        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Field mapping successful:")
            print(f"Mappings count: {len(data.get('mappings', []))}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test API endpoints')
    parser.add_argument('--url', type=str, default='http://localhost:8080',
                        help='Base URL of the API')
    parser.add_argument('--file', type=str, help='Path to a test file')
    args = parser.parse_args()

    # Test health endpoint
    if not test_health_endpoint(args.url):
        print("Health check failed, exiting")
        return 1

    # If a file is provided, test the upload and analyze endpoints
    if args.file:
        print("\n=== Testing Workflow 1: Upload File First, Then Analyze ===")
        # Test upload endpoint (upload-file)
        job_id = test_upload_endpoint(args.url, args.file)
        if not job_id:
            print("Upload failed, exiting")
            return 1

        # Wait a moment for the server to process the file
        time.sleep(1)

        # Test analyze endpoint with job ID
        if not test_analyze_endpoint_with_job_id(args.url, job_id):
            print("Analysis with job ID failed, exiting")
            return 1

        # Test map-fields endpoint
        if not test_map_fields_endpoint(args.url, job_id):
            print("Field mapping failed, exiting")
            return 1

        print("\n=== Testing Workflow 2: Direct Analysis with File Upload ===")
        # Test analyze endpoint with direct file upload
        job_id = test_analyze_endpoint_with_file(args.url, args.file)
        if not job_id:
            print("Direct analysis with file upload failed, exiting")
            return 1

    print("\nAll tests completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
