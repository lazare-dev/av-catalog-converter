#!/usr/bin/env python3
"""
Test script for the AV Catalog Converter API
"""
import requests
import json
import sys
import os

def test_analyze_file(file_path):
    """
    Test the analyze_file endpoint
    """
    print(f"Testing analyze_file with {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
        
    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")
    
    # Upload file to analyze_file endpoint
    url = "http://localhost:8080/api/analyze-file"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(url, files=files)
            
        # Check response
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(json.dumps(data, indent=2))
            
            # Check structure info
            structure = data.get('structure', {})
            file_info = data.get('file_info', {})
            
            print("\nStructure Info:")
            print(f"  Row count: {structure.get('row_count', 'N/A')}")
            print(f"  Data rows: {structure.get('data_rows', 'N/A')}")
            print(f"  Effective rows: {structure.get('effective_rows', 'N/A')}")
            
            print("\nFile Info:")
            print(f"  Filename: {file_info.get('filename', 'N/A')}")
            print(f"  Size: {file_info.get('size', 'N/A') / (1024 * 1024):.2f} MB")
            print(f"  Parser: {file_info.get('parser', 'N/A')}")
            print(f"  Product count: {file_info.get('product_count', 'N/A')}")
            
            # Check if it's a KEF file
            if "KEF" in file_path.upper():
                print("\nKEF file detected, checking special handling:")
                print(f"  KEF price list flag: {file_info.get('kef_price_list', False)}")
                print(f"  Product count should be 247: {file_info.get('product_count', 0) == 247}")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <file_path>")
        sys.exit(1)
        
    test_analyze_file(sys.argv[1])
