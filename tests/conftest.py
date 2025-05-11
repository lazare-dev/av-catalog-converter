"""
Test configuration for pytest
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import optimized component fixtures
from tests.conftest_optimized import (
    small_csv_file, medium_csv_file, small_excel_file, medium_excel_file,
    mock_rate_limiter, mock_adaptive_cache, mock_phi_client, mock_parallel_processor
)

# Define fixtures that can be reused across tests

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return pd.DataFrame({
        'SKU': ['ABC123', 'DEF456', 'GHI789'],
        'Product Name': ['HD Camera', 'Wireless Mic', 'Audio Mixer'],
        'Price': [299.99, 149.50, 499.00],
        'Category': ['Video', 'Audio', 'Audio'],
        'Manufacturer': ['Sony', 'Shure', 'Yamaha']
    })

@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing"""
    return {
        "products": [
            {
                "sku": "ABC123",
                "name": "HD Camera",
                "price": 299.99,
                "category": "Video",
                "manufacturer": "Sony"
            },
            {
                "sku": "DEF456",
                "name": "Wireless Mic",
                "price": 149.50,
                "category": "Audio",
                "manufacturer": "Shure"
            },
            {
                "sku": "GHI789",
                "name": "Audio Mixer",
                "price": 499.00,
                "category": "Audio",
                "manufacturer": "Yamaha"
            }
        ]
    }

@pytest.fixture
def sample_xml_data():
    """Sample XML data for testing"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<catalog>
    <product>
        <sku>ABC123</sku>
        <name>HD Camera</name>
        <price>299.99</price>
        <category>Video</category>
        <manufacturer>Sony</manufacturer>
    </product>
    <product>
        <sku>DEF456</sku>
        <name>Wireless Mic</name>
        <price>149.50</price>
        <category>Audio</category>
        <manufacturer>Shure</manufacturer>
    </product>
    <product>
        <sku>GHI789</sku>
        <name>Audio Mixer</name>
        <price>499.00</price>
        <category>Audio</category>
        <manufacturer>Yamaha</manufacturer>
    </product>
</catalog>
"""

@pytest.fixture
def sample_yaml_data():
    """Sample YAML data for testing"""
    return """
products:
  - sku: ABC123
    name: HD Camera
    price: 299.99
    category: Video
    manufacturer: Sony
  - sku: DEF456
    name: Wireless Mic
    price: 149.50
    category: Audio
    manufacturer: Shure
  - sku: GHI789
    name: Audio Mixer
    price: 499.00
    category: Audio
    manufacturer: Yamaha
"""

@pytest.fixture
def temp_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file for testing"""
    file_path = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def temp_json_file(sample_json_data, tmp_path):
    """Create a temporary JSON file for testing"""
    import json
    file_path = tmp_path / "test_data.json"
    with open(file_path, 'w') as f:
        json.dump(sample_json_data, f)
    return file_path

@pytest.fixture
def temp_xml_file(sample_xml_data, tmp_path):
    """Create a temporary XML file for testing"""
    file_path = tmp_path / "test_data.xml"
    with open(file_path, 'w') as f:
        f.write(sample_xml_data)
    return file_path

@pytest.fixture
def temp_yaml_file(sample_yaml_data, tmp_path):
    """Create a temporary YAML file for testing"""
    file_path = tmp_path / "test_data.yaml"
    with open(file_path, 'w') as f:
        f.write(sample_yaml_data)
    return file_path
