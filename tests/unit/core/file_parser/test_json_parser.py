"""
Unit tests for the JSON parser
"""
import pytest
import pandas as pd
import json
from pathlib import Path

from core.file_parser.json_parser import JSONParser


class TestJSONParser:
    """Test cases for JSONParser"""

    def test_init(self, temp_json_file):
        """Test initialization"""
        parser = JSONParser(temp_json_file)
        assert parser.file_path == Path(temp_json_file)

    def test_parse_array(self, tmp_path, sample_json_data):
        """Test parsing a JSON file with an array of objects"""
        # Create a JSON file with an array of objects
        array_file = tmp_path / "array.json"
        with open(array_file, 'w') as f:
            json.dump(sample_json_data["products"], f)
        
        # Parse the file
        parser = JSONParser(array_file)
        result = parser.parse()
        
        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that the data matches the expected structure
        assert len(result) == 3
        assert list(result.columns) == ['sku', 'name', 'price', 'category', 'manufacturer']
        assert result.iloc[0]['sku'] == 'ABC123'
        assert result.iloc[1]['name'] == 'Wireless Mic'
        assert result.iloc[2]['price'] == 499.00

    def test_parse_object(self, tmp_path, sample_json_data):
        """Test parsing a JSON file with a single object"""
        # Create a JSON file with a single object
        object_file = tmp_path / "object.json"
        with open(object_file, 'w') as f:
            json.dump(sample_json_data["products"][0], f)
        
        # Parse the file
        parser = JSONParser(object_file)
        result = parser.parse()
        
        # Check that the result is a DataFrame with a single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ['sku', 'name', 'price', 'category', 'manufacturer']
        assert result.iloc[0]['sku'] == 'ABC123'
        assert result.iloc[0]['name'] == 'HD Camera'

    def test_parse_nested(self, tmp_path):
        """Test parsing a JSON file with nested objects"""
        # Create a JSON file with nested objects
        nested_data = {
            "store": {
                "name": "AV Equipment Store",
                "location": {
                    "city": "New York",
                    "state": "NY"
                },
                "inventory": [
                    {
                        "sku": "ABC123",
                        "details": {
                            "name": "HD Camera",
                            "specs": {
                                "resolution": "1080p",
                                "weight": "1.2kg"
                            }
                        },
                        "price": 299.99
                    },
                    {
                        "sku": "DEF456",
                        "details": {
                            "name": "Wireless Mic",
                            "specs": {
                                "frequency": "2.4GHz",
                                "battery": "8 hours"
                            }
                        },
                        "price": 149.50
                    }
                ]
            }
        }
        
        nested_file = tmp_path / "nested.json"
        with open(nested_file, 'w') as f:
            json.dump(nested_data, f)
        
        # Parse the file
        parser = JSONParser(nested_file)
        result = parser.parse()
        
        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that nested objects are flattened
        assert 'store_name' in result.columns
        assert 'store_location_city' in result.columns
        assert 'store_location_state' in result.columns

    def test_parse_empty(self, tmp_path):
        """Test parsing an empty JSON file"""
        # Create an empty JSON file
        empty_file = tmp_path / "empty.json"
        with open(empty_file, 'w') as f:
            f.write("{}")
        
        # Parse the file
        parser = JSONParser(empty_file)
        result = parser.parse()
        
        # Check that the result is an empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Empty object becomes a single row
        assert len(result.columns) == 0  # With no columns

    def test_error_handling(self, tmp_path):
        """Test handling of invalid JSON"""
        # Create an invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{invalid json")
        
        # Parse the file
        parser = JSONParser(invalid_file)
        result = parser.parse()
        
        # Check that the result is an empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
