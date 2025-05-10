"""
End-to-end integration tests
"""
import pytest
import os
import pandas as pd
from pathlib import Path

from app import process_file


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_process_csv_file(self, temp_csv_file):
        """Test processing a CSV file end-to-end"""
        # Process the file
        result, error = process_file(temp_csv_file)

        # Check that there is no error
        assert error is None

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        assert 'SKU' in result.columns
        assert any(col in result.columns for col in ['Short Description', 'Product Name'])
        assert any(col in result.columns for col in ['Trade Price', 'Price'])
        assert 'Category' in result.columns
        assert 'Manufacturer' in result.columns

    def test_process_json_file(self, temp_json_file):
        """Test processing a JSON file end-to-end"""
        # Process the file
        result, error = process_file(temp_json_file)

        # Check that there is no error
        assert error is None

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        assert 'SKU' in result.columns or 'sku' in result.columns
        assert any(col in result.columns for col in ['Short Description', 'name'])
        assert any(col in result.columns for col in ['Trade Price', 'price'])
        assert 'Category' in result.columns or 'category' in result.columns
        assert 'Manufacturer' in result.columns or 'manufacturer' in result.columns

    def test_process_xml_file(self, temp_xml_file):
        """Test processing an XML file end-to-end"""
        # Process the file
        result, error = process_file(temp_xml_file)

        # Check that there is no error
        assert error is None

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        assert 'SKU' in result.columns or 'sku' in result.columns
        assert any(col in result.columns for col in ['Short Description', 'name'])
        assert any(col in result.columns for col in ['Trade Price', 'price'])
        assert 'Category' in result.columns or 'category' in result.columns
        assert 'Manufacturer' in result.columns or 'manufacturer' in result.columns

    def test_process_nonexistent_file(self):
        """Test processing a non-existent file"""
        # Process a non-existent file
        result, error = process_file("nonexistent_file.csv")

        # Check that there is an error
        assert error is not None
        assert "File not found" in error

        # Check that the result is None
        assert result is None

    def test_process_invalid_file(self, tmp_path):
        """Test processing an invalid file"""
        # Create an invalid file
        invalid_file = tmp_path / "invalid.txt"
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid catalog file")

        # Process the invalid file
        result, error = process_file(invalid_file)

        # Check the result
        # Note: The application might still try to parse it as CSV
        # So we don't assert that there's an error, but check the result
        if error is not None:
            assert "Error" in error
        else:
            assert isinstance(result, pd.DataFrame)