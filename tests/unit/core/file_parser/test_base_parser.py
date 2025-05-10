"""
Unit tests for the base parser
"""
import pytest
import pandas as pd
from pathlib import Path
import os

from core.file_parser.base_parser import BaseParser


class TestBaseParser:
    """Test cases for BaseParser"""

    def test_init(self, tmp_path):
        """Test initialization"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Create a parser
        parser = BaseParser(test_file)
        
        # Check that the file path is set correctly
        assert parser.file_path == Path(test_file)
        assert parser.encoding is None
        assert parser.detected_headers is None
        assert parser.data_boundaries is None

    def test_parse_not_implemented(self, tmp_path):
        """Test that parse() raises NotImplementedError"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Create a parser
        parser = BaseParser(test_file)
        
        # Check that parse() raises NotImplementedError
        with pytest.raises(NotImplementedError):
            parser.parse()

    def test_get_sample(self, tmp_path, monkeypatch):
        """Test get_sample()"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Create a parser
        parser = BaseParser(test_file)
        
        # Mock the parse() method to return a DataFrame
        sample_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        monkeypatch.setattr(parser, 'parse', lambda: sample_data)
        
        # Get a sample
        sample = parser.get_sample(3)
        
        # Check that the sample is a DataFrame with 3 rows
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) == 3
        assert list(sample.columns) == ['A', 'B']
        assert list(sample['A']) == [1, 2, 3]
        assert list(sample['B']) == ['a', 'b', 'c']

    def test_clean_column_names(self, tmp_path):
        """Test clean_column_names()"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Create a parser
        parser = BaseParser(test_file)
        
        # Create a DataFrame with messy column names
        df = pd.DataFrame({
            'Product ID': [1, 2, 3],
            'Product Name!': ['a', 'b', 'c'],
            'Price ($)': [10, 20, 30],
            '': [1, 2, 3],  # Empty column name
            'Duplicate': [1, 2, 3],
            'Duplicate': [4, 5, 6]  # Duplicate column name
        })
        
        # Clean the column names
        result = parser.clean_column_names(df)
        
        # Check that the column names are cleaned
        assert 'Product ID' in result.columns
        assert 'Product Name' in result.columns
        assert 'Price' in result.columns or 'Price $' in result.columns
        assert 'Column_4' in result.columns or 'Unnamed' in result.columns[3]
        assert 'Duplicate' in result.columns
        assert 'Duplicate_1' in result.columns

    def test_preprocess_dataframe(self, tmp_path):
        """Test preprocess_dataframe()"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Create a parser
        parser = BaseParser(test_file)
        
        # Create a DataFrame with various issues
        df = pd.DataFrame({
            'A': [1, 2, None, 'none', 'NULL'],
            'B': ['a', 'b', 'N/A', 'NA', '#N/A'],
            'C': [None, None, None, None, None],  # Empty column
            'D': ['1', '2', '3', '4', '5']  # Numeric strings
        })
        
        # Preprocess the DataFrame
        result = parser.preprocess_dataframe(df)
        
        # Check that the DataFrame is preprocessed
        assert pd.isna(result.iloc[2]['A'])
        assert pd.isna(result.iloc[3]['A'])
        assert pd.isna(result.iloc[4]['A'])
        
        assert pd.isna(result.iloc[2]['B'])
        assert pd.isna(result.iloc[3]['B'])
        assert pd.isna(result.iloc[4]['B'])
        
        # Check that empty columns are removed
        assert 'C' not in result.columns
        
        # Check that numeric strings are converted to numbers
        assert result['D'].dtype in [int, float, 'int64', 'float64']
