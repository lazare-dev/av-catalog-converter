"""
Unit tests for the structure analyzer
"""
import pytest
import pandas as pd
import numpy as np

from services.structure.structure_analyzer import StructureAnalyzer


class TestStructureAnalyzer:
    """Test cases for StructureAnalyzer"""

    def test_init(self):
        """Test initialization"""
        analyzer = StructureAnalyzer()
        assert hasattr(analyzer, 'logger')

    def test_analyze(self, sample_csv_data):
        """Test analyzing a DataFrame"""
        # Create an analyzer
        analyzer = StructureAnalyzer()
        
        # Analyze the DataFrame
        result = analyzer.analyze(sample_csv_data)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the result contains the expected keys
        assert 'column_types' in result
        assert 'row_count' in result
        assert 'column_count' in result
        
        # Check specific values
        assert result['row_count'] == 3
        assert result['column_count'] == 5
        
        # Check column types
        column_types = result['column_types']
        assert 'SKU' in column_types
        assert 'Product Name' in column_types
        assert 'Price' in column_types
        assert 'Category' in column_types
        assert 'Manufacturer' in column_types
        
        # Check specific column types
        assert column_types['SKU']['type'] in ['id', 'string']
        assert column_types['Price']['type'] in ['price', 'decimal', 'numeric']

    def test_analyze_empty_dataframe(self):
        """Test analyzing an empty DataFrame"""
        # Create an analyzer
        analyzer = StructureAnalyzer()
        
        # Create an empty DataFrame
        df = pd.DataFrame()
        
        # Analyze the DataFrame
        result = analyzer.analyze(df)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the result contains the expected keys
        assert 'column_types' in result
        assert 'row_count' in result
        assert 'column_count' in result
        
        # Check specific values
        assert result['row_count'] == 0
        assert result['column_count'] == 0
        assert result['column_types'] == {}

    def test_analyze_with_missing_values(self):
        """Test analyzing a DataFrame with missing values"""
        # Create an analyzer
        analyzer = StructureAnalyzer()
        
        # Create a DataFrame with missing values
        df = pd.DataFrame({
            'SKU': ['ABC123', 'DEF456', None],
            'Product Name': ['HD Camera', None, 'Audio Mixer'],
            'Price': [299.99, 149.50, None],
            'Category': [None, 'Audio', 'Audio'],
            'Manufacturer': ['Sony', 'Shure', None]
        })
        
        # Analyze the DataFrame
        result = analyzer.analyze(df)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the result contains the expected keys
        assert 'column_types' in result
        assert 'row_count' in result
        assert 'column_count' in result
        assert 'missing_values' in result
        
        # Check specific values
        assert result['row_count'] == 3
        assert result['column_count'] == 5
        
        # Check missing values
        missing_values = result['missing_values']
        assert missing_values['SKU'] == 1
        assert missing_values['Product Name'] == 1
        assert missing_values['Price'] == 1
        assert missing_values['Category'] == 1
        assert missing_values['Manufacturer'] == 1

    def test_analyze_with_mixed_types(self):
        """Test analyzing a DataFrame with mixed types"""
        # Create an analyzer
        analyzer = StructureAnalyzer()
        
        # Create a DataFrame with mixed types
        df = pd.DataFrame({
            'Mixed': ['ABC123', 123, 456.78, None],
            'Numeric': [1, 2, 3, 4],
            'Text': ['a', 'b', 'c', 'd'],
            'Boolean': [True, False, True, False],
            'Date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'])
        })
        
        # Analyze the DataFrame
        result = analyzer.analyze(df)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check column types
        column_types = result['column_types']
        assert 'Mixed' in column_types
        assert 'Numeric' in column_types
        assert 'Text' in column_types
        assert 'Boolean' in column_types
        assert 'Date' in column_types
        
        # Check specific column types
        assert column_types['Numeric']['type'] in ['integer', 'numeric']
        assert column_types['Text']['type'] in ['string', 'text']
        assert column_types['Boolean']['type'] in ['boolean', 'binary']
        assert column_types['Date']['type'] in ['date', 'datetime']

    def test_detect_column_types(self):
        """Test detecting column types"""
        # Create an analyzer
        analyzer = StructureAnalyzer()
        
        # Create a DataFrame with various column types
        df = pd.DataFrame({
            'ID': ['ABC123', 'DEF456', 'GHI789'],
            'Name': ['HD Camera', 'Wireless Mic', 'Audio Mixer'],
            'Price': [299.99, 149.50, 499.00],
            'Category': ['Video', 'Audio', 'Audio'],
            'Manufacturer': ['Sony', 'Shure', 'Yamaha'],
            'In Stock': [True, False, True],
            'Date Added': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })
        
        # Detect column types
        result = analyzer._detect_column_types(df)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check specific column types
        assert result['ID']['type'] in ['id', 'string']
        assert result['Name']['type'] in ['string', 'text']
        assert result['Price']['type'] in ['price', 'decimal', 'numeric']
        assert result['Category']['type'] in ['category', 'string']
        assert result['Manufacturer']['type'] in ['string', 'text']
        assert result['In Stock']['type'] in ['boolean', 'binary']
        assert result['Date Added']['type'] in ['date', 'datetime']
