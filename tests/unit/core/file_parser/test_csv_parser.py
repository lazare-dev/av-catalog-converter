"""
Unit tests for the CSV parser
"""
import pytest
import pandas as pd
import os
from pathlib import Path

from core.file_parser.csv_parser import CSVParser


class TestCSVParser:
    """Test cases for CSVParser"""

    def test_init(self, temp_csv_file):
        """Test initialization"""
        parser = CSVParser(temp_csv_file)
        assert parser.file_path == Path(temp_csv_file)
        assert parser.delimiter is None
        assert parser.has_header is True

    def test_parse(self, temp_csv_file, sample_csv_data):
        """Test parsing a CSV file"""
        parser = CSVParser(temp_csv_file)
        result = parser.parse()
        
        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that the data matches the sample data
        pd.testing.assert_frame_equal(result, sample_csv_data)

    def test_detect_delimiter(self, tmp_path):
        """Test delimiter detection"""
        # Create CSV files with different delimiters
        comma_file = tmp_path / "comma.csv"
        with open(comma_file, 'w') as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6")
        
        tab_file = tmp_path / "tab.csv"
        with open(tab_file, 'w') as f:
            f.write("col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6")
        
        semicolon_file = tmp_path / "semicolon.csv"
        with open(semicolon_file, 'w') as f:
            f.write("col1;col2;col3\n1;2;3\n4;5;6")
        
        # Test comma delimiter detection
        comma_parser = CSVParser(comma_file)
        assert comma_parser.detect_delimiter() == ','
        
        # Test tab delimiter detection
        tab_parser = CSVParser(tab_file)
        assert tab_parser.detect_delimiter() == '\t'
        
        # Test semicolon delimiter detection
        semicolon_parser = CSVParser(semicolon_file)
        assert semicolon_parser.detect_delimiter() == ';'

    def test_detect_header(self, tmp_path):
        """Test header detection"""
        # Create CSV file with header
        with_header = tmp_path / "with_header.csv"
        with open(with_header, 'w') as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6")
        
        # Create CSV file without header
        without_header = tmp_path / "without_header.csv"
        with open(without_header, 'w') as f:
            f.write("1,2,3\n4,5,6\n7,8,9")
        
        # Test header detection
        with_header_parser = CSVParser(with_header)
        assert with_header_parser.detect_header() is True
        
        without_header_parser = CSVParser(without_header)
        assert without_header_parser.detect_header() is False

    def test_get_sample(self, temp_csv_file, sample_csv_data):
        """Test getting a sample of the data"""
        parser = CSVParser(temp_csv_file)
        sample = parser.get_sample(2)
        
        # Check that the sample has the right number of rows
        assert len(sample) == 2
        
        # Check that the sample data matches the first 2 rows of the sample data
        pd.testing.assert_frame_equal(sample, sample_csv_data.head(2))

    def test_parse_with_custom_delimiter(self, tmp_path):
        """Test parsing with a custom delimiter"""
        # Create a pipe-delimited file
        pipe_file = tmp_path / "pipe.csv"
        with open(pipe_file, 'w') as f:
            f.write("col1|col2|col3\n1|2|3\n4|5|6")
        
        # Parse with custom delimiter
        parser = CSVParser(pipe_file)
        parser.delimiter = '|'
        result = parser.parse()
        
        # Check the result
        assert list(result.columns) == ['col1', 'col2', 'col3']
        assert result.iloc[0, 0] == '1'
        assert result.iloc[0, 1] == '2'
        assert result.iloc[0, 2] == '3'
