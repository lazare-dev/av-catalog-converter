"""
Unit tests for the Excel parser with parallel processing
"""
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

from core.file_parser.excel_parser import ExcelParser
from utils.parallel.parallel_processor import ParallelProcessor


class TestExcelParserParallel:
    """Tests for the ExcelParser class with parallel processing"""

    def test_parse_large_file_with_parallel_processing(self, tmp_path):
        """Test parsing a large Excel file with parallel processing"""
        # Create a large Excel file
        file_path = tmp_path / "large.xlsx"
        
        # Create a DataFrame with 1,000 rows (small for testing but we'll mock the size)
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=1000),
            'B': np.random.randint(0, 100, size=1000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=1000),
            'D': np.random.random(size=1000)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Make the file appear large to trigger parallel processing
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 30MB (above the 20MB threshold)
            mock_getsize.return_value = 30 * 1024 * 1024
            
            # Create parser
            parser = ExcelParser(file_path)
            
            # Mock the sheet selection
            parser.sheet_name = "Sheet1"
            
            # Mock the ParallelProcessor
            with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                # Setup the mock
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                
                # Mock the _process_chunks method to return a list of DataFrames
                mock_processor._process_chunks.return_value = [df]
                
                # Parse the file
                result = parser.parse()
                
                # Check that ParallelProcessor was used
                mock_processor_class.assert_called_once_with(use_threads=True)
                
                # Check that the result is a DataFrame
                assert isinstance(result, pd.DataFrame)

    def test_parse_very_large_file_with_sheet_partitioning(self, tmp_path):
        """Test parsing a very large Excel file with sheet partitioning"""
        # Create a large Excel file
        file_path = tmp_path / "very_large.xlsx"
        
        # Create a DataFrame with 1,000 rows
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=1000),
            'B': np.random.randint(0, 100, size=1000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=1000),
            'D': np.random.random(size=1000)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Make the file appear very large to trigger sheet partitioning
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 150MB (above the 100MB threshold)
            mock_getsize.return_value = 150 * 1024 * 1024
            
            # Create parser
            parser = ExcelParser(file_path)
            
            # Mock the sheet selection
            parser.sheet_name = "Sheet1"
            
            # Mock the _can_partition_sheet method to return True
            with patch.object(parser, '_can_partition_sheet', return_value=True):
                # Mock the _estimate_row_count method to return a large number
                with patch.object(parser, '_estimate_row_count', return_value=100000):
                    # Mock the ParallelProcessor
                    with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                        # Setup the mock
                        mock_processor = MagicMock()
                        mock_processor_class.return_value = mock_processor
                        
                        # Mock the _process_chunks method to return a list of DataFrames
                        first_part = df.iloc[:500].copy()
                        second_part = df.iloc[500:].copy()
                        mock_processor._process_chunks.return_value = [first_part, second_part]
                        
                        # Parse the file
                        result = parser.parse()
                        
                        # Check that ParallelProcessor was used
                        mock_processor_class.assert_called_once_with(use_threads=True)
                        
                        # Check that the result is a DataFrame with the expected number of rows
                        assert isinstance(result, pd.DataFrame)
                        assert len(result) == len(df)

    def test_parse_small_file_without_parallel_processing(self, tmp_path):
        """Test parsing a small Excel file without parallel processing"""
        # Create a small Excel file
        file_path = tmp_path / "small.xlsx"
        
        # Create a DataFrame with 100 rows
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=100),
            'B': np.random.randint(0, 100, size=100),
            'C': np.random.choice(['X', 'Y', 'Z'], size=100),
            'D': np.random.random(size=100)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Make the file appear small to avoid parallel processing
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 10MB (below the 20MB threshold)
            mock_getsize.return_value = 10 * 1024 * 1024
            
            # Create parser
            parser = ExcelParser(file_path)
            
            # Mock the sheet selection
            parser.sheet_name = "Sheet1"
            
            # Mock the ParallelProcessor to ensure it's not used
            with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                # Parse the file
                result = parser.parse()
                
                # Check that ParallelProcessor was not used
                mock_processor_class.assert_not_called()
                
                # Check that the result has the same structure
                assert list(result.columns) == list(df.columns)
                assert len(result) == len(df)

    def test_can_partition_sheet(self, tmp_path):
        """Test the _can_partition_sheet method"""
        # Create an Excel file
        file_path = tmp_path / "test.xlsx"
        
        # Create a DataFrame
        df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Create parser
        parser = ExcelParser(file_path)
        parser.sheet_name = "Sheet1"
        
        # Test the method
        result = parser._can_partition_sheet()
        
        # Should be able to partition a simple sheet
        assert result is True

    def test_estimate_row_count(self, tmp_path):
        """Test the _estimate_row_count method"""
        # Create an Excel file
        file_path = tmp_path / "test.xlsx"
        
        # Create a DataFrame with 100 rows
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Create parser
        parser = ExcelParser(file_path)
        parser.sheet_name = "Sheet1"
        
        # Mock openpyxl to return a specific row count
        with patch('openpyxl.load_workbook') as mock_load_workbook:
            mock_wb = MagicMock()
            mock_load_workbook.return_value = mock_wb
            
            mock_ws = MagicMock()
            mock_wb.__getitem__.return_value = mock_ws
            mock_wb.sheetnames = ["Sheet1"]
            
            # Set max_row to 100
            mock_ws.max_row = 100
            
            # Test the method
            result = parser._estimate_row_count()
            
            # Should return the mocked row count
            assert result == 100

    def test_error_handling_during_parallel_processing(self, tmp_path):
        """Test error handling during parallel processing"""
        # Create an Excel file
        file_path = tmp_path / "error.xlsx"
        
        # Create a DataFrame
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Make the file appear large
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 150 * 1024 * 1024
            
            # Create parser
            parser = ExcelParser(file_path)
            parser.sheet_name = "Sheet1"
            
            # Mock _can_partition_sheet to return True
            with patch.object(parser, '_can_partition_sheet', return_value=True):
                # Mock _estimate_row_count to return a large number
                with patch.object(parser, '_estimate_row_count', return_value=100000):
                    # Mock ParallelProcessor to raise an exception
                    with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                        mock_processor = MagicMock()
                        mock_processor_class.return_value = mock_processor
                        mock_processor._process_chunks.side_effect = Exception("Test error")
                        
                        # Parse the file - should fall back to standard parsing
                        result = parser.parse()
                        
                        # Check that the result is still a DataFrame
                        assert isinstance(result, pd.DataFrame)
                        assert list(result.columns) == list(df.columns)
                        assert len(result) == len(df)
