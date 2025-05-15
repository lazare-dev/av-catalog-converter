"""
Unit tests for the CSV parser with parallel processing
"""
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

from core.file_parser.csv_parser import CSVParser
from utils.parallel.parallel_processor import ParallelProcessor


class TestCSVParserParallel:
    """Tests for the CSVParser class with parallel processing"""

    def test_parse_large_file_with_parallel_processing(self, tmp_path):
        """Test parsing a large CSV file with parallel processing"""
        # Create a large CSV file
        file_path = tmp_path / "large.csv"

        # Create a DataFrame with 10,000 rows
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=10000),
            'B': np.random.randint(0, 100, size=10000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=10000),
            'D': np.random.random(size=10000)
        })

        # Save to CSV
        df.to_csv(file_path, index=False)

        # Make the file appear large to trigger parallel processing
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 60MB (above the 50MB threshold)
            mock_getsize.return_value = 60 * 1024 * 1024

            # Create parser
            parser = CSVParser(file_path)

            # Mock the ParallelProcessor
            with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                # Setup the mock
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                mock_processor.process_file.return_value = df

                # Parse the file
                result = parser.parse()

                # Check that ParallelProcessor was used
                mock_processor_class.assert_called_once_with(use_threads=True)
                mock_processor.process_file.assert_called_once()

                # Check that the result is correct
                pd.testing.assert_frame_equal(result, df)

    def test_parse_very_large_file_with_smaller_chunks(self, tmp_path):
        """Test parsing a very large CSV file with smaller chunks"""
        # Create a large CSV file
        file_path = tmp_path / "very_large.csv"

        # Create a DataFrame with 1,000 rows (small for testing)
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=1000),
            'B': np.random.randint(0, 100, size=1000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=1000),
            'D': np.random.random(size=1000)
        })

        # Save to CSV
        df.to_csv(file_path, index=False)

        # Make the file appear very large to trigger smaller chunks
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 600MB (above the 500MB threshold)
            mock_getsize.return_value = 600 * 1024 * 1024

            # Create parser
            parser = CSVParser(file_path)

            # Mock the ParallelProcessor
            with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                # Setup the mock
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                mock_processor.process_file.return_value = df

                # Parse the file
                result = parser.parse()

                # Check that ParallelProcessor was used with smaller chunk size
                mock_processor_class.assert_called_once_with(use_threads=True)

                # Get the chunk_size parameter
                call_args = mock_processor.process_file.call_args[1]
                assert 'chunk_size' in call_args
                assert call_args['chunk_size'] == 50000  # Should use smaller chunks

                # Check that the result is correct
                pd.testing.assert_frame_equal(result, df)

    def test_parse_small_file_without_parallel_processing(self, tmp_path):
        """Test parsing a small CSV file without parallel processing"""
        # Create a small CSV file
        file_path = tmp_path / "small.csv"

        # Create a DataFrame with 100 rows
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=100),
            'B': np.random.randint(0, 100, size=100),
            'C': np.random.choice(['X', 'Y', 'Z'], size=100),
            'D': np.random.random(size=100)
        })

        # Save to CSV
        df.to_csv(file_path, index=False)

        # Make the file appear small to avoid parallel processing
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 10MB (below the 50MB threshold)
            mock_getsize.return_value = 10 * 1024 * 1024

            # Create parser
            parser = CSVParser(file_path)

            # Mock the ParallelProcessor to ensure it's not used
            with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                # Parse the file
                result = parser.parse()

                # Check that ParallelProcessor was not used
                mock_processor_class.assert_not_called()

                # Check that the result has the same structure
                assert list(result.columns) == list(df.columns)
                assert len(result) == len(df)

    def test_integration_with_real_parallel_processing(self, tmp_path):
        """Integration test with real parallel processing"""
        # Create a medium-sized CSV file
        file_path = tmp_path / "medium.csv"

        # Create a DataFrame with 5,000 rows
        df = pd.DataFrame({
            'A': np.random.randint(0, 100, size=5000),
            'B': np.random.randint(0, 100, size=5000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=5000),
            'D': np.random.random(size=5000)
        })

        # Save to CSV
        df.to_csv(file_path, index=False)

        # Make the file appear large enough for parallel processing
        with patch('os.path.getsize') as mock_getsize:
            # Set file size to 60MB
            mock_getsize.return_value = 60 * 1024 * 1024

            # Create parser
            parser = CSVParser(file_path)

            # Parse the file with real parallel processing
            result = parser.parse()

            # Check that the result is correct
            assert list(result.columns) == list(df.columns)
            assert len(result) == len(df)

            # Check some values to ensure data integrity
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric columns, check that values are approximately equal
                    result_values = sorted(result[col].iloc[:10].astype(float).tolist())
                    df_values = sorted(df[col].iloc[:10].astype(float).tolist())

                    # Check that the values are approximately equal
                    for rv, dv in zip(result_values, df_values):
                        assert abs(rv - dv) < 1e-10, f"Values differ: {rv} vs {dv}"
                else:
                    # For non-numeric columns, exact comparison is fine
                    assert set(result[col].iloc[:10]) == set(df[col].iloc[:10])

    def test_error_handling_during_parallel_processing(self, tmp_path):
        """Test error handling during parallel processing"""
        # Create a CSV file
        file_path = tmp_path / "error.csv"

        # Create a DataFrame
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })

        # Save to CSV
        df.to_csv(file_path, index=False)

        # Make the file appear large
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 60 * 1024 * 1024

            # Create parser
            parser = CSVParser(file_path)

            # Mock ParallelProcessor to raise an exception
            with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                mock_processor.process_file.side_effect = Exception("Test error")

                # Parse the file - should fall back to standard parsing
                result = parser.parse()

                # Check that the result is still a DataFrame
                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == list(df.columns)
                assert len(result) == len(df)
