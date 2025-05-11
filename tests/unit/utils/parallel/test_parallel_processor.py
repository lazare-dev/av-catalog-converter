"""
Unit tests for the parallel processor utility
"""
import pytest
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from utils.parallel.parallel_processor import ParallelProcessor

# Define process functions at module level so they can be pickled
def simple_process_func(chunk):
    """Simple processing function that doubles the values"""
    return chunk * 2

def sleep_process_func(chunk):
    """Processing function with a sleep to simulate work"""
    time.sleep(0.01)
    return chunk * 2

def dict_process_func(chunk):
    """Processing function that returns a dictionary of stats"""
    return {'sum': chunk.sum().sum(), 'mean': chunk.mean().mean()}

def error_process_func(chunk):
    """Processing function that raises an error for specific values"""
    if 50 in chunk['A'].values:
        raise ValueError("Test error")
    return chunk

def merge_results_func(results):
    """Merge function for dictionary results"""
    return {
        'sum': sum(r['sum'] for r in results),
        'mean': sum(r['mean'] for r in results) / len(results)
    }


class TestParallelProcessor:
    """Tests for the ParallelProcessor class"""

    def test_init(self):
        """Test initialization with default parameters"""
        processor = ParallelProcessor()
        assert processor.max_workers > 0
        assert processor.use_threads  # Default is now True

        # Test with custom parameters
        processor = ParallelProcessor(max_workers=2, use_threads=False)
        assert processor.max_workers == 2
        assert not processor.use_threads

    def test_process_dataframe_small(self):
        """Test processing a small DataFrame (should skip parallelization)"""
        # Create a small DataFrame
        df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20)
        })

        # Process the DataFrame
        processor = ParallelProcessor()
        result = processor.process_dataframe(df, simple_process_func, chunk_size=20)

        # Check the result
        expected = df * 2
        pd.testing.assert_frame_equal(result, expected)

    def test_process_dataframe_large(self):
        """Test processing a large DataFrame with parallelization"""
        # Create a large DataFrame
        df = pd.DataFrame({
            'A': range(1000),
            'B': range(1000, 2000)
        })

        # Process the DataFrame
        processor = ParallelProcessor(max_workers=4)
        result = processor.process_dataframe(df, sleep_process_func, chunk_size=100)

        # Check the result
        expected = df * 2
        pd.testing.assert_frame_equal(result, expected)

    def test_process_dataframe_with_custom_merge(self):
        """Test processing with a custom merge function"""
        # Create a DataFrame
        df = pd.DataFrame({
            'A': range(500),
            'B': range(500, 1000)
        })

        # Process the DataFrame
        processor = ParallelProcessor(max_workers=2)
        result = processor.process_dataframe(
            df, dict_process_func, chunk_size=100, merge_func=merge_results_func
        )

        # Check the result
        assert 'sum' in result
        assert 'mean' in result
        assert result['sum'] == df.sum().sum()
        assert abs(result['mean'] - df.mean().mean()) < 0.001

    def test_process_file_csv(self, tmp_path):
        """Test processing a CSV file"""
        # Create a test CSV file
        file_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'A': range(500),
            'B': range(500, 1000)
        })
        df.to_csv(file_path, index=False)

        # Process the file
        processor = ParallelProcessor(max_workers=2)
        result = processor.process_file(
            file_path, simple_process_func, chunk_size=100
        )

        # Check the result
        expected = df * 2
        pd.testing.assert_frame_equal(result, expected)

    def test_executor_type(self):
        """Test the _executor_type method"""
        # Default should be threads
        processor = ParallelProcessor()
        assert processor._executor_type() == "threads"

        # Test with processes
        processor = ParallelProcessor(use_threads=False)
        assert processor._executor_type() == "processes"

        # Test with threads
        processor = ParallelProcessor(use_threads=True)
        assert processor._executor_type() == "threads"

    def test_process_chunks(self):
        """Test the _process_chunks method directly"""
        # Create chunks
        chunks = [
            pd.DataFrame({'A': range(i, i+10)})
            for i in range(0, 50, 10)
        ]

        # Process chunks with threads
        processor = ParallelProcessor(max_workers=2, use_threads=True)
        results = processor._process_chunks(chunks, simple_process_func)

        # Check the results
        assert len(results) == len(chunks)
        for i, result in enumerate(results):
            expected = chunks[i] * 2
            pd.testing.assert_frame_equal(result, expected)

        # Process chunks with processes
        processor = ParallelProcessor(max_workers=2, use_threads=False)
        results = processor._process_chunks(chunks, simple_process_func)

        # Check the results
        assert len(results) == len(chunks)
        for i, result in enumerate(results):
            expected = chunks[i] * 2
            pd.testing.assert_frame_equal(result, expected)

    def test_error_handling(self):
        """Test error handling during parallel processing"""
        # Create a DataFrame
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })

        # Process the DataFrame and check that the error is propagated
        processor = ParallelProcessor(max_workers=2)
        with pytest.raises(ValueError, match="Test error"):
            processor.process_dataframe(df, error_process_func, chunk_size=20)
