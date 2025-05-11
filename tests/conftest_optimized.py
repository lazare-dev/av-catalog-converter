"""
Test fixtures for optimized components
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixtures for parallel processing tests
@pytest.fixture
def small_csv_file():
    """Return the path to a small CSV file"""
    file_path = FIXTURES_DIR / "small_catalog.csv"
    if not file_path.exists():
        # Create a small CSV file if it doesn't exist
        df = pd.DataFrame({
            'SKU': [f"SKU{i}" for i in range(10)],
            'Product Name': [f"Product {i}" for i in range(10)],
            'Price': np.random.random(size=10) * 1000,
            'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=10),
            'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=10)
        })
        df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def medium_csv_file():
    """Return the path to a medium-sized CSV file"""
    file_path = FIXTURES_DIR / "medium_catalog.csv"
    if not file_path.exists():
        # Create a medium CSV file if it doesn't exist
        df = pd.DataFrame({
            'SKU': [f"SKU{i}" for i in range(1000)],
            'Product Name': [f"Product {i}" for i in range(1000)],
            'Price': np.random.random(size=1000) * 1000,
            'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=1000),
            'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=1000)
        })
        df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def small_excel_file():
    """Return the path to a small Excel file"""
    file_path = FIXTURES_DIR / "small_catalog.xlsx"
    if not file_path.exists():
        # Create a small Excel file if it doesn't exist
        csv_path = FIXTURES_DIR / "small_catalog.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_excel(file_path, index=False)
        else:
            df = pd.DataFrame({
                'SKU': [f"SKU{i}" for i in range(10)],
                'Product Name': [f"Product {i}" for i in range(10)],
                'Price': np.random.random(size=10) * 1000,
                'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=10),
                'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=10)
            })
            df.to_excel(file_path, index=False)
    return file_path

@pytest.fixture
def medium_excel_file():
    """Return the path to a medium-sized Excel file"""
    file_path = FIXTURES_DIR / "medium_catalog.xlsx"
    if not file_path.exists():
        # Create a medium Excel file if it doesn't exist
        csv_path = FIXTURES_DIR / "medium_catalog.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_excel(file_path, index=False)
        else:
            df = pd.DataFrame({
                'SKU': [f"SKU{i}" for i in range(1000)],
                'Product Name': [f"Product {i}" for i in range(1000)],
                'Price': np.random.random(size=1000) * 1000,
                'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=1000),
                'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=1000)
            })
            df.to_excel(file_path, index=False)
    return file_path

# Fixtures for rate limiting tests
@pytest.fixture
def mock_rate_limiter():
    """Return a mock rate limiter"""
    mock_limiter = MagicMock()
    mock_limiter.bucket = MagicMock()
    mock_limiter.bucket.consume.return_value = True
    mock_limiter.get_stats.return_value = {
        'total_requests': 10,
        'limited_requests': 2,
        'wait_time': 1.5,
        'tokens_per_second': 1.0,
        'available_tokens': 8.0,
        'max_tokens': 10,
        'utilization': 0.2
    }
    return mock_limiter

# Fixtures for caching tests
@pytest.fixture
def mock_adaptive_cache():
    """Return a mock adaptive cache"""
    mock_cache = MagicMock()
    mock_cache.get.side_effect = lambda k: f"Cached value for {k}" if k in ["key1", "key2"] else None
    mock_cache.get_stats.return_value = {
        'size': 2,
        'max_size': 1000,
        'base_ttl': 3600,
        'avg_ttl': 5400,
        'hits': 10,
        'misses': 5,
        'hit_ratio': 0.67,
        'evictions': 1,
        'expirations': 2
    }
    return mock_cache

# Fixtures for LLM tests
@pytest.fixture
def mock_phi_client():
    """Return a mock Phi client"""
    mock_client = MagicMock()
    mock_client.generate_response.side_effect = lambda p: f"Response to {p}"
    mock_client.get_stats.return_value = {
        'total_generations': 10,
        'total_tokens_generated': 500,
        'average_generation_time': 0.5,
        'cache_hits': 5,
        'model_id': 'microsoft/phi-2',
        'is_initialized': True,
        'rate_limited_count': 2,
        'rate_limited_wait_time': 1.5
    }
    return mock_client

# Fixtures for parallel processing tests
@pytest.fixture
def mock_parallel_processor():
    """Return a mock parallel processor"""
    mock_processor = MagicMock()
    mock_processor.process_dataframe.side_effect = lambda df, *args, **kwargs: df
    mock_processor.process_file.side_effect = lambda file_path, *args, **kwargs: pd.read_csv(file_path) if str(file_path).endswith('.csv') else pd.read_excel(file_path)
    return mock_processor
