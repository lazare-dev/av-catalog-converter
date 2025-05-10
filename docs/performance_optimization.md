# Performance Optimization Guide

This guide provides strategies and best practices for optimizing the performance of the AV Catalog Converter application.

## Profiling Tools

The application includes built-in profiling tools to help identify performance bottlenecks:

### Running Profiling

```bash
# Profile the entire application with a specific file
python profile_app.py --file path/to/catalog.csv --component all

# Profile only the parsing component
python profile_app.py --file path/to/catalog.csv --component parsing

# Profile only the field mapping component
python profile_app.py --file path/to/catalog.csv --component mapping
```

### Analyzing Profiling Results

```bash
# Analyze a specific profile file
python analyze_profile.py --profile profiling/end_to_end_catalog.csv.prof

# Analyze all profile files in a directory
python analyze_profile.py --dir profiling
```

## Common Performance Bottlenecks

### 1. File Parsing

File parsing is often the most time-consuming operation, especially for large files.

**Optimization Strategies:**
- Use chunked reading for large files (implemented in `CSVParser`)
- Implement parallel processing for multi-sheet Excel files
- Cache parsed results for frequently accessed files
- Use memory-mapped files for very large datasets

### 2. Field Mapping

Field mapping can be slow when dealing with many columns or complex mapping rules.

**Optimization Strategies:**
- Cache mapping results for similar file structures
- Implement a more efficient pattern matching algorithm
- Use a pre-computed lookup table for common field names
- Parallelize mapping of independent columns

### 3. Value Normalization

Normalizing values, especially text fields, can be computationally expensive.

**Optimization Strategies:**
- Cache normalized values
- Use vectorized operations instead of row-by-row processing
- Implement batch processing for large datasets
- Use compiled regular expressions

## Memory Optimization

### 1. Chunked Processing

For large files, process the data in chunks to reduce memory usage:

```python
# Example of chunked processing
def process_large_file(file_path):
    chunks = pd.read_csv(file_path, chunksize=10000)
    results = []
    
    for chunk in chunks:
        # Process the chunk
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    # Combine results
    return pd.concat(results)
```

### 2. Memory Profiling

Use the `profile_memory` decorator to track memory usage:

```python
from utils.profiling.profiler import profile_memory

@profile_memory
def my_memory_intensive_function():
    # Function code
```

### 3. Reducing DataFrame Memory Usage

```python
def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    # Convert object columns to categories when appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df
```

## Caching Strategies

### 1. Function Result Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param1, param2):
    # Expensive calculation
    return result
```

### 2. Disk Caching

```python
import os
import pickle
from pathlib import Path

def cached_operation(input_data, cache_key):
    """Perform operation with disk caching"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    # Check if cached result exists
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Perform operation
    result = perform_expensive_operation(input_data)
    
    # Cache the result
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result
```

## Parallel Processing

### 1. Using ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_in_parallel(items, process_func, max_workers=None):
    """Process items in parallel"""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, items))
    
    return results
```

### 2. Parallel File Processing

```python
def process_files_in_parallel(file_paths):
    """Process multiple files in parallel"""
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, path) for path in file_paths]
        results = [future.result() for future in futures]
    
    return results
```

## Database Integration

For very large datasets, consider using a database backend:

```python
import sqlite3
import pandas as pd

def process_with_database(large_df):
    """Process a large DataFrame using SQLite"""
    # Create a temporary database
    conn = sqlite3.connect(':memory:')
    
    # Write DataFrame to database
    large_df.to_sql('data', conn, index=False)
    
    # Perform operations using SQL
    result = pd.read_sql("""
        SELECT category, COUNT(*) as count
        FROM data
        GROUP BY category
        ORDER BY count DESC
    """, conn)
    
    conn.close()
    return result
```

## Monitoring Performance

Add performance monitoring to track long-term trends:

```python
def log_performance_metrics(operation, start_time, end_time, data_size):
    """Log performance metrics for monitoring"""
    elapsed = end_time - start_time
    throughput = data_size / elapsed if elapsed > 0 else 0
    
    logger.info(f"Performance metrics for {operation}",
               elapsed_time=elapsed,
               data_size=data_size,
               throughput=throughput)
```

## Recommended Optimizations

Based on profiling of typical usage patterns, here are the recommended optimizations:

1. Implement chunked processing for CSV and Excel parsers
2. Add caching for field mapping results
3. Optimize regular expressions in value normalization
4. Implement parallel processing for batch file operations
5. Add memory optimization for large DataFrames

These optimizations should significantly improve performance for most use cases.
