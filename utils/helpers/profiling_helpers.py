# utils/helpers/profiling_helpers.py
"""
Performance profiling utilities
"""
import logging
import time
import functools
from typing import Callable, Any, Dict
import tracemalloc
from contextlib import contextmanager

def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    
    Args:
        func (Callable): Function to measure
        
    Returns:
        Callable: Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper

def memory_profile(func: Callable) -> Callable:
    """
    Decorator to measure memory usage
    
    Args:
        func (Callable): Function to measure
        
    Returns:
        Callable: Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        
        result = func(*args, **kwargs)
        
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Compare snapshots
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        # Get top 5 memory differences
        top_stats = stats[:5]
        
        logger.debug(f"Memory profile for '{func.__name__}':")
        for stat in top_stats:
            logger.debug(f"  {stat}")
        
        return result
    
    return wrapper

@contextmanager
def profile_block(name: str = "Code Block"):
    """
    Context manager for profiling code blocks
    
    Args:
        name (str): Name of the code block
    """
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    
    try:
        yield
    finally:
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"Block '{name}' executed in {execution_time:.4f} seconds")
        
        # Compare snapshots
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        # Get top 3 memory differences
        top_stats = stats[:3]
        
        if top_stats:
            logger.debug(f"Memory usage for block '{name}':")
            for stat in top_stats:
                logger.debug(f"  {stat}")

def get_performance_stats() -> Dict[str, Any]:
    """
    Get current performance statistics
    
    Returns:
        Dict[str, Any]: Performance statistics
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    stats = {
        "memory_usage": {
            "rss": process.memory_info().rss / (1024 * 1024),  # MB
            "vms": process.memory_info().vms / (1024 * 1024),  # MB
        },
        "cpu_usage": process.cpu_percent(interval=0.1),
        "thread_count": process.num_threads(),
        "open_files": len(process.open_files()),
    }
    
    return stats