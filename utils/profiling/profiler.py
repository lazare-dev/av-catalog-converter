"""
Performance profiling utilities
"""
import time
import cProfile
import pstats
import io
import functools
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from utils.logging.logger import Logger

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])

class Profiler:
    """
    Utility for profiling code execution
    
    This class provides tools for measuring execution time and profiling code.
    It can be used as a decorator, context manager, or directly.
    
    Usage:
        # As a decorator
        @Profiler.profile
        def my_function():
            # Function code
            
        # As a context manager
        with Profiler() as profiler:
            # Code to profile
            
        # Direct usage
        profiler = Profiler()
        profiler.start()
        # Code to profile
        profiler.stop()
        profiler.print_stats()
    """
    
    def __init__(self, name: Optional[str] = None, enabled: bool = True):
        """
        Initialize the profiler
        
        Args:
            name: Name for this profiling session
            enabled: Whether profiling is enabled
        """
        self.name = name
        self.enabled = enabled
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.profiler: Optional[cProfile.Profile] = None
        self.logger = Logger.get_logger(__name__)
    
    def __enter__(self) -> 'Profiler':
        """
        Start profiling when entering a context
        
        Returns:
            The profiler instance
        """
        self.start()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Stop profiling when exiting a context
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.stop()
        if exc_type is None:
            self.print_stats()
    
    def start(self) -> None:
        """Start profiling"""
        if not self.enabled:
            return
            
        self.start_time = time.time()
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        if self.name:
            self.logger.debug(f"Started profiling: {self.name}")
    
    def stop(self) -> None:
        """Stop profiling"""
        if not self.enabled or self.profiler is None or self.start_time is None:
            return
            
        self.profiler.disable()
        self.end_time = time.time()
        
        if self.name:
            elapsed = self.end_time - self.start_time
            self.logger.debug(f"Stopped profiling: {self.name} (elapsed: {elapsed:.4f}s)")
    
    def print_stats(self, sort_by: str = 'cumulative', limit: int = 20) -> None:
        """
        Print profiling statistics
        
        Args:
            sort_by: Field to sort by ('cumulative', 'time', 'calls', etc.)
            limit: Maximum number of functions to show
        """
        if not self.enabled or self.profiler is None:
            return
            
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(limit)
        
        if self.name and self.start_time is not None and self.end_time is not None:
            elapsed = self.end_time - self.start_time
            header = f"Profiling results for {self.name} (elapsed: {elapsed:.4f}s)"
            print(f"\n{header}")
            print("=" * len(header))
        
        print(s.getvalue())
    
    def save_stats(self, filename: Optional[str] = None) -> str:
        """
        Save profiling statistics to a file
        
        Args:
            filename: Output filename (default: profile_{name}_{timestamp}.prof)
            
        Returns:
            Path to the saved file
        """
        if not self.enabled or self.profiler is None:
            return ""
            
        if filename is None:
            timestamp = int(time.time())
            name = self.name or "profile"
            filename = f"profile_{name}_{timestamp}.prof"
        
        # Create directory if it doesn't exist
        output_dir = Path("profiling")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        self.profiler.dump_stats(str(output_path))
        
        self.logger.info(f"Saved profiling stats to {output_path}")
        return str(output_path)
    
    @staticmethod
    def profile(func: F) -> F:
        """
        Decorator to profile a function
        
        Args:
            func: The function to profile
            
        Returns:
            The decorated function
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Profiler(name=func.__name__):
                return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    @staticmethod
    def time_function(func: F) -> F:
        """
        Decorator to time a function's execution
        
        Args:
            func: The function to time
            
        Returns:
            The decorated function
        """
        logger = Logger.get_logger(__name__)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            logger.info(f"Function {func.__name__} took {elapsed:.4f} seconds to execute")
            return result
        
        return cast(F, wrapper)


def profile_memory(func: F) -> F:
    """
    Decorator to profile memory usage of a function
    
    Args:
        func: The function to profile
        
    Returns:
        The decorated function
    """
    logger = Logger.get_logger(__name__)
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            logger.info(f"Function {func.__name__} memory usage: {mem_diff:.2f} MB (before: {mem_before:.2f} MB, after: {mem_after:.2f} MB)")
            
            return result
            
        except ImportError:
            logger.warning("psutil not installed, memory profiling disabled")
            return func(*args, **kwargs)
    
    return cast(F, wrapper)
