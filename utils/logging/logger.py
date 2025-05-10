"""
Logging utility for consistent logging across all modules
"""
import logging
import inspect
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])

class Logger:
    """
    Logger utility class for consistent logging across all modules
    
    Usage:
        # Create a logger for a module
        logger = Logger.get_logger(__name__)
        
        # Log messages
        logger.info("This is an info message")
        logger.error("This is an error message", extra={"key": "value"})
        
        # Log with context
        with logger.context(operation="process_file", file_path="example.csv"):
            logger.info("Processing file")
            
        # Decorate functions for automatic entry/exit logging
        @logger.logged
        def my_function(arg1, arg2):
            # Function code
            return result
    """
    
    @staticmethod
    def get_logger(name: str) -> 'Logger':
        """
        Get a logger instance for the specified name
        
        Args:
            name: The name of the logger (typically __name__)
            
        Returns:
            Logger: A Logger instance
        """
        return Logger(name)
    
    def __init__(self, name: str):
        """
        Initialize the logger
        
        Args:
            name: The name of the logger
        """
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log a debug message
        
        Args:
            message: The message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log an info message
        
        Args:
            message: The message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log a warning message
        
        Args:
            message: The message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Log an error message
        
        Args:
            message: The message to log
            exc_info: Whether to include exception info
            **kwargs: Additional context to include in the log
        """
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Log a critical message
        
        Args:
            message: The message to log
            exc_info: Whether to include exception info
            **kwargs: Additional context to include in the log
        """
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def _log(self, level: int, message: str, exc_info: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Internal method to log a message
        
        Args:
            level: The log level
            message: The message to log
            exc_info: Whether to include exception info
            **kwargs: Additional context to include in the log
        """
        # Merge the current context with the provided kwargs
        extra = {**self._context, **kwargs}
        
        # Get the caller information
        frame = inspect.currentframe()
        if frame:
            caller_frame = frame.f_back
            if caller_frame:
                extra['file'] = caller_frame.f_code.co_filename
                extra['line'] = caller_frame.f_lineno
                extra['function'] = caller_frame.f_code.co_name
        
        # Log the message
        self._logger.log(level, message, exc_info=exc_info, extra={'extra': extra})
    
    def context(self, **kwargs: Any) -> 'LoggerContext':
        """
        Create a context manager for adding context to logs
        
        Args:
            **kwargs: Context key-value pairs
            
        Returns:
            LoggerContext: A context manager
        """
        return LoggerContext(self, kwargs)
    
    def logged(self, func: F) -> F:
        """
        Decorator to log function entry and exit
        
        Args:
            func: The function to decorate
            
        Returns:
            The decorated function
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log function entry
            self.debug(f"Entering {func.__name__}")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log function exit
                self.debug(f"Exiting {func.__name__}")
                
                return result
            except Exception as e:
                # Log exception
                self.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                raise
        
        return cast(F, wrapper)


class LoggerContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, logger: Logger, context: Dict[str, Any]):
        """
        Initialize the context manager
        
        Args:
            logger: The logger instance
            context: The context to add
        """
        self.logger = logger
        self.context = context
        self.previous_context: Dict[str, Any] = {}
    
    def __enter__(self) -> 'LoggerContext':
        """
        Enter the context
        
        Returns:
            LoggerContext: The context manager
        """
        # Save the previous context
        self.previous_context = self.logger._context.copy()
        
        # Update the logger's context
        self.logger._context.update(self.context)
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        # Restore the previous context
        self.logger._context = self.previous_context
