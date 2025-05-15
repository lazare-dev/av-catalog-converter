"""
Logging utility for consistent logging across all modules
"""
import logging
import inspect
import functools
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Try to import psutil for enhanced process information
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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
        # Add context_data attribute for backward compatibility
        self.context_data = {}

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

    def _log(self, level: int, message: str, exc_info: Optional[bool] = None, stack_info: bool = False, **kwargs: Any) -> None:
        """
        Internal method to log a message with enhanced context

        Args:
            level: The log level
            message: The message to log
            exc_info: Whether to include exception info
            stack_info: Whether to include stack info
            **kwargs: Additional context to include in the log
        """
        # Merge the current context with the provided kwargs
        extra = {**self._context, **kwargs}

        # Get the caller information (more detailed)
        frame = inspect.currentframe()
        if frame:
            caller_frame = frame.f_back
            if caller_frame:
                # Basic caller info
                extra['file'] = caller_frame.f_code.co_filename
                extra['line'] = caller_frame.f_lineno
                extra['function'] = caller_frame.f_code.co_name

                # Enhanced caller info
                extra['module'] = inspect.getmodule(caller_frame).__name__ if inspect.getmodule(caller_frame) else "unknown"

                # Add local variables if in debug level
                if level <= logging.DEBUG:
                    try:
                        # Get local variables from the caller's frame
                        local_vars = {k: v for k, v in caller_frame.f_locals.items()
                                     if not k.startswith('_') and k != 'self' and not inspect.ismodule(v)}

                        # Add a subset of local variables to avoid excessive logging
                        # Filter out large objects, functions, etc.
                        filtered_vars = {}
                        for k, v in local_vars.items():
                            # Skip if it's a common large object type
                            if isinstance(v, (list, dict, set)) and len(v) > 10:
                                filtered_vars[k] = f"{type(v).__name__} with {len(v)} items"
                            elif hasattr(v, '__dict__') and not isinstance(v, type):
                                filtered_vars[k] = f"{type(v).__name__} instance"
                            elif not isinstance(v, (str, int, float, bool, type(None))):
                                filtered_vars[k] = f"{type(v).__name__}"
                            else:
                                filtered_vars[k] = v

                        extra['local_vars'] = filtered_vars
                    except Exception:
                        # Ignore errors when trying to get local variables
                        pass

        # Add timestamp for precise timing analysis
        extra['timestamp'] = datetime.now().isoformat()

        # Add thread information
        import threading
        extra['thread_name'] = threading.current_thread().name
        extra['thread_id'] = threading.get_ident()

        # Add process information
        extra['process_id'] = os.getpid()

        # Add enhanced process information if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                extra['process_info'] = {
                    'pid': process.pid,
                    'memory_percent': process.memory_percent(),
                    'cpu_percent': process.cpu_percent(interval=0.1),
                    'num_threads': process.num_threads()
                }
            except Exception:
                # Ignore errors when trying to get process info
                pass

        # Log the message with enhanced information
        self._logger.log(level, message, exc_info=exc_info, stack_info=stack_info, extra={'extra': extra})

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
