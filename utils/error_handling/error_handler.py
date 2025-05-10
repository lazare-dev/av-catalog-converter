"""
Error handling utilities
"""
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import functools

from utils.logging.logger import Logger

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')

class AppError(Exception):
    """Base exception class for application errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 code: Optional[str] = None, http_status: int = 500):
        """
        Initialize the exception
        
        Args:
            message: Error message
            details: Additional error details
            code: Error code
            http_status: HTTP status code for API responses
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.code = code
        self.http_status = http_status
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary
        
        Returns:
            Dictionary representation of the exception
        """
        result = {
            'error': self.message,
            'code': self.code,
        }
        
        if self.details:
            result['details'] = self.details
            
        return result
    
    def __str__(self) -> str:
        """
        String representation of the exception
        
        Returns:
            String representation
        """
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message


class ValidationError(AppError):
    """Exception for validation errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 field: Optional[str] = None):
        """
        Initialize the exception
        
        Args:
            message: Error message
            details: Additional error details
            field: Field that failed validation
        """
        super().__init__(message, details, code='VALIDATION_ERROR', http_status=400)
        self.field = field
        
        if field and 'field' not in self.details:
            self.details['field'] = field


class NotFoundError(AppError):
    """Exception for resource not found errors"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 resource_id: Optional[str] = None):
        """
        Initialize the exception
        
        Args:
            message: Error message
            resource_type: Type of resource that was not found
            resource_id: ID of the resource that was not found
        """
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
            
        super().__init__(message, details, code='NOT_FOUND', http_status=404)


class ProcessingError(AppError):
    """Exception for processing errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 component: Optional[str] = None):
        """
        Initialize the exception
        
        Args:
            message: Error message
            details: Additional error details
            component: Component where the error occurred
        """
        super().__init__(message, details, code='PROCESSING_ERROR', http_status=500)
        
        if component and 'component' not in self.details:
            self.details['component'] = component


class ErrorHandler:
    """
    Utility for handling errors consistently
    
    This class provides tools for handling errors in a consistent way across the application.
    It can be used as a decorator or directly.
    
    Usage:
        # As a decorator
        @ErrorHandler.handle_errors
        def my_function():
            # Function code
            
        # Direct usage
        try:
            # Code that might raise an exception
        except Exception as e:
            ErrorHandler.handle_exception(e)
    """
    
    @staticmethod
    def handle_errors(func: F) -> F:
        """
        Decorator to handle errors consistently
        
        Args:
            func: The function to decorate
            
        Returns:
            The decorated function
        """
        logger = Logger.get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except AppError as e:
                # Log application errors
                logger.error(f"Application error in {func.__name__}: {str(e)}",
                            error_code=e.code,
                            error_details=e.details)
                raise
            except Exception as e:
                # Log and convert other exceptions
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}",
                            exc_info=True)
                
                # Convert to application error
                raise ProcessingError(
                    message=f"An unexpected error occurred: {str(e)}",
                    component=func.__module__
                ) from e
        
        return cast(F, wrapper)
    
    @staticmethod
    def handle_exception(e: Exception, logger: Optional[Logger] = None) -> AppError:
        """
        Handle an exception and convert it to an application error
        
        Args:
            e: The exception to handle
            logger: Logger to use (optional)
            
        Returns:
            An application error
        """
        if logger is None:
            logger = Logger.get_logger(__name__)
        
        if isinstance(e, AppError):
            # Already an application error
            logger.error(f"Application error: {str(e)}",
                        error_code=e.code,
                        error_details=e.details)
            return e
        
        # Convert to application error
        logger.error(f"Unexpected error: {str(e)}",
                    exc_info=True)
        
        return ProcessingError(
            message=f"An unexpected error occurred: {str(e)}"
        )
    
    @staticmethod
    def safe_execute(func: Callable[..., R], *args: Any, **kwargs: Any) -> Tuple[Optional[R], Optional[AppError]]:
        """
        Execute a function safely and return the result or error
        
        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (result, error) where one is None
        """
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            error = ErrorHandler.handle_exception(e)
            return None, error
