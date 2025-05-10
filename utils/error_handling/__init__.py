"""
Error handling utilities package
"""
from utils.error_handling.error_handler import (
    AppError, ValidationError, NotFoundError, ProcessingError, ErrorHandler
)

__all__ = [
    'AppError',
    'ValidationError',
    'NotFoundError',
    'ProcessingError',
    'ErrorHandler'
]
