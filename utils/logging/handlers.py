# utils/logging/handlers.py
"""
Custom log handlers
"""
import logging
import os
import sys
from pathlib import Path

class RotatingFileHandler(logging.Handler):
    """Custom handler for file rotation with size limit"""
    
    def __init__(self, filename, max_bytes=10485760, backup_count=5, encoding=None):
        """
        Initialize the handler
        
        Args:
            filename (str): Log filename
            max_bytes (int): Maximum file size in bytes before rotating
            backup_count (int): Number of backup files to keep
            encoding (str, optional): File encoding
        """
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding or 'utf-8'
        self.current_size = 0
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Check if file exists and get size
        path = Path(filename)
        if path.exists():
            self.current_size = path.stat().st_size
            
        # Open file for writing
        self.file = open(filename, 'a', encoding=self.encoding)
    
    def emit(self, record):
        """
        Emit a log record
        
        Args:
            record (LogRecord): The log record to emit
        """
        try:
            msg = self.format(record)
            msg_bytes = (msg + '\n').encode(self.encoding)
            msg_size = len(msg_bytes)
            
            # Check if rotation needed
            if self.current_size + msg_size > self.max_bytes:
                self._rotate_files()
                
            # Write log entry
            self.file.write(msg + '\n')
            self.file.flush()
            self.current_size += msg_size
            
        except Exception:
            self.handleError(record)
    
    def _rotate_files(self):
        """Rotate log files"""
        self.file.close()
        
        # Delete oldest log file if it exists
        oldest = f"{self.filename}.{self.backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)
            
        # Rotate existing backup files
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.filename}.{i}"
            dst = f"{self.filename}.{i+1}"
            if os.path.exists(src):
                os.rename(src, dst)
                
        # Rename current log file
        if os.path.exists(self.filename):
            os.rename(self.filename, f"{self.filename}.1")
            
        # Open new file
        self.file = open(self.filename, 'a', encoding=self.encoding)
        self.current_size = 0
    
    def close(self):
        """Close the handler"""
        if hasattr(self, 'file') and self.file:
            self.file.close()
            
        super().close()