# utils/logging/formatters.py
"""
Custom log formatters
"""
import logging
import json
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console logs"""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'    # Reset
    }
    
    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        
        # Add color if supported
        if levelname in self.COLORS:
            message = f"{self.COLORS[levelname]}{message}{self.COLORS['RESET']}"
            
        return message

class JSONFormatter(logging.Formatter):
    """Formatter for JSON-structured logs"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        if hasattr(record, 'extra') and record.extra:
            log_data.update(record.extra)
            
        return json.dumps(log_data)