# utils/logging/formatters.py
"""
Custom log formatters
"""
import logging
import json
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console logs with enhanced detail"""

    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m',    # Reset
        'BOLD': '\033[1m',     # Bold
        'UNDERLINE': '\033[4m', # Underline
        'FILENAME': '\033[36m', # Cyan
        'LINENO': '\033[36m',   # Cyan
        'FUNCNAME': '\033[36m'  # Cyan
    }

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        """Initialize with a more detailed format if none provided"""
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d (%(funcName)s) - %(message)s"
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        levelname = record.levelname

        # Add detailed information to the record
        record.pathname_colored = f"{self.COLORS['FILENAME']}{record.pathname}{self.COLORS['RESET']}"
        record.filename_colored = f"{self.COLORS['FILENAME']}{record.filename}{self.COLORS['RESET']}"
        record.lineno_colored = f"{self.COLORS['LINENO']}{record.lineno}{self.COLORS['RESET']}"
        record.funcName_colored = f"{self.COLORS['FUNCNAME']}{record.funcName}{self.COLORS['RESET']}"

        # Format the message
        message = super().format(record)

        # Add color to the level name
        if levelname in self.COLORS:
            # Replace the level name with a colored version
            colored_levelname = f"{self.COLORS[levelname]}{levelname:8}{self.COLORS['RESET']}"
            message = message.replace(f"[{levelname}]", f"[{colored_levelname}]")

        return message

class JSONFormatter(logging.Formatter):
    """Formatter for JSON-structured logs with enhanced detail"""

    def format(self, record):
        # Create a detailed log data structure
        log_data = {
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
            "level": {
                "name": record.levelname,
                "number": record.levelno
            },
            "logger": record.name,
            "source": {
                "module": record.module,
                "function": record.funcName,
                "file": record.pathname,
                "filename": record.filename,
                "line": record.lineno
            },
            "message": record.getMessage(),
            "process": {
                "id": record.process,
                "name": record.processName
            },
            "thread": {
                "id": record.thread,
                "name": record.threadName
            },
            "created": record.created,
            "msecs": record.msecs,
            "relative_created": record.relativeCreated
        }

        # Add exception info if present
        if record.exc_info:
            import traceback
            exc_type, exc_value, exc_tb = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb) if exc_tb else None
            }

        # Add stack info if present
        if record.stack_info:
            log_data["stack_info"] = record.stack_info

        # Add extra context if present
        if hasattr(record, 'extra') and record.extra:
            log_data["context"] = record.extra

        # Add any additional attributes from the record
        for attr, value in record.__dict__.items():
            if attr not in ["args", "msg", "exc_info", "exc_text", "stack_info", "extra",
                           "name", "levelname", "levelno", "pathname", "filename", "module",
                           "lineno", "funcName", "created", "msecs", "relativeCreated",
                           "thread", "threadName", "process", "processName"]:
                try:
                    # Try to serialize the value
                    json.dumps({attr: value})
                    log_data[attr] = value
                except (TypeError, OverflowError):
                    # If it can't be serialized, convert to string
                    log_data[attr] = str(value)

        return json.dumps(log_data)