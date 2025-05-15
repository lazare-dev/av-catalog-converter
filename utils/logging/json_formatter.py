"""
JSON formatter for logging
"""
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List


class JsonFormatter(logging.Formatter):
    """
    Formatter for JSON-structured logs

    This formatter converts log records to JSON format, making them easier to parse
    and analyze with log management tools.
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
        """
        Initialize the formatter

        Args:
            fmt: Format string (not used in JSON formatter)
            datefmt: Date format string
        """
        super().__init__(fmt, datefmt)
        self.datefmt = datefmt or "%Y-%m-%dT%H:%M:%S.%fZ"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON with enhanced detail

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log record with detailed information
        """
        # Create base log data with enhanced details
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "level_number": record.levelno,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "process": {
                "id": record.process,
                "name": record.processName
            },
            "thread": {
                "id": record.thread,
                "name": record.threadName
            },
            "pathname": record.pathname,
            "filename": record.filename,
            "created": record.created,
            "msecs": record.msecs,
            "relative_created": record.relativeCreated,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self._format_exception(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            log_data["stack_info"] = record.stack_info

        # Add extra context if present
        if hasattr(record, 'extra') and record.extra:
            # Ensure all values are JSON serializable
            context = self._ensure_serializable(record.extra)
            log_data["context"] = context

        # Add any additional attributes from the record
        for attr, value in record.__dict__.items():
            if attr not in log_data and attr not in ('args', 'msg', 'exc_info', 'exc_text', 'stack_info', 'extra'):
                try:
                    log_data[attr] = self._ensure_serializable(value)
                except Exception:
                    # Skip attributes that can't be serialized
                    pass

        # Convert to JSON
        try:
            return json.dumps(log_data)
        except Exception as e:
            # Fallback if JSON serialization fails
            return json.dumps({
                "timestamp": self.formatTime(record, self.datefmt),
                "level": "ERROR",
                "logger": record.name,
                "message": f"Error formatting log record: {str(e)}",
                "original_message": record.getMessage(),
                "error": str(e)
            })

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """
        Format the record's creation time

        Args:
            record: Log record
            datefmt: Date format string

        Returns:
            Formatted timestamp
        """
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat() + "Z"

    def _format_exception(self, exc_info: Tuple[Any, Any, Any]) -> Dict[str, Any]:
        """
        Format exception information

        Args:
            exc_info: Exception information tuple

        Returns:
            Formatted exception information
        """
        exc_type, exc_value, exc_tb = exc_info

        # Format traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)

        return {
            "type": exc_type.__name__ if exc_type else None,
            "message": str(exc_value),
            "traceback": tb_lines
        }

    def _ensure_serializable(self, obj: Any) -> Any:
        """
        Ensure an object is JSON serializable

        Args:
            obj: Object to make serializable

        Returns:
            JSON serializable object
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._ensure_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            # Convert objects to dictionaries
            return self._ensure_serializable(obj.__dict__)
        else:
            # Convert anything else to string
            return str(obj)
