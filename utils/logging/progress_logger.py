# utils/logging/progress_logger.py
"""
Progress tracking logger
"""
import logging
import time
from typing import Optional, Dict, Any, Callable

class ProgressLogger:
    """Utility for tracking and logging task progress"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the progress logger

        Args:
            logger (Logger, optional): Logger to use for output
        """
        self.logger = logger or logging.getLogger(__name__)
        self.current_task = None
        self.start_time = None
        self.progress = 0
        self.status = "idle"
        self.callbacks = []

    def start_task(self, task_name: str, task_details: dict = None):
        """
        Start a new task with enhanced logging

        Args:
            task_name (str): Name of the task
            task_details (dict, optional): Additional details about the task
        """
        self.current_task = task_name
        self.start_time = time.time()
        self.progress = 0
        self.status = "running"
        self.task_details = task_details or {}
        self.steps_completed = 0
        self.total_steps = self.task_details.get('total_steps', 0)

        # Log with detailed information
        log_data = {
            "task": task_name,
            "status": "started",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": self.task_details
        }

        self.logger.info(f"Started task: {task_name}", extra={"progress_info": log_data})
        self._notify_callbacks()

    def update_task(self, message: str, progress: int, step_name: str = None, step_details: dict = None):
        """
        Update task progress with enhanced logging

        Args:
            message (str): Progress message
            progress (int): Progress percentage (0-100)
            step_name (str, optional): Name of the current step
            step_details (dict, optional): Additional details about the step
        """
        if not self.current_task:
            return

        self.progress = min(100, max(0, progress))
        self.steps_completed += 1 if step_name else 0

        # Calculate elapsed time and estimated time remaining
        elapsed = time.time() - self.start_time

        # Estimate time remaining if we have progress > 0
        eta = None
        if self.progress > 0:
            eta = (elapsed / self.progress) * (100 - self.progress)

        # Create detailed log data
        log_data = {
            "task": self.current_task,
            "status": "in_progress",
            "progress": self.progress,
            "message": message,
            "elapsed_seconds": elapsed,
            "estimated_seconds_remaining": eta,
            "step": {
                "name": step_name,
                "details": step_details or {},
                "number": self.steps_completed,
                "total": self.total_steps
            }
        }

        # Log with different levels based on progress
        if self.progress < 25:
            self.logger.debug(f"Progress ({self.progress}%): {message}", extra={"progress_info": log_data})
        elif self.progress < 50:
            self.logger.info(f"Progress ({self.progress}%): {message}", extra={"progress_info": log_data})
        elif self.progress < 75:
            self.logger.info(f"Progress ({self.progress}%): {message}", extra={"progress_info": log_data})
        else:
            self.logger.info(f"Progress ({self.progress}%): {message}", extra={"progress_info": log_data})

        self._notify_callbacks()

    def complete_task(self, message: str = "Task completed", result_details: dict = None):
        """
        Mark the current task as completed with enhanced logging

        Args:
            message (str, optional): Completion message
            result_details (dict, optional): Details about the task result
        """
        if not self.current_task:
            return

        elapsed = time.time() - self.start_time
        self.progress = 100
        self.status = "completed"

        # Create detailed log data
        log_data = {
            "task": self.current_task,
            "status": "completed",
            "elapsed_seconds": elapsed,
            "message": message,
            "result": result_details or {},
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps
        }

        self.logger.info(
            f"Task completed: {self.current_task} ({elapsed:.2f}s) - {message}",
            extra={"progress_info": log_data}
        )
        self._notify_callbacks()

        # Reset task
        self.current_task = None
        self.start_time = None
        self.task_details = {}
        self.steps_completed = 0
        self.total_steps = 0

    def fail_task(self, error_message: str, error_details: dict = None):
        """
        Mark the current task as failed with enhanced logging

        Args:
            error_message (str): Error message
            error_details (dict, optional): Additional details about the error
        """
        if not self.current_task:
            return

        elapsed = time.time() - self.start_time
        self.status = "failed"

        # Create detailed log data
        log_data = {
            "task": self.current_task,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "error": error_message,
            "error_details": error_details or {},
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps
        }

        self.logger.error(
            f"Task failed: {self.current_task} ({elapsed:.2f}s) - {error_message}",
            exc_info=True,
            extra={"progress_info": log_data}
        )
        self._notify_callbacks()

        # Reset task
        self.current_task = None
        self.start_time = None
        self.task_details = {}
        self.steps_completed = 0
        self.total_steps = 0

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a progress update callback

        Args:
            callback (Callable): Function to call with progress updates
        """
        self.callbacks.append(callback)

    def _notify_callbacks(self):
        """
        Notify all registered callbacks with current progress
        """
        progress_info = {
            "task": self.current_task,
            "progress": self.progress,
            "status": self.status,
            "elapsed": time.time() - self.start_time if self.start_time else 0
        }

        for callback in self.callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {str(e)}")