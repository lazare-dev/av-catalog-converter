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
    
    def start_task(self, task_name: str):
        """
        Start a new task
        
        Args:
            task_name (str): Name of the task
        """
        self.current_task = task_name
        self.start_time = time.time()
        self.progress = 0
        self.status = "running"
        
        self.logger.info(f"Started task: {task_name}")
        self._notify_callbacks()
    
    def update_task(self, message: str, progress: int):
        """
        Update task progress
        
        Args:
            message (str): Progress message
            progress (int): Progress percentage (0-100)
        """
        if not self.current_task:
            return
            
        self.progress = min(100, max(0, progress))
        
        self.logger.info(f"Progress ({self.progress}%): {message}")
        self._notify_callbacks()
    
    def complete_task(self, message: str = "Task completed"):
        """
        Mark the current task as completed
        
        Args:
            message (str, optional): Completion message
        """
        if not self.current_task:
            return
            
        elapsed = time.time() - self.start_time
        self.progress = 100
        self.status = "completed"
        
        self.logger.info(f"Task completed: {self.current_task} ({elapsed:.2f}s) - {message}")
        self._notify_callbacks()
        
        # Reset task
        self.current_task = None
        self.start_time = None
    
    def fail_task(self, error_message: str):
        """
        Mark the current task as failed
        
        Args:
            error_message (str): Error message
        """
        if not self.current_task:
            return
            
        elapsed = time.time() - self.start_time
        self.status = "failed"
        
        self.logger.error(f"Task failed: {self.current_task} ({elapsed:.2f}s) - {error_message}")
        self._notify_callbacks()
        
        # Reset task
        self.current_task = None
        self.start_time = None
    
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