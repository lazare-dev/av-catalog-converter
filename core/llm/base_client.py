# core/llm/base_client.py
"""
Common LLM interface with enhanced logging for troubleshooting
"""
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from utils.logging.logger import Logger

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients with enhanced logging"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the LLM client

        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        self.model_config = model_config
        self.logger = Logger.get_logger(__name__)

        # Log initialization with detailed configuration
        self.logger.info(f"Initializing {self.__class__.__name__}",
                        client_type=self.__class__.__name__,
                        model_id=model_config.get('model_id', 'unknown'),
                        model_type=model_config.get('model_type', 'unknown'),
                        quantization=model_config.get('quantization'),
                        device_map=model_config.get('device_map', 'auto'))

    @abstractmethod
    def initialize_model(self):
        """
        Initialize and load the model
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model

        Args:
            prompt (str): Input prompt

        Returns:
            str: Model response
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts

        Args:
            prompts (List[str]): List of input prompts

        Returns:
            List[str]: List of model responses
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model

        Returns:
            Dict[str, Any]: Model metadata
        """
        return {
            "model_id": self.model_config.get("model_id", "unknown"),
            "quantization": self.model_config.get("quantization", None),
            "max_tokens": self.model_config.get("max_new_tokens", "unknown"),
        }

    def cleanup(self):
        """Release resources used by the model with enhanced logging"""
        if hasattr(self, 'model'):
            model_info = {
                'model_id': self.model_config.get('model_id', 'unknown'),
                'model_type': self.model_config.get('model_type', 'unknown'),
                'client_type': self.__class__.__name__
            }

            self.logger.info(f"Releasing resources for {model_info['model_id']}", **model_info)

            # Track memory before cleanup
            try:
                import psutil
                memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
                model_info['memory_before_mb'] = memory_before
            except (ImportError, Exception):
                pass

            # Delete model
            cleanup_start = time.time()
            del self.model
            self.model = None

            # Force garbage collection
            import gc
            gc.collect()

            # Track memory after cleanup
            try:
                import psutil
                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
                model_info['memory_after_mb'] = memory_after
                model_info['memory_freed_mb'] = memory_before - memory_after
            except (ImportError, Exception):
                pass

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    # Track GPU memory before cleanup
                    cuda_allocated_before = torch.cuda.memory_allocated() / (1024 * 1024)
                    cuda_reserved_before = torch.cuda.memory_reserved() / (1024 * 1024)
                    model_info['cuda_allocated_before_mb'] = cuda_allocated_before
                    model_info['cuda_reserved_before_mb'] = cuda_reserved_before

                    # Clear cache
                    torch.cuda.empty_cache()

                    # Track GPU memory after cleanup
                    cuda_allocated_after = torch.cuda.memory_allocated() / (1024 * 1024)
                    cuda_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)
                    model_info['cuda_allocated_after_mb'] = cuda_allocated_after
                    model_info['cuda_reserved_after_mb'] = cuda_reserved_after
                    model_info['cuda_freed_mb'] = cuda_reserved_before - cuda_reserved_after
            except (ImportError, Exception) as e:
                model_info['cuda_error'] = str(e)

            # Calculate total cleanup time
            cleanup_time = time.time() - cleanup_start
            model_info['cleanup_time_seconds'] = cleanup_time

            self.logger.info(f"Model resources released in {cleanup_time:.2f}s", **model_info)
