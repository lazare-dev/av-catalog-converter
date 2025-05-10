# core/llm/base_client.py
"""
Common LLM interface
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the LLM client
        
        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        self.model_config = model_config
        self.logger = logging.getLogger(__name__)
    
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
        """Release resources used by the model"""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Model resources released")
