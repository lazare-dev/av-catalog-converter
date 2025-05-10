# core/llm/llm_factory.py
"""
LLM provider factory
"""
import logging
import os
from typing import Dict, Any, Optional, List
import time

from core.llm.base_client import BaseLLMClient
from core.llm.gemma_client import GemmaClient
from core.llm.phi_client import PhiClient
from config.settings import MODEL_CONFIG, ALT_MODEL_CONFIG

class LLMFactory:
    """Factory for creating appropriate LLM client"""
    
    # Mapping of model types to client classes
    CLIENT_MAP = {
        "gemma": GemmaClient,
        "phi": PhiClient,
    }
    
    # Keep track of initialized clients
    _clients = {}
    
    @classmethod
    def _get_model_type(cls, model_id: str) -> str:
        """
        Determine model type from model ID
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            str: Model type key
        """
        model_id = model_id.lower()
        
        if "gemma" in model_id:
            return "gemma"
        elif "phi" in model_id:
            return "phi"
        else:
            # Default to gemma
            return "gemma"
    
    @classmethod
    def create_client(cls, model_config: Optional[Dict[str, Any]] = None) -> BaseLLMClient:
        """
        Create and return the appropriate LLM client
        
        Args:
            model_config (Dict[str, Any], optional): Model configuration,
                defaults to config.settings.MODEL_CONFIG
            
        Returns:
            BaseLLMClient: Appropriate LLM client instance
        """
        logger = logging.getLogger(__name__)
        
        # Use provided config or fall back to default
        config = model_config or MODEL_CONFIG
        
        # Create a cache key based on the config
        cache_key = str(hash(frozenset(config.items())))
        
        # Return cached client if available
        if cache_key in cls._clients:
            logger.debug(f"Using cached LLM client for {config.get('model_id', 'unknown')}")
            return cls._clients[cache_key]
        
        model_id = config.get("model_id", "")
        model_type = cls._get_model_type(model_id)
        
        if model_type not in cls.CLIENT_MAP:
            fallback_config = ALT_MODEL_CONFIG
            fallback_model_id = fallback_config.get("model_id", "")
            fallback_type = cls._get_model_type(fallback_model_id)
            
            logger.warning(
                f"Unsupported model type: {model_type}, falling back to {fallback_type}"
            )
            
            model_type = fallback_type
            config = fallback_config
            
        client_class = cls.CLIENT_MAP[model_type]
        logger.info(f"Creating {client_class.__name__} for model {config['model_id']}")
        
        # Check available GPU memory before initializing
        cls._check_gpu_memory()
        
        # Create client
        client = client_class(config)
        
        # Cache the client
        cls._clients[cache_key] = client
        
        return client
    
    @classmethod
    def _check_gpu_memory(cls):
        """Check available GPU memory and log warning if low"""
        try:
            import torch
            if torch.cuda.is_available():
                # Get available memory for each GPU
                for i in range(torch.cuda.device_count()):
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    free_memory_gb = free_memory / (1024**3)
                    
                    logger = logging.getLogger(__name__)
                    if free_memory_gb < 2.0:
                        logger.warning(f"Low GPU memory on device {i}: {free_memory_gb:.2f} GB free")
                    else:
                        logger.info(f"GPU {i} memory: {free_memory_gb:.2f} GB free")
        except Exception as e:
            # Don't fail if torch isn't available or other issues
            pass
    
    @classmethod
    def cleanup_clients(cls):
        """Release resources for all initialized clients"""
        logger = logging.getLogger(__name__)
        
        for key, client in cls._clients.items():
            logger.info(f"Cleaning up LLM client: {client.__class__.__name__}")
