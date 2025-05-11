# core/llm/llm_factory.py
"""
LLM provider factory for AV Catalog Converter
Manages creation and caching of LLM clients
"""
import logging
import traceback
from typing import Dict, Any, Optional, List

from core.llm.base_client import BaseLLMClient
from core.llm.phi_client import PhiClient
from config.settings import MODEL_CONFIG

class LLMFactory:
    """Factory for creating and managing LLM clients"""

    # Client mapping - currently only using Phi-2 model
    CLIENT_MAP = {
        "phi": PhiClient,
    }

    # Cache of initialized clients
    _clients = {}

    # Track initialization errors
    _init_errors = {}

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

        # Map model ID to client type
        if "phi" in model_id:
            return "phi"

        # Default to phi for now as it's our primary model
        return "phi"

    # Track initialization time
    _init_time = 0

    @classmethod
    def create_client(cls, model_config: Optional[Dict[str, Any]] = None) -> BaseLLMClient:
        """
        Create and return the appropriate LLM client
        Uses caching to avoid recreating clients

        Args:
            model_config (Dict[str, Any], optional): Model configuration,
                defaults to config.settings.MODEL_CONFIG

        Returns:
            BaseLLMClient: Appropriate LLM client instance
        """
        import time
        logger = logging.getLogger(__name__)

        # Use provided config or fall back to default
        config = model_config or MODEL_CONFIG.copy()

        # Ensure required fields are present
        if "model_id" not in config:
            config["model_id"] = "microsoft/phi-2"
            logger.warning(f"No model_id specified, using default: {config['model_id']}")

        # Create a cache key based on the config
        cache_key = str(hash(frozenset(config.items())))

        # Fast path: Return cached client if available and not in error state
        if cache_key in cls._clients and cache_key not in cls._init_errors:
            logger.debug(f"Using cached LLM client for {config.get('model_id')}")
            return cls._clients[cache_key]

        # If we previously had an error with this config, log a warning
        if cache_key in cls._init_errors:
            logger.warning(f"Previous initialization error with {config.get('model_id')}: {cls._init_errors[cache_key]}")
            # Remove from error cache to allow retry
            del cls._init_errors[cache_key]

        # Determine model type
        model_id = config.get("model_id", "")
        model_type = cls._get_model_type(model_id)

        if model_type not in cls.CLIENT_MAP:
            logger.warning(f"Unsupported model type: {model_type}, falling back to Phi-2 model")
            model_type = "phi"

        # Get client class
        client_class = cls.CLIENT_MAP[model_type]
        logger.info(f"Creating {client_class.__name__} for model {config['model_id']}")

        # Track initialization time
        start_time = time.time()

        try:
            # Create client
            client = client_class(config)

            # Cache the client
            cls._clients[cache_key] = client

            # Log initialization time
            init_time = time.time() - start_time
            cls._init_time = init_time
            logger.info(f"Initialized {client_class.__name__} in {init_time:.2f}s")

            return client

        except Exception as e:
            error_msg = f"Failed to initialize {client_class.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())

            # Store error for future reference
            cls._init_errors[cache_key] = str(e)

            # Fall back to a simpler configuration if possible
            if "quantization" in config and config["quantization"] is not None:
                logger.info("Attempting to create client without quantization")
                fallback_config = config.copy()
                fallback_config["quantization"] = None
                return cls.create_client(fallback_config)

            # If we can't create a client, raise an exception
            raise RuntimeError(f"Could not initialize LLM client: {error_msg}")

    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types

        Returns:
            List[str]: List of available model types
        """
        return list(cls.CLIENT_MAP.keys())

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        Get statistics about LLM clients

        Returns:
            Dict[str, Any]: Statistics about LLM clients
        """
        stats = {
            'clients_count': len(cls._clients),
            'errors_count': len(cls._init_errors),
            'init_time': cls._init_time,
            'available_models': cls.get_available_models(),
            'clients': {}
        }

        # Add stats for each client
        for key, client in cls._clients.items():
            client_name = client.__class__.__name__
            if hasattr(client, 'get_stats'):
                stats['clients'][client_name] = client.get_stats()
            elif hasattr(client, 'model_id'):
                stats['clients'][client_name] = {'model_id': client.model_id}
            else:
                stats['clients'][client_name] = {'status': 'active'}

        return stats

    @classmethod
    def cleanup_clients(cls):
        """Release resources for all initialized clients"""
        logger = logging.getLogger(__name__)

        for key, client in cls._clients.items():
            try:
                logger.info(f"Cleaning up LLM client: {client.__class__.__name__}")
                client.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up client: {str(e)}")

        # Clear caches
        cls._clients.clear()
        cls._init_errors.clear()
