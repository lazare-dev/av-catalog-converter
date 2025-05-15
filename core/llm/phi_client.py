# core/llm/phi_client.py
"""
Phi-specific LLM implementation for AV Catalog Converter
Optimized for Microsoft's Phi-2 model with enhanced performance
"""
import logging
import os
import time
import traceback
from typing import Dict, Any, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# Check if BitsAndBytesConfig is available (newer versions of transformers)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

from core.llm.base_client import BaseLLMClient
from utils.caching.adaptive_cache import AdaptiveCache
from utils.rate_limiting.rate_limiter import RateLimiter
from utils.logging.logger import Logger

class PhiClient(BaseLLMClient):
    """Client for Microsoft's Phi-2 models with optimizations"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the Phi client

        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        super().__init__(model_config)

        # Set default model if not specified
        if "model_id" not in self.model_config or not self.model_config["model_id"]:
            self.model_config["model_id"] = "microsoft/phi-2"
            self.logger.info("No model_id specified, using default: microsoft/phi-2")

        # Initialize model and tokenizer to None
        self.model = None
        self.tokenizer = None
        self.text_generation_pipeline = None
        self._is_initialized = False

        # Initialize cache if enabled
        self.cache = None
        self.cache_hits = 0
        if self.model_config.get("cache_enabled", True):
            cache_type = self.model_config.get("cache_type", "adaptive")
            if cache_type == "adaptive":
                self.cache = AdaptiveCache(
                    base_ttl=self.model_config.get("cache_ttl", 3600),
                    max_size=self.model_config.get("cache_max_size", 1000)
                )
                self.logger.info("Initialized adaptive cache",
                               base_ttl=self.model_config.get("cache_ttl", 3600),
                               max_size=self.model_config.get("cache_max_size", 1000))
            elif cache_type == "memory":
                from utils.caching.memory_cache import MemoryCache
                self.cache = MemoryCache(ttl=self.model_config.get("cache_ttl", 3600))
                self.logger.info("Initialized memory cache",
                               ttl=self.model_config.get("cache_ttl", 3600))
            elif cache_type == "disk":
                from utils.caching.disk_cache import DiskCache
                self.cache = DiskCache(ttl=self.model_config.get("cache_ttl", 3600))
                self.logger.info("Initialized disk cache",
                               ttl=self.model_config.get("cache_ttl", 3600))

        # Initialize rate limiter if enabled
        self.rate_limiter = None
        self.rate_limited_count = 0
        self.rate_limited_wait_time = 0
        if self.model_config.get("rate_limiting_enabled", True):
            requests_per_minute = self.model_config.get("requests_per_minute", 60)
            burst_size = self.model_config.get("burst_size", 10)

            # Calculate token cost based on prompt length
            def token_cost_func(prompt):
                # Estimate token count (rough approximation)
                return max(1, len(prompt) // 4)

            self.rate_limiter = RateLimiter(
                requests_per_minute=requests_per_minute,
                burst_size=burst_size,
                token_cost_func=token_cost_func
            )
            self.logger.info("Initialized rate limiter",
                           requests_per_minute=requests_per_minute,
                           burst_size=burst_size)

    def initialize_model(self):
        """Initialize the Phi model and tokenizer"""
        if self._is_initialized:
            self.logger.debug("Model already initialized")
            return

        self.logger.info(f"Initializing Phi model: {self.model_config['model_id']}")

        try:
            # Load tokenizer
            self.logger.debug("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["model_id"])

            # Configure quantization if enabled
            quantization = self.model_config.get("quantization", None)
            device_map = self.model_config.get("device_map", "auto")

            # Log memory information
            if psutil:
                mem_info = psutil.virtual_memory()
                self.logger.info(f"Available memory: {mem_info.available / (1024 * 1024 * 1024):.2f} GB")

            # Load model with quantization if specified
            if quantization == "8bit" and BITSANDBYTES_AVAILABLE:
                self.logger.info("Loading model with 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["model_id"],
                    device_map=device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            elif quantization == "4bit" and BITSANDBYTES_AVAILABLE:
                self.logger.info("Loading model with 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["model_id"],
                    device_map=device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            else:
                self.logger.info(f"Loading model with standard settings (device_map={device_map})")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["model_id"],
                    device_map=device_map,
                    trust_remote_code=True
                )

            # Create text generation pipeline
            self.logger.debug("Creating text generation pipeline")
            self.text_generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.model_config.get("max_new_tokens", 256),
                temperature=self.model_config.get("temperature", 0.7),
                top_p=self.model_config.get("top_p", 0.9),
                top_k=self.model_config.get("top_k", 50),
                repetition_penalty=self.model_config.get("repetition_penalty", 1.1)
            )

            self._is_initialized = True
            self.logger.info("Phi model initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Phi model: {str(e)}", exc_info=True)
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Phi model: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using Phi

        Args:
            prompt (str): Input prompt

        Returns:
            str: Model response
        """
        # Check cache first if enabled
        if self.cache is not None:
            cached_response = self.cache.get(prompt)
            if cached_response is not None:
                self.logger.debug("Using cached response")
                self.cache_hits += 1
                return cached_response

        # Initialize model if not already done
        if not self._is_initialized:
            try:
                self.initialize_model()
            except Exception as init_error:
                error_msg = f"Failed to initialize model: {str(init_error)}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        # Apply rate limiting if enabled
        if self.rate_limiter is not None:
            rate_limit_start = time.time()
            token_cost = max(1, len(prompt) // 4)  # Estimate token count
            fallback_on_rate_limit = self.model_config.get("fallback_on_rate_limit", True)

            self.logger.debug(f"Applying rate limiting with token cost: {token_cost}")

            # Use the check_limit method to check if we're rate limited
            if hasattr(self.rate_limiter, 'check_limit'):
                if not self.rate_limiter.check_limit(token_cost, wait=False):
                    self.rate_limited_count += 1
                    self.logger.warning("Rate limit exceeded, request rejected")

                    if fallback_on_rate_limit:
                        self.logger.info("Using fallback mechanism for rate-limited request")
                        return self._generate_fallback_response(prompt)
                    else:
                        return "Error: Rate limit exceeded. Please try again later."
            else:
                # Fallback to old method if check_limit doesn't exist
                if not self.rate_limiter.bucket.consume(token_cost, wait=False):
                    self.rate_limited_count += 1
                    self.logger.warning("Rate limit exceeded, request rejected")

                    if fallback_on_rate_limit:
                        self.logger.info("Using fallback mechanism for rate-limited request")
                        return self._generate_fallback_response(prompt)
                    else:
                        return "Error: Rate limit exceeded. Please try again later."

            # Track wait time
            rate_limit_time = time.time() - rate_limit_start
            if rate_limit_time > 0.01:  # Only count significant waits
                self.rate_limited_wait_time += rate_limit_time
                self.logger.debug(f"Rate limiting caused wait of {rate_limit_time:.2f}s")

        try:
            # Generate response using the text generation pipeline
            self.logger.debug(f"Generating response for prompt: {prompt[:50]}...")
            start_time = time.time()
            
            # Format the prompt for Phi
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate text
            outputs = self.text_generation_pipeline(
                formatted_prompt,
                return_full_text=False,
                do_sample=True
            )
            
            generation_time = time.time() - start_time
            
            # Extract the generated text
            response = outputs[0]['generated_text'].strip()
            
            self.logger.debug(f"Generated response in {generation_time:.2f}s")

            # Cache the response if caching is enabled
            if self.cache is not None:
                self.cache.set(prompt, response)

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

            # Return error message
            return f"Error: {error_msg}"

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt for Phi model

        Args:
            prompt (str): Input prompt

        Returns:
            str: Formatted prompt
        """
        # Simple formatting for Phi
        return f"Instruction: {prompt}\nResponse:"

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a fallback response when rate limited

        Args:
            prompt (str): Input prompt

        Returns:
            str: Fallback response
        """
        # Simple fallback response
        return "I'm currently experiencing high demand. Please try again in a moment."

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts

        Args:
            prompts (List[str]): List of input prompts

        Returns:
            List[str]: List of model responses
        """
        responses = []
        for prompt in prompts:
            responses.append(self.generate_response(prompt))
        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model

        Returns:
            Dict[str, Any]: Model metadata
        """
        info = super().get_model_info()
        info.update({
            "model_type": "phi",
            "is_initialized": self._is_initialized,
            "cache_hits": self.cache_hits,
            "rate_limited_count": self.rate_limited_count,
            "rate_limited_wait_time": self.rate_limited_wait_time
        })
        return info

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the client's performance

        Returns:
            Dict[str, Any]: Statistics
        """
        stats = {
            "model_id": self.model_config.get("model_id", "unknown"),
            "model_type": "phi",
            "is_initialized": self._is_initialized,
            "total_generations": 0,  # Will be updated if available
            "cache_hits": self.cache_hits,
            "rate_limited_count": self.rate_limited_count,
            "rate_limited_wait_time": self.rate_limited_wait_time
        }

        # Add cache stats if available
        if self.cache is not None and hasattr(self.cache, "get_stats"):
            cache_stats = self.cache.get_stats()
            stats.update({
                "cache_" + k: v for k, v in cache_stats.items()
            })

        # Add rate limiter stats if available
        if self.rate_limiter is not None and hasattr(self.rate_limiter, "get_stats"):
            rate_stats = self.rate_limiter.get_stats()
            stats.update({
                "rate_" + k: v for k, v in rate_stats.items()
            })

        return stats
