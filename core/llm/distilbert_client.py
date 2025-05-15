# core/llm/distilbert_client.py
"""
DistilBERT-specific LLM implementation for AV Catalog Converter
Optimized for Hugging Face's DistilBERT model with enhanced performance
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

from transformers import AutoTokenizer, AutoModelForMaskedLM
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

class DistilBERTClient(BaseLLMClient):
    """Client for Hugging Face's DistilBERT models with optimizations"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the DistilBERT client

        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        super().__init__(model_config)

        # Set default model if not specified
        if "model_id" not in self.model_config or not self.model_config["model_id"]:
            self.model_config["model_id"] = "distilbert-base-uncased"
            self.logger.info("No model_id specified, using default: distilbert-base-uncased")

        # Initialize model and tokenizer to None
        self.model = None
        self.tokenizer = None
        self.fill_mask_pipeline = None
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
        """
        Initialize and load the DistilBERT model with optimizations
        """
        if self._is_initialized:
            self.logger.debug("Model already initialized")
            return

        self.logger.info("Initializing DistilBERT model",
                       model_id=self.model_config["model_id"],
                       device_map=self.model_config.get("device_map", "auto"))

        try:
            # Print available memory if psutil is available
            if psutil:
                self.logger.info(f"Available memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
            else:
                self.logger.info("psutil not available, skipping memory check")

            # Check for environment variable override for quantization
            env_quantization = os.environ.get('MODEL_QUANTIZATION')
            if env_quantization:
                self.logger.info(f"Using environment-specified quantization: {env_quantization}")
                quantization = env_quantization
            else:
                # Use config-specified quantization
                quantization = self.model_config.get("quantization")

            quantization_config = None

            if quantization and BITSANDBYTES_AVAILABLE:
                self.logger.info(f"Using {quantization} quantization")
                if quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                elif quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            else:
                if quantization and not BITSANDBYTES_AVAILABLE:
                    self.logger.warning(f"Quantization {quantization} requested but bitsandbytes not available")

            # Load tokenizer
            self.logger.info(f"Loading tokenizer for {self.model_config['model_id']}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["model_id"])

            # Load model with proper configuration
            self.logger.info(f"Loading model {self.model_config['model_id']}")

            # Use masked language model for text generation
            model_kwargs = {
                "low_cpu_mem_usage": self.model_config.get("low_cpu_mem_usage", True),
            }

            # Add torch dtype if torch is available and CUDA is available
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16

                # Add quantization config if available and on GPU
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
            elif torch:
                model_kwargs["torch_dtype"] = torch.float32
                # Don't use quantization on CPU

            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_config["model_id"],
                **model_kwargs
            )

            # Create fill-mask pipeline for text generation
            # Note: Some versions of transformers don't accept 'device' parameter directly
            # So we'll create the pipeline without specifying device and then move the model
            try:
                device = "cuda" if torch and hasattr(torch, 'cuda') and torch.cuda.is_available() else "cpu"
                self.logger.info(f"Creating fill-mask pipeline with device: {device}")

                # First try with device parameter
                self.fill_mask_pipeline = pipeline(
                    "fill-mask",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device
                )
            except TypeError as e:
                # If device parameter is not accepted, create without it
                self.logger.warning(f"Creating pipeline without device parameter: {str(e)}")
                self.fill_mask_pipeline = pipeline(
                    "fill-mask",
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            # Move model to the appropriate device after pipeline creation
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                self.model.to("cuda")
            else:
                self.model.to("cpu")

            self._is_initialized = True
            self.logger.info("DistilBERT model initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize DistilBERT model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Don't raise the exception - we'll handle it gracefully in the generate_response method
            self._is_initialized = False

    def _generate_text_with_masked_lm(self, prompt: str) -> str:
        """
        Generate text using the masked language model approach

        Args:
            prompt (str): Input prompt with [MASK] tokens

        Returns:
            str: Generated text with masks filled
        """
        # Ensure prompt contains mask tokens
        if "[MASK]" not in prompt:
            # Add mask token at the end for continuation
            prompt += " [MASK]" * 5

        # Process the prompt in chunks if it's too long
        max_length = self.tokenizer.model_max_length - 10
        result = ""

        # Split long prompts into chunks
        if len(self.tokenizer.encode(prompt)) > max_length:
            chunks = []
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

            for i in range(0, len(tokens), max_length):
                chunk_tokens = tokens[i:i+max_length]
                chunk = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk)

            # Process each chunk
            for chunk in chunks:
                if "[MASK]" in chunk:
                    chunk_result = self._process_masked_chunk(chunk)
                    result += chunk_result
                else:
                    result += chunk
        else:
            result = self._process_masked_chunk(prompt)

        return result

    def _process_masked_chunk(self, text: str) -> str:
        """
        Process a chunk of text with mask tokens

        Args:
            text (str): Text chunk with [MASK] tokens

        Returns:
            str: Processed text with masks filled
        """
        # Count mask tokens
        mask_count = text.count("[MASK]")

        # If no masks, return the text as is
        if mask_count == 0:
            return text

        # Fill masks one by one
        for _ in range(mask_count):
            if "[MASK]" not in text:
                break

            try:
                # Get predictions for the first mask
                predictions = self.fill_mask_pipeline(text, top_k=1)

                # Handle different return formats
                if isinstance(predictions, list) and isinstance(predictions[0], list):
                    # Multiple masks were filled at once
                    prediction = predictions[0][0]
                elif isinstance(predictions, list):
                    # Single mask was filled
                    prediction = predictions[0]
                else:
                    # Unexpected format
                    self.logger.warning(f"Unexpected prediction format: {type(predictions)}")
                    break

                # Replace the text with the filled version
                text = prediction["sequence"]

            except Exception as e:
                self.logger.error(f"Error filling mask: {str(e)}")
                break

        return text

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using DistilBERT

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
            if not hasattr(self.rate_limiter, 'check_limit'):
                # Fallback to old method if check_limit doesn't exist
                if not self.rate_limiter.bucket.consume(token_cost, wait=False):
                    self.rate_limited_count += 1
                    self.logger.warning("Rate limit exceeded, request rejected")

                    if fallback_on_rate_limit:
                        self.logger.info("Using fallback mechanism for rate-limited request")
                        return self._generate_fallback_response(prompt)
                    else:
                        return "Error: Rate limit exceeded. Please try again later."
            else:
                # Use the new check_limit method
                if not self.rate_limiter.check_limit(token_cost, wait=False):
                    self.rate_limited_count += 1
                    self.logger.warning("Rate limit exceeded, request rejected")

                    if fallback_on_rate_limit:
                        self.logger.info("Using fallback mechanism for rate-limited request")
                        return self._generate_fallback_response(prompt)
                    else:
                        return "Error: Rate limit exceeded. Please try again later."

            # If we're here, we've consumed tokens successfully
            # Track wait time
            rate_limit_time = time.time() - rate_limit_start
            if rate_limit_time > 0.01:  # Only count significant waits
                self.rate_limited_wait_time += rate_limit_time
                self.logger.debug(f"Rate limiting caused wait of {rate_limit_time:.2f}s")

        try:
            # Generate response using masked language model approach
            response = self._generate_text_with_masked_lm(prompt)

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
            "model_type": "distilbert",
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
            "model_type": "distilbert",
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

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a fallback response when rate limited

        Args:
            prompt (str): Input prompt

        Returns:
            str: Fallback response
        """
        self.logger.info("Generating fallback response")

        # Simple keyword-based response for common tasks
        lower_prompt = prompt.lower()

        if "field mapping" in lower_prompt or "map fields" in lower_prompt:
            return '{"mappings": {"Please try again later": "Rate limit exceeded"}}'

        if "structure analysis" in lower_prompt or "analyze structure" in lower_prompt:
            return '{"structure_notes": "Rate limit exceeded. Please try again later."}'

        if "category" in lower_prompt or "categorize" in lower_prompt:
            return '{"category": "Unknown", "reason": "Rate limit exceeded"}'

        # Generic fallback
        return "Rate limit exceeded. Please try again later."

    def cleanup(self):
        """Release resources used by the model"""
        super().cleanup()

        # Additional cleanup specific to DistilBERT
        if hasattr(self, 'fill_mask_pipeline'):
            del self.fill_mask_pipeline
            self.fill_mask_pipeline = None
