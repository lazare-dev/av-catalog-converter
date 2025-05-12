# core/llm/gpt_client.py
"""
GPT-specific LLM implementation for AV Catalog Converter
Optimized for OpenAI's GPT-2 model
"""
import logging
import time
import traceback
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.llm.base_client import BaseLLMClient
from utils.caching.adaptive_cache import AdaptiveCache
from utils.rate_limiting.rate_limiter import RateLimiter

class GPTClient(BaseLLMClient):
    """Client for OpenAI's GPT-2 models"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the GPT client

        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self._initialization_error = None

        # Initialize counters for statistics
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0
        self.cache_hits = 0
        self.rate_limited_count = 0
        self.rate_limited_wait_time = 0

        # Setup cache if enabled
        self.cache = None
        if model_config.get("cache_enabled", True):
            # Get cache configuration
            cache_type = model_config.get("cache_type", "adaptive")
            base_ttl = model_config.get("cache_ttl", 3600)  # Default: 1 hour
            max_size = model_config.get("cache_max_size", 1000)

            if cache_type == "adaptive":
                self.cache = AdaptiveCache(
                    base_ttl=base_ttl,
                    max_size=max_size,
                    cleanup_interval=300  # Clean up every 5 minutes
                )
                self.logger.info(f"Adaptive LLM response caching enabled (base TTL: {base_ttl}s, max size: {max_size})")
            else:
                # Import MemoryCache for backward compatibility
                from utils.caching.memory_cache import MemoryCache
                self.cache = MemoryCache(ttl=base_ttl)
                self.logger.info(f"Standard LLM response caching enabled (TTL: {base_ttl}s)")

        # Set default parameters if not provided
        self.max_new_tokens = model_config.get("max_new_tokens", 128)
        self.temperature = model_config.get("temperature", 0.1)
        self.top_p = model_config.get("top_p", 0.9)
        self.repetition_penalty = model_config.get("repetition_penalty", 1.0)

        # Setup rate limiting
        self.rate_limiter = None
        if model_config.get("rate_limiting_enabled", True):
            # Calculate token cost based on prompt length
            def token_cost_func(prompt):
                # Estimate token count (rough approximation)
                return max(1, len(prompt) // 4)

            # Get rate limiting parameters
            requests_per_minute = model_config.get("requests_per_minute", 60)
            burst_size = model_config.get("burst_size", 1000)  # Increased to handle larger prompts

            self.rate_limiter = RateLimiter(
                requests_per_minute=requests_per_minute,
                burst_size=burst_size,
                token_cost_func=token_cost_func
            )
            self.logger.info(f"Rate limiting enabled: {requests_per_minute} requests/minute, burst size: {burst_size}")

    def initialize_model(self):
        """
        Initialize and load the GPT-2 model
        """
        if self._is_initialized:
            return

        if self._initialization_error:
            self.logger.warning(f"Previous initialization failed: {self._initialization_error}")

        self.logger.info(f"Initializing GPT model: {self.model_config['model_id']}")

        try:
            # Load tokenizer with proper error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config["model_id"],
                    use_fast=True
                )
                # Ensure the tokenizer has padding token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as tokenizer_error:
                self.logger.error(f"Error loading tokenizer: {str(tokenizer_error)}")
                self.logger.debug(traceback.format_exc())
                raise RuntimeError(f"Failed to load tokenizer: {str(tokenizer_error)}")

            # Load model with proper error handling and memory optimization
            try:
                # Set low_cpu_mem_usage to True for better memory management
                self.logger.info(f"Loading model {self.model_config['model_id']} with device_map={self.model_config.get('device_map', 'auto')}")

                # Print available memory
                import psutil
                self.logger.info(f"Available memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")

                # Load the model with conservative settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["model_id"],
                    device_map=self.model_config.get("device_map", "cpu"),
                    torch_dtype=torch.float32,  # Use float32 for better compatibility
                    low_cpu_mem_usage=True
                )
                self.logger.info(f"Model loaded successfully")
            except Exception as model_error:
                self.logger.error(f"Error loading model: {str(model_error)}")
                self.logger.debug(traceback.format_exc())
                raise RuntimeError(f"Failed to load model: {str(model_error)}")

            self._is_initialized = True
            self._initialization_error = None
            self.logger.info(f"GPT model initialized successfully")

        except Exception as e:
            error_msg = f"Error initializing GPT model: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self._initialization_error = str(e)
            raise RuntimeError(error_msg)

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt for the GPT-2 model

        Args:
            prompt (str): Raw prompt

        Returns:
            str: Formatted prompt
        """
        # GPT-2 works well with a simple prompt format
        return f"{prompt}\n"

    def _extract_response(self, generated_text: str, formatted_prompt: str) -> str:
        """
        Extract the model's response from the generated text

        Args:
            generated_text (str): Full generated text
            formatted_prompt (str): The formatted prompt that was used

        Returns:
            str: Extracted response
        """
        # Remove the prompt from the beginning
        if generated_text.startswith(formatted_prompt):
            response = generated_text[len(formatted_prompt):].strip()
        else:
            # If the prompt isn't at the beginning (unusual), try to find the response part
            response = generated_text.replace(formatted_prompt, "", 1).strip()

        return response

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the GPT-2 model

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

        # If initialization failed previously and we couldn't initialize now, return error
        if not self._is_initialized:
            return f"Error: Model not initialized. Previous error: {self._initialization_error}"

        # Format prompt for GPT-2
        formatted_prompt = self._format_prompt(prompt)

        # Apply rate limiting if enabled
        if self.rate_limiter is not None:
            rate_limit_start = time.time()
            token_cost = max(1, len(prompt) // 4)  # Estimate token count

            self.logger.debug(f"Applying rate limiting with token cost: {token_cost}")
            if not self.rate_limiter.bucket.consume(token_cost, wait=True):
                self.logger.warning("Rate limit exceeded, request rejected")
                self.rate_limited_count += 1
                return "Error: Rate limit exceeded. Please try again later."

            rate_limit_time = time.time() - rate_limit_start
            if rate_limit_time > 0.01:  # Only count significant waits
                self.rate_limited_count += 1
                self.rate_limited_wait_time += rate_limit_time
                self.logger.info(f"Request rate limited, waited {rate_limit_time:.2f}s")

        try:
            # Tokenize input with optimized settings
            input_ids = self.tokenizer.encode(formatted_prompt,
                                            return_tensors="pt",
                                            truncation=True,
                                            max_length=self.model_config.get("max_input_length", 512))

            # Check if model is available
            if self.model is None:
                return "Error: Model not loaded properly"

            # Move inputs to the same device as the model
            input_ids = input_ids.to(self.model.device)
            input_tokens = len(input_ids[0])

            # Generate response
            start_time = time.time()

            # Use inference mode for better performance
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Performance optimizations
                    use_cache=True,
                    num_beams=1,
                    early_stopping=False
                )

            generation_time = time.time() - start_time

            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the response part
            response = self._extract_response(generated_text, formatted_prompt)

            # Update stats
            self.total_generations += 1
            self.total_tokens_generated += len(outputs[0]) - input_tokens
            self.total_generation_time += generation_time

            tokens_per_second = (len(outputs[0]) - input_tokens) / generation_time if generation_time > 0 else 0
            self.logger.debug(f"Generated response in {generation_time:.2f}s ({len(outputs[0]) - input_tokens} tokens, {tokens_per_second:.1f} tokens/sec)")

            # Cache the response if enabled
            if self.cache is not None:
                self.cache.set(prompt, response)

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
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
            response = self.generate_response(prompt)
            responses.append(response)

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the model

        Returns:
            Dict[str, Any]: Usage statistics
        """
        avg_generation_time = 0
        if self.total_generations > 0:
            avg_generation_time = self.total_generation_time / self.total_generations

        stats = {
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "average_generation_time": avg_generation_time,
            "cache_hits": self.cache_hits,
            "model_id": self.model_config.get("model_id", "unknown"),
            "is_initialized": self._is_initialized,
            "rate_limited_count": self.rate_limited_count,
            "rate_limited_wait_time": self.rate_limited_wait_time
        }

        # Add rate limiter stats if available
        if self.rate_limiter is not None:
            rate_limiter_stats = self.rate_limiter.get_stats()
            stats["rate_limiter"] = rate_limiter_stats

        return stats

    def cleanup(self):
        """
        Release resources used by the GPT model
        """
        if not self._is_initialized:
            return

        self.logger.info("Cleaning up GPT model resources")

        try:
            # Delete model and tokenizer
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_initialized = False
            self.logger.info("GPT model resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error cleaning up GPT model: {str(e)}")
            self.logger.debug(traceback.format_exc())
