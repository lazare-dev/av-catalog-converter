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

        # Get token count
        try:
            token_count = len(self.tokenizer.encode(prompt))
            self.logger.info(f"Prompt token count: {token_count}/{max_length}")

            # Split long prompts into chunks
            if token_count > max_length:
                self.logger.info(f"Prompt exceeds token limit ({token_count} > {max_length}), chunking prompt")
                chunks = self._chunk_prompt(prompt, max_length)
                self.logger.info(f"Split prompt into {len(chunks)} chunks")

                # Process each chunk
                for i, chunk in enumerate(chunks):
                    self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    if "[MASK]" in chunk:
                        chunk_result = self._process_masked_chunk(chunk)
                        result += chunk_result
                    else:
                        result += chunk
            else:
                result = self._process_masked_chunk(prompt)

            return result

        except Exception as e:
            self.logger.error(f"Error in _generate_text_with_masked_lm: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

            # Try fallback chunking approach if tokenization fails
            try:
                self.logger.info("Attempting fallback chunking approach")
                chunks = self._chunk_text_by_size(prompt, chunk_size=400)  # Approximate size
                self.logger.info(f"Split prompt into {len(chunks)} chunks using fallback approach")

                # Process each chunk
                for i, chunk in enumerate(chunks):
                    self.logger.info(f"Processing chunk {i+1}/{len(chunks)} (fallback)")
                    if "[MASK]" in chunk:
                        chunk_result = self._process_masked_chunk(chunk)
                        result += chunk_result
                    else:
                        result += chunk

                return result
            except Exception as fallback_error:
                self.logger.error(f"Fallback chunking failed: {str(fallback_error)}")
                # If all chunking attempts fail, return the original error
                raise e

    def _chunk_prompt(self, prompt: str, max_length: int) -> list:
        """
        Split a prompt into chunks that respect token limits

        Args:
            prompt (str): Input prompt
            max_length (int): Maximum token length per chunk

        Returns:
            list: List of prompt chunks
        """
        chunks = []
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Log token distribution
        self.logger.debug(f"Total tokens: {len(tokens)}, max_length: {max_length}")

        # Split tokens into chunks
        for i in range(0, len(tokens), max_length):
            chunk_tokens = tokens[i:i+max_length]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)

            # Log chunk information
            self.logger.debug(f"Chunk {len(chunks)}: {len(chunk_tokens)} tokens, {len(chunk)} chars")

        return chunks

    def _chunk_text_by_size(self, text: str, chunk_size: int) -> list:
        """
        Split text into chunks by character count (fallback method)

        Args:
            text (str): Input text
            chunk_size (int): Approximate characters per chunk

        Returns:
            list: List of text chunks
        """
        # Simple chunking by character count
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk)

        return chunks

    def process_field_mapping(self, standard_fields, input_columns, column_samples, structure_info=None):
        """
        Process field mapping with chunking to handle large inputs

        Args:
            standard_fields (dict): Standard field definitions
            input_columns (list): Input column names
            column_samples (dict): Sample data for each column
            structure_info (dict, optional): Structure analysis info

        Returns:
            dict: Mapping results
        """
        from prompts.field_mapping import get_field_mapping_prompt
        from utils.parsers.json_parser import JSONParser

        self.logger.info(f"Processing field mapping with chunking for {len(input_columns)} columns")

        # Initialize parser
        json_parser = JSONParser()

        # Estimate token count for the full input
        # This is a rough estimation to check if we need chunking
        total_token_estimate = 0
        for col in input_columns:
            # Add tokens for column name
            total_token_estimate += len(col.split()) + 1

            # Add tokens for sample data
            if col in column_samples:
                for sample in column_samples[col]:
                    if sample and isinstance(sample, str):
                        total_token_estimate += len(sample.split()) + 1

        self.logger.info(f"Estimated total tokens for all columns: {total_token_estimate}")

        # Get model's max token limit
        max_tokens = self.tokenizer.model_max_length if hasattr(self, 'tokenizer') else 512
        self.logger.info(f"Model max token limit: {max_tokens}")

        # If we have a small number of columns or estimated tokens are well below limit, process normally
        if len(input_columns) <= 5 and total_token_estimate < (max_tokens // 2):
            self.logger.info(f"Small input (columns: {len(input_columns)}, est. tokens: {total_token_estimate}), processing without chunking")
            try:
                prompt = get_field_mapping_prompt(standard_fields, input_columns, column_samples, structure_info or {})

                # Log the actual token count of the prompt
                if hasattr(self, 'tokenizer'):
                    actual_tokens = len(self.tokenizer.encode(prompt))
                    self.logger.info(f"Actual token count for prompt: {actual_tokens}/{max_tokens}")

                    if actual_tokens > max_tokens:
                        self.logger.warning(f"Prompt exceeds token limit ({actual_tokens} > {max_tokens}), falling back to chunking")
                        # Fall through to chunking logic
                    else:
                        response = self.generate_response(prompt)
                        result = json_parser.parse(response)
                        self.logger.info(f"Successfully processed without chunking")
                        return result
                else:
                    # If we can't check token count, just try it
                    response = self.generate_response(prompt)
                    result = json_parser.parse(response)
                    self.logger.info(f"Successfully processed without chunking")
                    return result
            except Exception as e:
                self.logger.error(f"Error processing without chunking: {str(e)}")
                self.logger.info(f"Falling back to chunking approach")
                # Fall through to chunking logic

        # For larger inputs, process in chunks
        self.logger.info(f"Large input with {len(input_columns)} columns, processing in chunks")

        # Use smaller chunks for very large inputs
        chunk_size = 5 if total_token_estimate > 1000 else 10
        self.logger.info(f"Using chunk size of {chunk_size} columns")

        # Split columns into chunks
        column_chunks = []
        for i in range(0, len(input_columns), chunk_size):
            chunk = input_columns[i:i+chunk_size]
            column_chunks.append(chunk)

        self.logger.info(f"Split into {len(column_chunks)} chunks")

        # Process each chunk
        all_mappings = {}
        chunk_errors = []

        for i, chunk in enumerate(column_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(column_chunks)} with {len(chunk)} columns")

            # Create chunk-specific samples
            chunk_samples = {col: column_samples.get(col, []) for col in chunk}

            # Generate prompt for this chunk
            chunk_prompt = get_field_mapping_prompt(standard_fields, chunk, chunk_samples, structure_info or {})

            # Log token count for this chunk
            if hasattr(self, 'tokenizer'):
                chunk_tokens = len(self.tokenizer.encode(chunk_prompt))
                self.logger.info(f"Chunk {i+1} token count: {chunk_tokens}/{max_tokens}")

                if chunk_tokens > max_tokens:
                    self.logger.warning(f"Chunk {i+1} exceeds token limit ({chunk_tokens} > {max_tokens})")
                    # Try with even smaller chunk or skip if necessary
                    if len(chunk) > 2:
                        self.logger.info(f"Splitting chunk {i+1} into smaller pieces")
                        # Process this chunk in even smaller pieces
                        sub_chunks = []
                        for j in range(0, len(chunk), 2):
                            sub_chunk = chunk[j:j+2]
                            sub_chunks.append(sub_chunk)

                        for j, sub_chunk in enumerate(sub_chunks):
                            self.logger.info(f"Processing sub-chunk {j+1}/{len(sub_chunks)} of chunk {i+1}")
                            sub_samples = {col: chunk_samples.get(col, []) for col in sub_chunk}
                            sub_prompt = get_field_mapping_prompt(standard_fields, sub_chunk, sub_samples, structure_info or {})

                            try:
                                sub_response = self.generate_response(sub_prompt)
                                sub_result = json_parser.parse(sub_response)

                                if isinstance(sub_result, dict) and 'field_mappings' in sub_result:
                                    all_mappings.update(sub_result['field_mappings'])
                                    self.logger.info(f"Added {len(sub_result['field_mappings'])} mappings from sub-chunk {j+1} of chunk {i+1}")
                            except Exception as e:
                                self.logger.error(f"Error processing sub-chunk {j+1} of chunk {i+1}: {str(e)}")
                                chunk_errors.append(f"Sub-chunk {j+1} of chunk {i+1}: {str(e)}")

                        # Skip the main chunk processing since we've handled it in sub-chunks
                        continue
                    else:
                        self.logger.warning(f"Chunk {i+1} is too small to split further, will attempt to process anyway")

            # Process chunk
            try:
                chunk_response = self.generate_response(chunk_prompt)
                chunk_result = json_parser.parse(chunk_response)

                # Extract mappings
                if isinstance(chunk_result, dict) and 'field_mappings' in chunk_result:
                    all_mappings.update(chunk_result['field_mappings'])
                    self.logger.info(f"Added {len(chunk_result['field_mappings'])} mappings from chunk {i+1}")
                elif isinstance(chunk_result, dict) and 'mappings' in chunk_result:
                    # Handle old format
                    self.logger.info(f"Converting old mappings format from chunk {i+1}")
                    all_mappings.update(chunk_result['mappings'])
                    self.logger.info(f"Added {len(chunk_result['mappings'])} mappings from chunk {i+1} (old format)")
                else:
                    self.logger.warning(f"Invalid response format from chunk {i+1}: {chunk_result}")
                    chunk_errors.append(f"Chunk {i+1}: Invalid response format")
            except Exception as e:
                self.logger.error(f"Error processing chunk {i+1}: {str(e)}")
                chunk_errors.append(f"Chunk {i+1}: {str(e)}")

        # Combine results
        result = {
            "field_mappings": all_mappings,
            "notes": f"Processed in {len(column_chunks)} chunks due to large input"
        }

        if chunk_errors:
            result["errors"] = chunk_errors
            result["notes"] += f" with {len(chunk_errors)} errors"

        self.logger.info(f"Completed field mapping with {len(all_mappings)} total mappings")

        # Convert to old format for backward compatibility if needed
        if not all_mappings:
            result["mappings"] = {}
            self.logger.warning("No mappings found, returning empty mappings dict for backward compatibility")

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

        # Check if the text is too long for the model
        try:
            if hasattr(self, 'tokenizer'):
                token_count = len(self.tokenizer.encode(text))
                max_length = self.tokenizer.model_max_length

                if token_count > max_length:
                    self.logger.warning(f"Text chunk exceeds token limit ({token_count} > {max_length}), truncating")
                    # Truncate the text to fit within token limits
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    truncated_tokens = tokens[:max_length-2]  # Leave room for special tokens
                    text = self.tokenizer.decode(truncated_tokens)

                    # Ensure we still have at least one mask token
                    if "[MASK]" not in text:
                        text += " [MASK]"

                    self.logger.info(f"Truncated text to {len(self.tokenizer.encode(text))} tokens")
        except Exception as e:
            self.logger.warning(f"Error checking token count: {str(e)}")
            # Continue with processing, the fill_mask_pipeline will handle any errors

        # Fill masks one by one
        for i in range(mask_count):
            if "[MASK]" not in text:
                break

            try:
                # Get predictions for the first mask
                self.logger.debug(f"Filling mask {i+1}/{mask_count}")
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
                self.logger.debug(f"Filled mask {i+1}, new text length: {len(text)}")

            except Exception as e:
                # Check if this is a token limit error
                error_str = str(e).lower()
                if 'token' in error_str and ('limit' in error_str or 'exceed' in error_str or 'length' in error_str):
                    self.logger.warning(f"Token limit error while filling mask: {str(e)}")

                    # Try to recover by truncating the text
                    try:
                        if hasattr(self, 'tokenizer'):
                            # Truncate more aggressively
                            tokens = self.tokenizer.encode(text, add_special_tokens=False)
                            # Use 75% of max length to be safe
                            safe_length = int(self.tokenizer.model_max_length * 0.75)
                            truncated_tokens = tokens[:safe_length]
                            text = self.tokenizer.decode(truncated_tokens)

                            # Ensure we still have at least one mask token
                            if "[MASK]" not in text:
                                text += " [MASK]"

                            self.logger.info(f"Recovered by truncating text to {len(self.tokenizer.encode(text))} tokens")
                            # Try again with the truncated text in the next iteration
                            continue
                    except Exception as truncate_error:
                        self.logger.error(f"Error during truncation recovery: {str(truncate_error)}")

                # For other errors, log and break
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

        # Check if prompt is likely to exceed token limits
        try:
            if hasattr(self, 'tokenizer'):
                # Estimate token count
                estimated_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token
                max_tokens = self.tokenizer.model_max_length

                # Log token count estimate
                self.logger.info(f"Estimated token count: {estimated_tokens}/{max_tokens}")

                # If prompt is likely to exceed token limits, use chunking directly
                if estimated_tokens > max_tokens:
                    self.logger.warning(f"Prompt likely exceeds token limit ({estimated_tokens} > {max_tokens})")
                    self.logger.info("Using chunking to handle large prompt")
                    # No need to check actual token count, _generate_text_with_masked_lm will handle chunking
        except Exception as e:
            self.logger.warning(f"Error estimating token count: {str(e)}")
            # Continue with processing, the generate method will handle any errors

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

            # Check if this is a token limit error
            error_str = str(e).lower()
            if 'token' in error_str and ('limit' in error_str or 'exceed' in error_str or 'length' in error_str):
                self.logger.warning("Token limit error detected, using fallback response")
                return self._generate_fallback_response(prompt)

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
        Generate a fallback response when rate limited or token limit exceeded

        Args:
            prompt (str): Input prompt

        Returns:
            str: Fallback response
        """
        self.logger.info("Generating fallback response")

        # Simple keyword-based response for common tasks
        lower_prompt = prompt.lower()

        # Determine if this is a token limit issue or rate limit issue
        is_token_limit = False

        # Check if we have a tokenizer and can check token count
        try:
            if hasattr(self, 'tokenizer'):
                estimated_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token
                max_tokens = self.tokenizer.model_max_length

                if estimated_tokens > max_tokens:
                    is_token_limit = True
                    self.logger.info(f"Fallback due to token limit: {estimated_tokens} > {max_tokens}")
        except Exception as e:
            self.logger.warning(f"Error checking token count for fallback: {str(e)}")

        # Prepare reason text based on the issue type
        reason = "Token limit exceeded" if is_token_limit else "Rate limit exceeded"

        # Field mapping fallback
        if "field mapping" in lower_prompt or "map fields" in lower_prompt:
            # Extract column names from prompt if possible
            columns = []
            try:
                import re
                # Look for column names in the prompt
                col_match = re.search(r'columns:\s*\[(.*?)\]', prompt, re.IGNORECASE | re.DOTALL)
                if col_match:
                    cols_str = col_match.group(1)
                    # Extract column names
                    cols = [c.strip().strip("'\"") for c in cols_str.split(',')]
                    columns = [c for c in cols if c]  # Filter out empty strings

                    self.logger.info(f"Extracted {len(columns)} columns from prompt for fallback mapping")
            except Exception as e:
                self.logger.warning(f"Error extracting columns from prompt: {str(e)}")

            # If we found columns, create a simple mapping
            if columns:
                mappings = {}
                for col in columns:
                    col_lower = col.lower()
                    # Simple heuristic mapping based on column name
                    if any(term in col_lower for term in ["sku", "id", "code", "product"]):
                        mappings["SKU"] = col
                    elif any(term in col_lower for term in ["desc", "name", "title"]):
                        mappings["Short Description"] = col
                    elif any(term in col_lower for term in ["manuf", "brand", "make", "vendor"]):
                        mappings["Manufacturer"] = col
                    elif any(term in col_lower for term in ["price", "cost", "msrp"]):
                        mappings["Trade Price"] = col

                # Create a JSON response with the mappings
                import json
                response = {
                    "field_mappings": mappings,
                    "notes": f"{reason}. Using fallback mapping based on column names.",
                    "fallback": True
                }
                return json.dumps(response)

            # Default field mapping fallback
            return f'{{"field_mappings": {{}}, "notes": "{reason}. Please try again with fewer columns or a smaller input.", "fallback": true}}'

        # Structure analysis fallback
        if "structure analysis" in lower_prompt or "analyze structure" in lower_prompt:
            return f'{{"structure_notes": "{reason}. Using fallback analysis.", "column_analysis": {{}}, "possible_field_mappings": {{}}, "data_quality_issues": [], "fallback": true}}'

        # Category fallback
        if "category" in lower_prompt or "categorize" in lower_prompt:
            return f'{{"category": "Unknown", "reason": "{reason}", "fallback": true}}'

        # Generic fallback
        return f"{reason}. Please try again with a smaller input or fewer columns."

    def cleanup(self):
        """Release resources used by the model"""
        super().cleanup()

        # Additional cleanup specific to DistilBERT
        if hasattr(self, 'fill_mask_pipeline'):
            del self.fill_mask_pipeline
            self.fill_mask_pipeline = None
