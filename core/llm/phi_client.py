# core/llm/phi_client.py
"""
Phi-specific LLM implementation
"""
import logging
import os
import tempfile
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from core.llm.base_client import BaseLLMClient
from utils.caching.memory_cache import MemoryCache

class PhiClient(BaseLLMClient):
    """Client for Microsoft's Phi models"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the Phi client
        
        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.tokenizer = None
        
        # Setup cache if enabled
        self.cache = None
        if model_config.get("cache_enabled", False):
            self.cache = MemoryCache()
            self.logger.info("LLM response caching enabled")
            
    def initialize_model(self):
        """
        Initialize and load the Phi model
        """
        if self.model is not None:
            return
            
        self.logger.info(f"Initializing Phi model: {self.model_config['model_id']}")
        
        # Configure quantization if specified
        quantization = self.model_config.get("quantization")
        quantization_config = None
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.logger.info("Using 4-bit quantization")
            
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.logger.info("Using 8-bit quantization")
            
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["model_id"],
                use_fast=True
            )
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            raise
            
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["model_id"],
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        self.logger.info(f"Phi model initialized successfully")
        
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the Phi model
        
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
                return cached_response
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
            
        # Format prompt for Phi (simpler than Gemma)
        formatted_prompt = f"Instruction: {prompt}\n\nResponse:"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.model_config.get("max_new_tokens", 512),
                    temperature=self.model_config.get("temperature", 0.3),
                    top_p=self.model_config.get("top_p", 0.95),
                    repetition_penalty=1.1,
                    do_sample=self.model_config.get("temperature", 0.3) > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generation_time = time.time() - start_time
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part (after the prompt)
            response = generated_text.replace(formatted_prompt, "").strip()
            
            self.logger.debug(f"Generated response in {generation_time:.2f}s")
            
            # Cache the response if enabled
            if self.cache is not None:
                self.cache.set(prompt, response)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
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

    def cleanup(self):
        """
        Release resources used by the Phi model
        """
        super().cleanup()
        
        # Additional Phi-specific cleanup if needed
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
