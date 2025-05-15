# core/llm/dummy_client.py
"""
Dummy LLM client for fallback when real LLM clients fail to initialize
This provides basic functionality without requiring any ML models
"""
import logging
import time
from typing import Dict, Any, List

from core.llm.base_client import BaseLLMClient
from utils.logging.logger import Logger

class DummyLLMClient(BaseLLMClient):
    """Dummy LLM client that provides basic responses without ML models"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the dummy client
        
        Args:
            model_config (Dict[str, Any]): Model configuration
        """
        super().__init__(model_config)
        self.logger.warning("Using DummyLLMClient - this provides limited functionality")
        self._is_initialized = True
        self.request_count = 0
        
    def initialize_model(self):
        """No-op for dummy client"""
        self.logger.info("DummyLLMClient does not need initialization")
        self._is_initialized = True
        
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response based on simple pattern matching
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Simple response based on pattern matching
        """
        self.request_count += 1
        self.logger.info(f"DummyLLMClient generating response for prompt: {prompt[:50]}...")
        
        # Log the request
        self.logger.debug(f"DummyLLMClient request #{self.request_count}",
                        prompt_length=len(prompt),
                        prompt_preview=prompt[:100])
        
        # Simple pattern matching for common tasks
        lower_prompt = prompt.lower()
        
        # Field mapping
        if "field mapping" in lower_prompt or "map fields" in lower_prompt:
            return self._generate_field_mapping_response(prompt)
            
        # Category extraction
        if "category" in lower_prompt or "categorize" in lower_prompt:
            return self._generate_category_response(prompt)
            
        # Structure analysis
        if "structure" in lower_prompt or "analyze" in lower_prompt:
            return self._generate_structure_response(prompt)
            
        # Generic response
        return "DummyLLMClient cannot provide a detailed response. Please use a real LLM client for better results."
        
    def _generate_field_mapping_response(self, prompt: str) -> str:
        """Generate a response for field mapping prompts"""
        # Extract potential column names from the prompt
        words = prompt.split()
        potential_columns = [w for w in words if len(w) > 3 and w.isalnum()]
        
        # Map to standard fields based on simple string matching
        mappings = {}
        
        field_patterns = {
            "sku": ["sku", "item", "product", "code", "number", "id"],
            "short_description": ["name", "title", "description", "desc", "product"],
            "long_description": ["description", "desc", "details", "specs", "specification"],
            "model": ["model", "type", "version"],
            "category_group": ["category", "group", "department", "section"],
            "category": ["category", "type", "class"],
            "manufacturer": ["manufacturer", "brand", "vendor", "make", "company"],
            "manufacturer_sku": ["mfr", "manufacturer", "vendor", "part", "number"],
            "price": ["price", "cost", "msrp", "rrp", "retail", "wholesale"]
        }
        
        for column in potential_columns:
            col_lower = column.lower()
            for field, patterns in field_patterns.items():
                for pattern in patterns:
                    if pattern in col_lower:
                        mappings[column] = {"field": field.replace("_", " ").title(), "confidence": 0.7}
                        break
        
        # Return JSON-like string
        if mappings:
            result = "{\n  \"mappings\": {\n"
            for col, mapping in mappings.items():
                result += f"    \"{col}\": {{\n      \"field\": \"{mapping['field']}\",\n      \"confidence\": {mapping['confidence']}\n    }},\n"
            result = result.rstrip(",\n") + "\n  }\n}"
            return result
        else:
            return "{\"mappings\": {}}"
            
    def _generate_category_response(self, prompt: str) -> str:
        """Generate a response for category extraction prompts"""
        lower_prompt = prompt.lower()
        
        # Simple category matching
        categories = {
            "display": ["tv", "television", "monitor", "display", "projector", "screen"],
            "audio": ["speaker", "headphone", "earphone", "microphone", "amplifier", "receiver"],
            "video": ["camera", "camcorder", "recorder", "player"],
            "control": ["remote", "control", "automation", "switch", "panel"],
            "cable": ["cable", "wire", "connector", "adapter", "hdmi", "usb"]
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in lower_prompt:
                    return f"{{\"category\": \"{category.title()}\", \"confidence\": 0.8}}"
        
        return "{\"category\": \"Unknown\", \"confidence\": 0.5}"
        
    def _generate_structure_response(self, prompt: str) -> str:
        """Generate a response for structure analysis prompts"""
        return "{\"structure_notes\": \"DummyLLMClient cannot perform detailed structure analysis. Please use a real LLM client for better results.\"}"
        
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts (List[str]): List of input prompts
            
        Returns:
            List[str]: List of responses
        """
        return [self.generate_response(prompt) for prompt in prompts]
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dict[str, Any]: Model metadata
        """
        info = super().get_model_info()
        info.update({
            "model_type": "dummy",
            "is_initialized": True,
            "request_count": self.request_count,
            "warning": "This is a dummy LLM client with limited functionality"
        })
        return info
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the client's performance
        
        Returns:
            Dict[str, Any]: Statistics
        """
        return {
            "model_type": "dummy",
            "is_initialized": True,
            "request_count": self.request_count,
            "warning": "This is a dummy LLM client with limited functionality"
        }
        
    def cleanup(self):
        """No-op for dummy client"""
        self.logger.info("DummyLLMClient cleanup - nothing to do")
