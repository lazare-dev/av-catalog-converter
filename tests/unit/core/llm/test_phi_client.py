"""
Unit tests for the PhiClient with rate limiting and adaptive caching
"""
import pytest
import time
import json
from unittest.mock import patch, MagicMock, ANY

from core.llm.phi_client import PhiClient
from utils.caching.adaptive_cache import AdaptiveCache
from utils.rate_limiting.rate_limiter import RateLimiter


class TestPhiClient:
    """Tests for the PhiClient class with rate limiting and adaptive caching"""

    def test_init_with_adaptive_cache(self):
        """Test initialization with adaptive cache"""
        # Create a client with adaptive cache
        model_config = {
            "model_id": "microsoft/phi-2",
            "cache_enabled": True,
            "cache_type": "adaptive",
            "cache_ttl": 7200,
            "cache_max_size": 500
        }
        
        client = PhiClient(model_config)
        
        # Check that cache was initialized correctly
        assert client.cache is not None
        assert isinstance(client.cache, AdaptiveCache)
        assert client.cache.base_ttl == 7200
        assert client.cache.max_size == 500

    def test_init_with_standard_cache(self):
        """Test initialization with standard memory cache"""
        # Create a client with standard cache
        model_config = {
            "model_id": "microsoft/phi-2",
            "cache_enabled": True,
            "cache_type": "memory",
            "cache_ttl": 7200
        }
        
        with patch("utils.caching.memory_cache.MemoryCache") as mock_cache:
            client = PhiClient(model_config)
            
            # Check that cache was initialized correctly
            assert client.cache is not None
            mock_cache.assert_called_once_with(ttl=7200)

    def test_init_with_rate_limiting(self):
        """Test initialization with rate limiting"""
        # Create a client with rate limiting
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": True,
            "requests_per_minute": 30,
            "burst_size": 5
        }
        
        client = PhiClient(model_config)
        
        # Check that rate limiter was initialized correctly
        assert client.rate_limiter is not None
        assert isinstance(client.rate_limiter, RateLimiter)
        assert client.rate_limiter.bucket.tokens_per_second == 0.5  # 30/60
        assert client.rate_limiter.bucket.max_tokens == 5

    def test_init_without_rate_limiting(self):
        """Test initialization without rate limiting"""
        # Create a client without rate limiting
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": False
        }
        
        client = PhiClient(model_config)
        
        # Check that rate limiter was not initialized
        assert client.rate_limiter is None

    @patch("torch.inference_mode")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_response_with_cache_hit(self, mock_tokenizer, mock_model, mock_inference_mode):
        """Test generate_response with a cache hit"""
        # Create a client with cache
        model_config = {
            "model_id": "microsoft/phi-2",
            "cache_enabled": True
        }
        
        client = PhiClient(model_config)
        
        # Mock the cache
        client.cache = MagicMock()
        client.cache.get.return_value = "Cached response"
        
        # Generate a response
        response = client.generate_response("Test prompt")
        
        # Check that the cache was used
        client.cache.get.assert_called_once_with("Test prompt")
        assert response == "Cached response"
        assert client.cache_hits == 1
        
        # Check that the model was not used
        mock_tokenizer.assert_not_called()
        mock_model.assert_not_called()

    @patch("torch.inference_mode")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_response_with_rate_limiting(self, mock_tokenizer, mock_model, mock_inference_mode):
        """Test generate_response with rate limiting"""
        # Create a client with rate limiting
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": True,
            "requests_per_minute": 60,
            "burst_size": 10
        }
        
        client = PhiClient(model_config)
        client._is_initialized = True
        
        # Mock the rate limiter
        client.rate_limiter = MagicMock()
        client.rate_limiter.bucket.consume.return_value = True
        
        # Mock the tokenizer and model
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_tensors = "pt"
        mock_tokenizer.return_value.eos_token_id = 0
        
        mock_model.return_value = MagicMock()
        mock_model.return_value.generate.return_value = [MagicMock()]
        
        # Generate a response
        client.generate_response("Test prompt")
        
        # Check that rate limiting was applied
        client.rate_limiter.bucket.consume.assert_called_once()

    @patch("torch.inference_mode")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_response_with_rate_limiting_exceeded(self, mock_tokenizer, mock_model, mock_inference_mode):
        """Test generate_response with rate limiting exceeded"""
        # Create a client with rate limiting
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": True
        }
        
        client = PhiClient(model_config)
        client._is_initialized = True
        
        # Mock the rate limiter to reject the request
        client.rate_limiter = MagicMock()
        client.rate_limiter.bucket.consume.return_value = False
        
        # Generate a response
        response = client.generate_response("Test prompt")
        
        # Check that an error response was returned
        assert "Rate limit exceeded" in response
        assert client.rate_limited_count == 1

    def test_get_stats_with_rate_limiter(self):
        """Test get_stats with rate limiter"""
        # Create a client with rate limiting
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": True
        }
        
        client = PhiClient(model_config)
        
        # Mock the rate limiter stats
        client.rate_limiter.get_stats = MagicMock(return_value={
            "total_requests": 10,
            "limited_requests": 2,
            "wait_time": 1.5
        })
        
        # Get stats
        stats = client.get_stats()
        
        # Check that rate limiter stats are included
        assert "rate_limiter" in stats
        assert stats["rate_limiter"]["total_requests"] == 10
        assert stats["rate_limiter"]["limited_requests"] == 2
        assert stats["rate_limiter"]["wait_time"] == 1.5
        
        # Check that rate limited count is included
        assert "rate_limited_count" in stats
        assert "rate_limited_wait_time" in stats

    @patch("torch.inference_mode")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_batch_generate(self, mock_tokenizer, mock_model, mock_inference_mode):
        """Test batch_generate with rate limiting and caching"""
        # Create a client with rate limiting and caching
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": True,
            "cache_enabled": True
        }
        
        client = PhiClient(model_config)
        client._is_initialized = True
        
        # Mock generate_response to return predictable responses
        client.generate_response = MagicMock(side_effect=lambda p: f"Response to {p}")
        
        # Generate batch responses
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = client.batch_generate(prompts)
        
        # Check that generate_response was called for each prompt
        assert client.generate_response.call_count == 3
        
        # Check the responses
        assert responses == ["Response to Prompt 1", "Response to Prompt 2", "Response to Prompt 3"]
