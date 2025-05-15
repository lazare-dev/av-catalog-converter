"""
Unit tests for the LLM Factory
"""
import pytest
from unittest.mock import patch, MagicMock

from core.llm.llm_factory import LLMFactory


class TestLLMFactory:
    """Tests for the LLMFactory class"""

    def setup_method(self):
        """Setup for each test"""
        # Clear the client cache before each test
        LLMFactory._clients = {}
        LLMFactory._init_errors = {}

    def test_get_model_type_explicit(self):
        """Test _get_model_type with explicit model type"""
        model_config = {"model_type": "distilbert"}
        assert LLMFactory._get_model_type(model_config) == "distilbert"

    def test_get_model_type_from_id(self):
        """Test _get_model_type inferred from model_id"""
        assert LLMFactory._get_model_type({"model_id": "gpt2"}) == "gpt2"
        assert LLMFactory._get_model_type({"model_id": "distilbert-base-uncased"}) == "distilbert"

    def test_get_model_type_default(self):
        """Test _get_model_type default"""
        assert LLMFactory._get_model_type({}) == "distilbert"
        assert LLMFactory._get_model_type({"model_id": "unknown-model"}) == "distilbert"

    def test_get_available_models(self):
        """Test get_available_models"""
        models = LLMFactory.get_available_models()
        assert "distilbert" in models
        assert "gpt2" in models
        assert "dummy" in models
