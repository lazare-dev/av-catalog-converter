"""
Unit tests for the imports of the new modules
"""
import pytest
import importlib


class TestImports:
    """Tests for the imports of the new modules"""

    def test_parallel_imports(self):
        """Test importing the parallel module"""
        # Import the module
        parallel = importlib.import_module('utils.parallel')

        # Check that the expected classes are exported
        assert hasattr(parallel, 'ParallelProcessor')

        # Import the class directly
        from utils.parallel import ParallelProcessor

        # Check that the class has the expected methods
        assert hasattr(ParallelProcessor, 'process_dataframe')
        assert hasattr(ParallelProcessor, 'process_file')
        assert hasattr(ParallelProcessor, '_process_chunks')

    def test_rate_limiting_imports(self):
        """Test importing the rate_limiting module"""
        # Import the module
        rate_limiting = importlib.import_module('utils.rate_limiting')

        # Check that the expected classes are exported
        assert hasattr(rate_limiting, 'RateLimiter')
        assert hasattr(rate_limiting, 'TokenBucket')

        # Import the classes directly
        from utils.rate_limiting import RateLimiter, TokenBucket

        # Check that the classes have the expected methods
        assert hasattr(RateLimiter, 'limit')
        assert hasattr(RateLimiter, 'get_stats')
        assert hasattr(TokenBucket, 'consume')
        assert hasattr(TokenBucket, 'get_status')

    def test_caching_imports(self):
        """Test importing the caching module"""
        # Import the module
        caching = importlib.import_module('utils.caching')

        # Check that the expected classes are exported
        assert hasattr(caching, 'MemoryCache')
        assert hasattr(caching, 'DiskCache')
        assert hasattr(caching, 'AdaptiveCache')

        # Import the classes directly
        from utils.caching import MemoryCache, DiskCache, AdaptiveCache

        # Check that the classes have the expected methods
        assert hasattr(MemoryCache, 'get')
        assert hasattr(MemoryCache, 'set')
        assert hasattr(DiskCache, 'get')
        assert hasattr(DiskCache, 'set')
        assert hasattr(AdaptiveCache, 'get')
        assert hasattr(AdaptiveCache, 'set')
        assert hasattr(AdaptiveCache, 'get_stats')

    def test_phi_client_imports(self):
        """Test importing the PhiClient"""
        # Import the class
        from core.llm.phi_client import PhiClient

        # Check that the class has the expected methods
        assert hasattr(PhiClient, 'generate_response')
        assert hasattr(PhiClient, 'batch_generate')
        assert hasattr(PhiClient, 'get_stats')

        # Create an instance to check instance attributes
        # We can't check class attributes since they're set in __init__
        instance = PhiClient({"model_id": "test"})
        assert hasattr(instance, 'rate_limiter')
        assert hasattr(instance, 'cache')


    def test_gpt_client_imports(self):
        """Test importing the GPTClient"""
        # Import the class
        from core.llm.gpt_client import GPTClient

        # Check that the class has the expected methods
        assert hasattr(GPTClient, 'generate_response')
        assert hasattr(GPTClient, 'batch_generate')
        assert hasattr(GPTClient, 'get_stats')

        # Create an instance to check instance attributes
        # We can't check class attributes since they're set in __init__
        instance = GPTClient({"model_id": "test"})
        assert hasattr(instance, 'rate_limiter')
        assert hasattr(instance, 'cache')

    def test_llm_factory_imports(self):
        """Test importing the LLMFactory"""
        # Import the class
        from core.llm.llm_factory import LLMFactory

        # Check that the class has the expected methods
        assert hasattr(LLMFactory, 'create_client')
        assert hasattr(LLMFactory, 'get_stats')
        assert hasattr(LLMFactory, 'cleanup_clients')

        # Check that the class has the new attributes
        assert hasattr(LLMFactory, '_init_time')

        # Check that the CLIENT_MAP includes GPT-2, DistilBERT, and PHI
        assert "gpt2" in LLMFactory.CLIENT_MAP
        assert "distilbert" in LLMFactory.CLIENT_MAP
        assert "phi" in LLMFactory.CLIENT_MAP

    def test_csv_parser_imports(self):
        """Test importing the CSVParser"""
        # Import the class
        from core.file_parser.csv_parser import CSVParser

        # Check that the class has the expected methods
        assert hasattr(CSVParser, 'parse')
        assert hasattr(CSVParser, 'get_headers')
        assert hasattr(CSVParser, 'get_sample')

    def test_excel_parser_imports(self):
        """Test importing the ExcelParser"""
        # Import the class
        from core.file_parser.excel_parser import ExcelParser

        # Check that the class has the expected methods
        assert hasattr(ExcelParser, 'parse')
        assert hasattr(ExcelParser, 'get_headers')
        assert hasattr(ExcelParser, 'get_sample')

        # Check that the class has the new methods
        assert hasattr(ExcelParser, '_can_partition_sheet')
        assert hasattr(ExcelParser, '_estimate_row_count')
