"""
Integration tests for the optimized components
"""
import pytest
import pandas as pd
import numpy as np
import os
import time
from unittest.mock import patch, MagicMock

from core.file_parser.csv_parser import CSVParser
from core.file_parser.excel_parser import ExcelParser
from core.llm.phi_client import PhiClient
from core.llm.llm_factory import LLMFactory
from utils.caching.adaptive_cache import AdaptiveCache
from utils.rate_limiting.rate_limiter import RateLimiter
from utils.parallel.parallel_processor import ParallelProcessor


class TestOptimizedComponents:
    """Integration tests for the optimized components"""

    def test_csv_parser_with_parallel_processing(self, tmp_path):
        """Test CSV parser with parallel processing"""
        # Create a large CSV file
        file_path = tmp_path / "large.csv"
        
        # Create a DataFrame with 5,000 rows
        df = pd.DataFrame({
            'SKU': [f"SKU{i}" for i in range(5000)],
            'Product Name': [f"Product {i}" for i in range(5000)],
            'Price': np.random.random(size=5000) * 1000,
            'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=5000),
            'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=5000)
        })
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        # Make the file appear large
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 60 * 1024 * 1024
            
            # Create parser
            parser = CSVParser(file_path)
            
            # Parse the file
            start_time = time.time()
            result = parser.parse()
            parse_time = time.time() - start_time
            
            # Check that the result is correct
            assert list(result.columns) == list(df.columns)
            assert len(result) == len(df)
            
            # Log the parsing time
            print(f"Parsed 5,000 rows in {parse_time:.2f} seconds")

    def test_excel_parser_with_parallel_processing(self, tmp_path):
        """Test Excel parser with parallel processing"""
        # Create a large Excel file
        file_path = tmp_path / "large.xlsx"
        
        # Create a DataFrame with 1,000 rows
        df = pd.DataFrame({
            'SKU': [f"SKU{i}" for i in range(1000)],
            'Product Name': [f"Product {i}" for i in range(1000)],
            'Price': np.random.random(size=1000) * 1000,
            'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=1000),
            'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=1000)
        })
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        
        # Make the file appear large
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 30 * 1024 * 1024
            
            # Create parser
            parser = ExcelParser(file_path)
            
            # Parse the file
            start_time = time.time()
            result = parser.parse()
            parse_time = time.time() - start_time
            
            # Check that the result is correct
            assert list(result.columns) == list(df.columns)
            assert len(result) == len(df)
            
            # Log the parsing time
            print(f"Parsed 1,000 rows in {parse_time:.2f} seconds")

    def test_phi_client_with_adaptive_cache(self):
        """Test PhiClient with adaptive cache"""
        # Create a client with adaptive cache
        model_config = {
            "model_id": "microsoft/phi-2",
            "cache_enabled": True,
            "cache_type": "adaptive",
            "cache_ttl": 3600,
            "cache_max_size": 100
        }
        
        # Mock the model initialization
        with patch.object(PhiClient, 'initialize_model'):
            client = PhiClient(model_config)
            client._is_initialized = True
            
            # Mock the generate_response method
            with patch.object(client, 'generate_response', side_effect=lambda p: f"Response to {p}"):
                # Test the cache
                for i in range(5):
                    # Make repeated requests to the same prompts
                    for prompt in ["Prompt A", "Prompt B", "Prompt C"]:
                        response = client.generate_response(prompt)
                        assert response == f"Response to {prompt}"
                
                # Check cache stats
                stats = client.get_stats()
                assert stats['cache_hits'] > 0
                
                # Check adaptive cache stats
                assert isinstance(client.cache, AdaptiveCache)
                cache_stats = client.cache.get_stats()
                assert cache_stats['hits'] > 0
                assert cache_stats['size'] <= 3  # Should have at most 3 entries

    def test_phi_client_with_rate_limiting(self):
        """Test PhiClient with rate limiting"""
        # Create a client with rate limiting
        model_config = {
            "model_id": "microsoft/phi-2",
            "rate_limiting_enabled": True,
            "requests_per_minute": 60,
            "burst_size": 5
        }
        
        # Mock the model initialization
        with patch.object(PhiClient, 'initialize_model'):
            client = PhiClient(model_config)
            client._is_initialized = True
            
            # Mock the actual response generation to avoid calling the model
            with patch.object(client, '_format_prompt', return_value="Formatted prompt"):
                with patch('torch.inference_mode'):
                    with patch.object(client, '_extract_response', return_value="Test response"):
                        # Create a mock tokenizer and model
                        client.tokenizer = MagicMock()
                        client.tokenizer.return_value = {"input_ids": MagicMock()}
                        client.tokenizer.decode.return_value = "Generated text"
                        
                        client.model = MagicMock()
                        client.model.generate.return_value = [MagicMock()]
                        
                        # Make a burst of requests
                        start_time = time.time()
                        for i in range(10):
                            response = client.generate_response(f"Prompt {i}")
                        burst_time = time.time() - start_time
                        
                        # Check rate limiter stats
                        assert client.rate_limiter is not None
                        stats = client.get_stats()
                        
                        # Should have rate limited some requests
                        assert stats['rate_limited_count'] > 0
                        
                        # Log the burst time
                        print(f"Processed 10 requests in {burst_time:.2f} seconds with rate limiting")

    def test_llm_factory_with_optimized_client(self):
        """Test LLMFactory with optimized client"""
        # Mock the PhiClient
        with patch('core.llm.phi_client.PhiClient') as mock_phi_client:
            # Setup the mock
            mock_client = MagicMock()
            mock_phi_client.return_value = mock_client
            
            # Create a client through the factory
            client = LLMFactory.create_client()
            
            # Check that the client was created with the right configuration
            mock_phi_client.assert_called_once()
            
            # Check that the factory recorded initialization time
            assert hasattr(LLMFactory, '_init_time')
            
            # Get factory stats
            stats = LLMFactory.get_stats()
            assert 'init_time' in stats
            assert 'clients_count' in stats
            assert 'clients' in stats

    def test_parallel_processor_with_large_dataframe(self):
        """Test ParallelProcessor with a large DataFrame"""
        # Create a large DataFrame
        df = pd.DataFrame({
            'A': range(10000),
            'B': range(10000, 20000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=10000)
        })
        
        # Define a processing function
        def process_func(chunk):
            # Simulate some processing
            time.sleep(0.001)
            return chunk.assign(D=chunk['A'] + chunk['B'])
        
        # Process with a single thread
        start_time = time.time()
        result_single = process_func(df)
        single_time = time.time() - start_time
        
        # Process with parallel processing
        processor = ParallelProcessor(max_workers=4)
        start_time = time.time()
        result_parallel = processor.process_dataframe(df, process_func, chunk_size=1000)
        parallel_time = time.time() - start_time
        
        # Check that the results are the same
        pd.testing.assert_frame_equal(result_single, result_parallel)
        
        # Log the times
        print(f"Single-threaded: {single_time:.2f}s, Parallel: {parallel_time:.2f}s")
        
        # Parallel should be faster
        assert parallel_time < single_time

    def test_end_to_end_with_all_optimizations(self, tmp_path):
        """Test end-to-end with all optimizations"""
        # Create a CSV file
        file_path = tmp_path / "test.csv"
        
        # Create a DataFrame
        df = pd.DataFrame({
            'SKU': [f"SKU{i}" for i in range(1000)],
            'Product Name': [f"Product {i}" for i in range(1000)],
            'Price': np.random.random(size=1000) * 1000,
            'Category': np.random.choice(['Audio', 'Video', 'Lighting'], size=1000),
            'Manufacturer': np.random.choice(['Sony', 'Panasonic', 'JBL'], size=1000)
        })
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        # Make the file appear large
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 60 * 1024 * 1024
            
            # Create parser
            parser = CSVParser(file_path)
            
            # Parse the file
            parsed_df = parser.parse()
            
            # Mock the LLM client
            with patch('core.llm.llm_factory.LLMFactory.create_client') as mock_create_client:
                mock_client = MagicMock()
                mock_create_client.return_value = mock_client
                mock_client.generate_response.return_value = json.dumps({
                    "SKU": "sku",
                    "Short Description": "product_name",
                    "Long Description": "",
                    "Model": "",
                    "Category Group": "",
                    "Category": "category",
                    "Manufacturer": "manufacturer",
                    "Manufacturer SKU": "",
                    "Image URL": "",
                    "Document Name": "",
                    "Document URL": "",
                    "Unit Of Measure": "",
                    "Buy Cost": "",
                    "Trade Price": "price",
                    "MSRP GBP": "",
                    "MSRP USD": "",
                    "MSRP EUR": "",
                    "Discontinued": ""
                })
                
                # Process the file through the app
                from app import process_file
                
                # Process the file
                start_time = time.time()
                result, error = process_file(file_path)
                process_time = time.time() - start_time
                
                # Check that there is no error
                assert error is None
                
                # Check that the result is a DataFrame
                assert isinstance(result, pd.DataFrame)
                
                # Check that the DataFrame has the expected columns
                assert 'SKU' in result.columns
                assert 'Short Description' in result.columns
                assert 'Trade Price' in result.columns
                assert 'Category' in result.columns
                assert 'Manufacturer' in result.columns
                
                # Log the processing time
                print(f"Processed file end-to-end in {process_time:.2f} seconds")
