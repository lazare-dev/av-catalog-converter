"""
Integration tests for the API with optimized components
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPIWithOptimizations:
    """Integration tests for the API with optimized components"""

    def test_health_check_with_llm_stats(self, client, mock_phi_client):
        """Test the health check endpoint with LLM stats"""
        # Mock the LLM factory to return our mock client
        with patch('core.llm.llm_factory.LLMFactory.create_client', return_value=mock_phi_client):
            with patch('core.llm.llm_factory.LLMFactory.get_stats') as mock_get_stats:
                # Setup mock stats
                mock_get_stats.return_value = {
                    'clients_count': 1,
                    'errors_count': 0,
                    'init_time': 0.5,
                    'available_models': ['phi'],
                    'clients': {
                        'PhiClient': {
                            'model_id': 'microsoft/phi-2',
                            'is_initialized': True,
                            'total_generations': 10,
                            'cache_hits': 5,
                            'rate_limited_count': 2
                        }
                    }
                }
                
                # Call the health check endpoint
                response = client.get('/api/health')
                
                # Check that the response is OK
                assert response.status_code == 200
                
                # Check that the response contains LLM information
                data = json.loads(response.data)
                assert 'llm' in data
                assert 'model_id' in data['llm']
                assert data['llm']['model_id'] == 'microsoft/phi-2'
                
                # Check that the response contains cache and rate limiting information
                assert 'cache_hits' in data['llm']
                assert 'rate_limited_count' in data['llm']

    def test_upload_file_with_parallel_processing(self, client, small_csv_file):
        """Test uploading a file with parallel processing"""
        # Mock the parallel processor
        with patch('utils.parallel.parallel_processor.ParallelProcessor') as mock_processor_class:
            # Setup the mock
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock the process_file function to use our fixture
            with patch('app.process_file') as mock_process_file:
                # Setup mock return value
                import pandas as pd
                df = pd.read_csv(small_csv_file)
                mock_process_file.return_value = (df, None)
                
                # Upload the file
                with open(small_csv_file, 'rb') as f:
                    response = client.post(
                        '/api/upload',
                        data={'file': (f, 'test.csv')},
                        content_type='multipart/form-data'
                    )
                
                # Check that the response is OK
                assert response.status_code == 200
                
                # Check that the response contains the expected data
                data = json.loads(response.data)
                assert 'success' in data
                assert data['success'] is True
                assert 'rows' in data
                assert data['rows'] == len(df)

    def test_analyze_file_with_optimized_components(self, client, small_csv_file):
        """Test analyzing a file with optimized components"""
        # Mock the structure analyzer
        with patch('services.structure.structure_analyzer.StructureAnalyzer.analyze') as mock_analyze:
            # Setup mock return value
            mock_analyze.return_value = {
                'row_count': 10,
                'column_count': 5,
                'column_types': {
                    'SKU': {'type': 'id', 'sample': 'ABC123'},
                    'Product Name': {'type': 'string', 'sample': 'HD Camera'},
                    'Price': {'type': 'price', 'sample': 299.99},
                    'Category': {'type': 'category', 'sample': 'Video'},
                    'Manufacturer': {'type': 'string', 'sample': 'Sony'}
                },
                'processing_time': 0.05,
                'parallel_processing': True,
                'cache_stats': {
                    'hits': 0,
                    'misses': 1,
                    'hit_ratio': 0.0
                }
            }
            
            # Upload the file
            with open(small_csv_file, 'rb') as f:
                response = client.post(
                    '/api/analyze',
                    data={'file': (f, 'test.csv')},
                    content_type='multipart/form-data'
                )
            
            # Check that the response is OK
            assert response.status_code == 200
            
            # Check that the response contains the expected data
            data = json.loads(response.data)
            assert 'success' in data
            assert data['success'] is True
            assert 'analysis' in data
            assert 'row_count' in data['analysis']
            assert 'column_count' in data['analysis']
            assert 'column_types' in data['analysis']
            
            # Check that the response contains optimization information
            assert 'processing_time' in data['analysis']
            assert 'parallel_processing' in data['analysis']
            assert 'cache_stats' in data['analysis']

    def test_map_fields_with_rate_limiting(self, client, small_csv_file, mock_phi_client):
        """Test mapping fields with rate limiting"""
        # Mock the LLM factory to return our mock client
        with patch('core.llm.llm_factory.LLMFactory.create_client', return_value=mock_phi_client):
            # Mock the semantic mapper
            with patch('services.mapping.semantic_mapper.SemanticMapper.map_fields') as mock_map_fields:
                # Setup mock return value
                mock_map_fields.return_value = {
                    'SKU': 'SKU',
                    'Short Description': 'Product Name',
                    'Long Description': 'Description',
                    'Category': 'Category',
                    'Manufacturer': 'Manufacturer',
                    'Trade Price': 'Price'
                }
                
                # Upload the file
                with open(small_csv_file, 'rb') as f:
                    response = client.post(
                        '/api/map',
                        data={
                            'file': (f, 'test.csv'),
                            'mapping_type': 'semantic'
                        },
                        content_type='multipart/form-data'
                    )
                
                # Check that the response is OK
                assert response.status_code == 200
                
                # Check that the response contains the expected data
                data = json.loads(response.data)
                assert 'success' in data
                assert data['success'] is True
                assert 'mapping' in data
                assert 'SKU' in data['mapping']
                assert 'Short Description' in data['mapping']
                assert 'Trade Price' in data['mapping']
                
                # Check that rate limiting was applied
                assert mock_phi_client.generate_response.called
