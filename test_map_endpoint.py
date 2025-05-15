#!/usr/bin/env python3
"""
Test script for the /api/map endpoint
"""
import os
import sys
import json
from flask import Flask
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the app
from app import app

def test_map_endpoint():
    """Test the /api/map endpoint with a mock file and mock SemanticMapper"""
    # Create a test client
    client = app.test_client()
    
    # Create a mock file
    test_file_path = 'test_data.csv'
    with open(test_file_path, 'w') as f:
        f.write('SKU,Product Name,Price,Category,Manufacturer\n')
        f.write('ABC123,HD Camera,299.99,Video,Sony\n')
        f.write('DEF456,Wireless Mic,149.50,Audio,Shure\n')
        f.write('GHI789,Audio Mixer,499.00,Audio,Yamaha\n')
    
    # Create a mock SemanticMapper
    mock_map_fields = MagicMock(return_value={
        'SKU': 'SKU',
        'Short Description': 'Product Name',
        'Long Description': 'Description',
        'Category': 'Category',
        'Manufacturer': 'Manufacturer',
        'Trade Price': 'Price'
    })
    
    # Mock the LLM factory to return a mock client
    mock_client = MagicMock()
    mock_client.generate_response.side_effect = lambda p: f"Response to {p}"
    mock_client.get_stats.return_value = {
        'total_generations': 10,
        'total_tokens_generated': 500,
        'average_generation_time': 0.5,
        'cache_hits': 5,
        'model_id': 'distilbert-base-uncased',
        'is_initialized': True,
        'rate_limited_count': 2,
        'rate_limited_wait_time': 1.5
    }
    
    # Patch the necessary components
    with patch('core.llm.llm_factory.LLMFactory.create_client', return_value=mock_client):
        with patch('services.mapping.semantic_mapper.SemanticMapper.map_fields', mock_map_fields):
            # Upload the file
            with open(test_file_path, 'rb') as f:
                response = client.post(
                    '/api/map',
                    data={
                        'file': (f, 'test.csv'),
                        'mapping_type': 'semantic'
                    },
                    content_type='multipart/form-data'
                )
            
            # Check that the response is OK
            print(f"Response status code: {response.status_code}")
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
            
            # Check the response data
            data = json.loads(response.data)
            print(f"Response data: {json.dumps(data, indent=2)}")
            
            # Check that the response contains the expected data
            assert 'success' in data, "Response missing 'success' field"
            assert data['success'] is True, "Response 'success' field is not True"
            assert 'mapping' in data, "Response missing 'mapping' field"
            assert 'SKU' in data['mapping'], "Response mapping missing 'SKU' field"
            assert 'Short Description' in data['mapping'], "Response mapping missing 'Short Description' field"
            assert 'Trade Price' in data['mapping'], "Response mapping missing 'Trade Price' field"
            
            # Check that rate limiting was applied
            assert mock_client.generate_response.called, "LLM client generate_response was not called"
            
            print("All tests passed!")
    
    # Clean up the test file
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

if __name__ == '__main__':
    test_map_endpoint()
