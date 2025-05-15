"""
Integration tests for the API endpoints
"""
import pytest
import os
import json
import tempfile
from pathlib import Path
import pandas as pd

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPI:
    """Integration tests for the API endpoints"""

    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get('/api/health')

        # Check that the response is OK
        assert response.status_code == 200

        # Check that the response contains the expected data
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'
        assert 'version' in data
        assert 'app_name' in data

    def test_upload_file(self, client, temp_csv_file):
        """Test uploading a file"""
        # Open the file
        with open(temp_csv_file, 'rb') as f:
            # Create a multipart form with the file
            data = {
                'file': (f, os.path.basename(temp_csv_file)),
                'format': 'csv'
            }

            # Send the request
            response = client.post(
                '/api/upload',
                data=data,
                content_type='multipart/form-data'
            )

            # Check that the response is OK
            assert response.status_code == 200

            # Check that the response is a file
            assert response.mimetype == 'text/csv'
            assert 'attachment' in response.headers['Content-Disposition']

    def test_upload_no_file(self, client):
        """Test uploading without a file"""
        # Send the request without a file
        response = client.post('/api/upload')

        # Check that the response is an error
        assert response.status_code == 400

        # Check that the response contains an error message
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file provided' in data['error']

    def test_analyze_file(self, client, sample_csv_data):
        """Test analyzing a file"""
        # Mock the analyze endpoint using a patch
        from unittest.mock import patch

        # Create a mock structure analyzer that returns a predefined structure
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

            # Create a temporary file in memory
            import io
            csv_data = sample_csv_data.to_csv(index=False).encode('utf-8')
            csv_file = io.BytesIO(csv_data)

            # Create a multipart form with the file
            data = {
                'file': (csv_file, 'test_data.csv')
            }

            # Send the request
            response = client.post(
                '/api/analyze',
                data=data,
                content_type='multipart/form-data'
            )

            # Check that the response is OK
            assert response.status_code == 200

        # Check that the response contains the expected data
        data = json.loads(response.data)

        # Handle both response formats (direct and nested in 'analysis')
        if 'success' in data and 'analysis' in data:
            # New format with 'success' and 'analysis' keys
            assert data['success'] is True

            # Check if structure, sample_data, and columns are in the root or in analysis
            if 'structure' in data:
                structure = data['structure']
                sample_data = data['sample_data']
                columns = data['columns']
            else:
                # Check if they're in the analysis key
                assert 'structure' in data['analysis'] or 'structure' in data

                # Use the structure from wherever it exists
                structure = data['structure'] if 'structure' in data else data['analysis']['structure']
                sample_data = data['sample_data'] if 'sample_data' in data else data['analysis']['sample_data']
                columns = data['columns'] if 'columns' in data else data['analysis']['columns']
        else:
            # Old format with direct keys
            assert 'structure' in data
            assert 'sample_data' in data
            assert 'columns' in data

            structure = data['structure']
            sample_data = data['sample_data']
            columns = data['columns']

        # Check structure
        assert 'column_types' in structure
        assert 'row_count' in structure
        assert 'column_count' in structure

        # Check sample data
        assert isinstance(sample_data, list)
        assert len(sample_data) > 0

        # Check columns
        assert isinstance(columns, list)
        assert len(columns) > 0

    def test_map_fields(self, client, sample_csv_data):
        """Test mapping fields"""
        # Create the request data
        columns = list(sample_csv_data.columns)
        sample_data = sample_csv_data.head(3).to_dict(orient='records')

        request_data = {
            'columns': columns,
            'sample_data': sample_data
        }

        # Send the request
        response = client.post(
            '/api/map-fields',
            json=request_data,
            content_type='application/json'
        )

        # Check that the response is OK
        assert response.status_code == 200

        # Check that the response contains the expected data
        data = json.loads(response.data)
        assert 'mappings' in data

        # Check mappings
        mappings = data['mappings']
        assert isinstance(mappings, list)
        assert len(mappings) > 0

        # Check mapping structure
        for mapping in mappings:
            assert 'source_field' in mapping
            assert 'target_field' in mapping
            assert 'confidence' in mapping