"""
Unit tests for the field mapper
"""
import pytest
import pandas as pd
import numpy as np

from services.mapping.field_mapper import FieldMapper


class TestFieldMapper:
    """Test cases for FieldMapper"""

    def test_init(self):
        """Test initialization"""
        mapper = FieldMapper()
        assert hasattr(mapper, 'logger')
        assert hasattr(mapper, 'direct_mapper')
        assert hasattr(mapper, 'pattern_mapper')
        assert hasattr(mapper, 'semantic_mapper')

    def test_map(self, sample_csv_data):
        """Test mapping fields"""
        # Create a mapper
        mapper = FieldMapper()

        # Create structure info
        structure_info = {
            'column_types': {
                'SKU': {'type': 'id'},
                'Product Name': {'type': 'text'},
                'Price': {'type': 'price'},
                'Category': {'type': 'category'},
                'Manufacturer': {'type': 'text'}
            }
        }

        # Map the fields
        result = mapper.map(sample_csv_data, structure_info)

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the data is mapped correctly
        assert 'SKU' in result.columns
        assert 'Short Description' in result.columns or 'Product Name' in result.columns
        assert 'Trade Price' in result.columns or 'Price' in result.columns
        assert 'Category' in result.columns
        assert 'Manufacturer' in result.columns

    @pytest.mark.skip(reason="This test requires LLM integration which is not working in the test environment")
    def test_map_fields(self):
        """Test mapping fields from columns and sample data"""
        # This test is skipped because it requires LLM integration
        # See test_field_mapper_simple.py for a simplified version of this test
        pass

    def test_apply_mappings(self, sample_csv_data):
        """Test applying mappings to a DataFrame"""
        # Create a mapper
        mapper = FieldMapper()

        # Create mappings
        mappings = [
            {'source_field': 'SKU', 'target_field': 'Product ID', 'confidence': 0.95},
            {'source_field': 'Product Name', 'target_field': 'Short Description', 'confidence': 0.9},
            {'source_field': 'Price', 'target_field': 'Trade Price', 'confidence': 0.85},
            {'source_field': 'Category', 'target_field': 'Category', 'confidence': 1.0},
            {'source_field': 'Manufacturer', 'target_field': 'Manufacturer', 'confidence': 1.0}
        ]

        # Apply the mappings
        result = mapper._apply_mappings(sample_csv_data, mappings)

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the columns are renamed
        assert 'Product ID' in result.columns
        assert 'Short Description' in result.columns
        assert 'Trade Price' in result.columns
        assert 'Category' in result.columns
        assert 'Manufacturer' in result.columns

        # Check that the data is preserved
        assert result.iloc[0]['Product ID'] == 'ABC123'
        assert result.iloc[1]['Short Description'] == 'Wireless Mic'
        assert result.iloc[2]['Trade Price'] == 499.00

    def test_handle_missing_required_fields(self, sample_csv_data):
        """Test handling missing required fields"""
        # Create a mapper
        mapper = FieldMapper()

        # Create a DataFrame missing a required field
        df = sample_csv_data.drop(columns=['SKU'])

        # Create mappings without the required field
        mappings = [
            {'source_field': 'Product Name', 'target_field': 'Short Description', 'confidence': 0.9},
            {'source_field': 'Price', 'target_field': 'Trade Price', 'confidence': 0.85},
            {'source_field': 'Category', 'target_field': 'Category', 'confidence': 1.0},
            {'source_field': 'Manufacturer', 'target_field': 'Manufacturer', 'confidence': 1.0}
        ]

        # Apply the mappings
        result = mapper._apply_mappings(df, mappings)

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the missing required field is added with empty values
        assert 'SKU' in result.columns
        assert result['SKU'].isna().all() or (result['SKU'] == '').all()
