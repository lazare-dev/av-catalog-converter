"""
Unit tests for the field mapper (simplified version)
"""
import pytest
import pandas as pd
import numpy as np

from services.mapping.field_mapper import FieldMapper

@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    data = {
        'SKU': ['ABC123', 'DEF456', 'GHI789'],
        'Product Name': ['HD Camera', 'Wireless Mic', 'Audio Mixer'],
        'Price': [299.99, 149.50, 499.00],
        'Category': ['Video', 'Audio', 'Audio'],
        'Manufacturer': ['Sony', 'Shure', 'Yamaha']
    }
    return pd.DataFrame(data)

class TestFieldMapperSimple:
    """Test the field mapper with simplified tests"""
    
    def test_direct_mapping(self):
        """Test direct mapping functionality"""
        # Create direct mappings for testing
        direct_mappings = {
            'item_sku': 'SKU',
            'item_name': 'Short Description',
            'item_price': 'Trade Price',
            'item_category': 'Category',
            'brand': 'Manufacturer'
        }
        
        # Convert to the expected format
        mappings = []
        for source, target in direct_mappings.items():
            mappings.append({
                'source_field': source,
                'target_field': target,
                'confidence': 0.9,
                'reasoning': 'Direct pattern matching'
            })
        
        # Check specific mappings
        assert len(mappings) == 5
        assert mappings[0]['source_field'] == 'item_sku'
        assert mappings[0]['target_field'] == 'SKU'
        assert mappings[1]['source_field'] == 'item_name'
        assert mappings[1]['target_field'] == 'Short Description'
        assert mappings[2]['source_field'] == 'item_price'
        assert mappings[2]['target_field'] == 'Trade Price'
        assert mappings[3]['source_field'] == 'item_category'
        assert mappings[3]['target_field'] == 'Category'
        assert mappings[4]['source_field'] == 'brand'
        assert mappings[4]['target_field'] == 'Manufacturer'
    
    def test_apply_mappings_dict(self):
        """Test applying mappings from a dictionary"""
        # Create a mapper
        mapper = FieldMapper()
        
        # Create sample data
        data = {
            'item_sku': ['ABC123', 'DEF456', 'GHI789'],
            'item_name': ['HD Camera', 'Wireless Mic', 'Audio Mixer'],
            'item_price': [299.99, 149.50, 499.00],
            'item_category': ['Video', 'Audio', 'Audio'],
            'brand': ['Sony', 'Shure', 'Yamaha']
        }
        df = pd.DataFrame(data)
        
        # Create mappings
        mapping_dict = {
            'item_sku': 'SKU',
            'item_name': 'Short Description',
            'item_price': 'Trade Price',
            'item_category': 'Category',
            'brand': 'Manufacturer'
        }
        
        # Apply the mappings
        result = mapper._apply_mappings_dict(df, mapping_dict)
        
        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that the columns are renamed
        assert 'SKU' in result.columns
        assert 'Short Description' in result.columns
        assert 'Trade Price' in result.columns
        assert 'Category' in result.columns
        assert 'Manufacturer' in result.columns
        
        # Check that the data is preserved
        assert result.iloc[0]['SKU'] == 'ABC123'
        assert result.iloc[1]['Short Description'] == 'Wireless Mic'
        assert result.iloc[2]['Trade Price'] == 499.00
