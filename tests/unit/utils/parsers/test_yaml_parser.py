"""
Unit tests for the YAML parser
"""
import pytest
from utils.parsers.yaml_parser import YAMLParser


class TestYAMLParser:
    """Test cases for YAMLParser"""

    def test_init(self):
        """Test initialization"""
        parser = YAMLParser()
        assert hasattr(parser, 'logger')

    def test_parse_yaml(self, sample_yaml_data):
        """Test parsing YAML data"""
        parser = YAMLParser()
        result = parser.parse(sample_yaml_data)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the data matches the expected structure
        assert 'products' in result
        assert len(result['products']) == 3
        assert result['products'][0]['sku'] == 'ABC123'
        assert result['products'][1]['name'] == 'Wireless Mic'
        assert result['products'][2]['price'] == 499.00

    def test_parse_yaml_in_code_block(self):
        """Test parsing YAML in a code block"""
        yaml_in_block = """
Here is some YAML data:

```yaml
products:
  - sku: ABC123
    name: HD Camera
    price: 299.99
  - sku: DEF456
    name: Wireless Mic
    price: 149.50
```

That's all.
"""
        parser = YAMLParser()
        result = parser.parse(yaml_in_block)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the data matches the expected structure
        assert 'products' in result
        assert len(result['products']) == 2
        assert result['products'][0]['sku'] == 'ABC123'
        assert result['products'][1]['name'] == 'Wireless Mic'

    def test_parse_yaml_in_generic_block(self):
        """Test parsing YAML in a generic code block"""
        yaml_in_generic_block = """
Here is some YAML data:

```
products:
  - sku: ABC123
    name: HD Camera
    price: 299.99
  - sku: DEF456
    name: Wireless Mic
    price: 149.50
```

That's all.
"""
        parser = YAMLParser()
        result = parser.parse(yaml_in_generic_block)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the data matches the expected structure
        assert 'products' in result
        assert len(result['products']) == 2

    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML"""
        invalid_yaml = """
This is not valid YAML:
  - item1
  item2: value
    nested: wrong indentation
"""
        parser = YAMLParser()
        result = parser.parse(invalid_yaml)
        
        # Check that the result is None for invalid YAML
        assert result is None

    def test_parse_empty_input(self):
        """Test parsing empty input"""
        parser = YAMLParser()
        result = parser.parse("")
        
        # Check that the result is None for empty input
        assert result is None
        
        result = parser.parse(None)
        assert result is None

    def test_extract_yaml_blocks(self):
        """Test extracting YAML blocks from text"""
        text_with_blocks = """
Here is some YAML:

```yaml
key1: value1
key2: value2
```

And another block:

```yml
key3: value3
key4: value4
```
"""
        parser = YAMLParser()
        blocks = parser._extract_yaml_blocks(text_with_blocks)
        
        # Check that both blocks are extracted
        assert len(blocks) == 2
        assert "key1: value1" in blocks[0]
        assert "key3: value3" in blocks[1]
