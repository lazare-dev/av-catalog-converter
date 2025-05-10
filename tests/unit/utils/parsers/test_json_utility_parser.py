"""
Unit tests for the JSON parser utility (utils.parsers.json_parser)
"""
import pytest
from utils.parsers.json_parser import JSONParser


class TestJSONParser:
    """Test cases for JSONParser utility"""

    def test_init(self):
        """Test initialization"""
        parser = JSONParser()
        assert hasattr(parser, 'logger')

    def test_parse_json(self):
        """Test parsing valid JSON"""
        json_str = '{"name": "HD Camera", "price": 299.99, "features": ["1080p", "Zoom"]}'
        parser = JSONParser()
        result = parser.parse(json_str)

        # Check that the result is a dictionary
        assert isinstance(result, dict)

        # Check that the data matches the expected structure
        assert result['name'] == 'HD Camera'
        assert result['price'] == 299.99
        assert result['features'] == ['1080p', 'Zoom']

    def test_parse_json_in_code_block(self):
        """Test parsing JSON in a code block"""
        json_in_block = """
Here is some JSON data:

```json
{
  "name": "HD Camera",
  "price": 299.99,
  "features": ["1080p", "Zoom"]
}
```

That's all.
"""
        parser = JSONParser()
        result = parser.parse(json_in_block)

        # Check that the result is a dictionary
        assert isinstance(result, dict)

        # Check that the data matches the expected structure
        assert result['name'] == 'HD Camera'
        assert result['price'] == 299.99
        assert result['features'] == ['1080p', 'Zoom']

    def test_parse_json_with_single_quotes(self):
        """Test parsing JSON with single quotes"""
        json_with_single_quotes = "{'name': 'HD Camera', 'price': 299.99}"
        parser = JSONParser()
        result = parser.parse(json_with_single_quotes)

        # Check that the result is a dictionary
        assert isinstance(result, dict)

        # Check that the data matches the expected structure
        assert result['name'] == 'HD Camera'
        assert result['price'] == 299.99

    def test_parse_json_with_unquoted_keys(self):
        """Test parsing JSON with unquoted keys"""
        json_with_unquoted_keys = "{name: 'HD Camera', price: 299.99}"
        parser = JSONParser()
        result = parser.parse(json_with_unquoted_keys)

        # Check that the result is a dictionary
        assert isinstance(result, dict)

        # Check that the data matches the expected structure
        assert result['name'] == 'HD Camera'
        assert result['price'] == 299.99

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON"""
        invalid_json = "{name: 'HD Camera', price: 299.99, features: ['1080p', 'Zoom'"  # Missing closing bracket
        parser = JSONParser()
        result = parser.parse(invalid_json)

        # Check that the result is None for invalid JSON
        assert result is None

    def test_extract_json_blocks(self):
        """Test extracting JSON blocks from text"""
        text_with_blocks = """
Here is some JSON:

```json
{
  "name": "HD Camera",
  "price": 299.99
}
```

And another block:

```
{
  "name": "Wireless Mic",
  "price": 149.50
}
```
"""
        parser = JSONParser()
        blocks = parser._extract_json_blocks(text_with_blocks)

        # Check that both blocks are extracted
        assert len(blocks) == 2
        assert '"name": "HD Camera"' in blocks[0]
        assert '"name": "Wireless Mic"' not in blocks[0]

    def test_extract_and_fix_json(self):
        """Test extracting and fixing JSON-like structures"""
        text_with_json_like = """
Here is some JSON-like data:

{
  name: 'HD Camera',
  price: 299.99,
  features: ['1080p', 'Zoom'],
}
"""
        parser = JSONParser()
        fixed_json = parser._extract_and_fix_json(text_with_json_like)

        # Check that the JSON is extracted and fixed
        assert fixed_json is not None
        assert '"name"' in fixed_json
        assert '"price"' in fixed_json
        assert '"features"' in fixed_json
        assert "'" not in fixed_json  # Single quotes should be replaced with double quotes
