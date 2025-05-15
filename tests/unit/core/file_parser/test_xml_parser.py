"""
Unit tests for the XML parser
"""
import pytest
import pandas as pd
from pathlib import Path

from core.file_parser.xml_parser import XMLParser


class TestXMLParser:
    """Test cases for XMLParser"""

    def test_init(self, temp_xml_file):
        """Test initialization"""
        parser = XMLParser(temp_xml_file)
        assert parser.file_path == Path(temp_xml_file)
        assert parser.root is None

    def test_parse(self, temp_xml_file):
        """Test parsing an XML file"""
        parser = XMLParser(temp_xml_file)
        result = parser.parse()

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the data matches the expected structure
        assert len(result) == 3
        assert 'sku' in result.columns
        assert 'name' in result.columns
        assert 'price' in result.columns
        assert 'category' in result.columns
        assert 'manufacturer' in result.columns

        # Check specific values
        assert result.iloc[0]['sku'] == 'ABC123'
        assert result.iloc[1]['name'] == 'Wireless Mic'
        assert result.iloc[2]['category'] == 'Audio'

    def test_detect_record_path(self, temp_xml_file):
        """Test record path detection"""
        parser = XMLParser(temp_xml_file)

        # Parse to initialize the root
        parser.parse()

        # Test record path detection
        record_path = parser._detect_record_path()
        assert record_path == 'product'  # The repeating element in the sample XML

    def test_parse_with_attributes(self, tmp_path):
        """Test parsing XML with attributes"""
        # Create XML file with attributes
        xml_with_attrs = tmp_path / "with_attrs.xml"
        with open(xml_with_attrs, 'w') as f:
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<catalog>
    <product id="1" status="in-stock">
        <sku>ABC123</sku>
        <name>HD Camera</name>
        <price currency="USD">299.99</price>
    </product>
    <product id="2" status="in-stock">
        <sku>DEF456</sku>
        <name>Wireless Mic</name>
        <price currency="USD">149.50</price>
    </product>
</catalog>
""")

        # Parse the file
        parser = XMLParser(xml_with_attrs)
        result = parser.parse()

        # Check that attributes are included (with or without @ prefix)
        assert 'id' in result.columns
        assert 'status' in result.columns
        assert 'sku' in result.columns
        assert 'name' in result.columns
        assert 'price_ currency' in result.columns or 'price_currency' in result.columns

    def test_parse_as_single_record(self, tmp_path):
        """Test parsing XML as a single record"""
        # Create XML file with a single record
        single_record = tmp_path / "single_record.xml"
        with open(single_record, 'w') as f:
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<product>
    <sku>ABC123</sku>
    <name>HD Camera</name>
    <price>299.99</price>
    <category>Video</category>
    <manufacturer>Sony</manufacturer>
</product>
""")

        # Parse the file
        parser = XMLParser(single_record)
        result = parser.parse()

        # Check that the result is a DataFrame with a single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'sku' in result.columns
        assert result.iloc[0]['sku'] == 'ABC123'

    def test_error_handling(self, tmp_path):
        """Test handling of invalid XML"""
        # Create an invalid XML file
        invalid_file = tmp_path / "invalid.xml"
        with open(invalid_file, 'w') as f:
            f.write("<invalid>This is not valid XML")

        # Parse the file
        parser = XMLParser(invalid_file)
        result = parser.parse()

        # Check that the result is an empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
