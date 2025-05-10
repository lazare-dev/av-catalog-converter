"""
Unit tests for the parser factory
"""
import pytest
from pathlib import Path
import pandas as pd

from core.file_parser.parser_factory import ParserFactory
from core.file_parser.csv_parser import CSVParser
from core.file_parser.json_parser import JSONParser
from core.file_parser.xml_parser import XMLParser
from core.file_parser.excel_parser import ExcelParser


class TestParserFactory:
    """Test cases for ParserFactory"""

    def test_create_parser_csv(self, temp_csv_file):
        """Test creating a CSV parser"""
        parser = ParserFactory.create_parser(temp_csv_file)
        assert isinstance(parser, CSVParser)
        assert parser.file_path == Path(temp_csv_file)

    def test_create_parser_json(self, temp_json_file):
        """Test creating a JSON parser"""
        parser = ParserFactory.create_parser(temp_json_file)
        assert isinstance(parser, JSONParser)
        assert parser.file_path == Path(temp_json_file)

    def test_create_parser_xml(self, temp_xml_file):
        """Test creating an XML parser"""
        parser = ParserFactory.create_parser(temp_xml_file)
        assert isinstance(parser, XMLParser)
        assert parser.file_path == Path(temp_xml_file)

    def test_file_not_found(self):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            ParserFactory.create_parser("non_existent_file.csv")

    def test_fallback_detection(self, temp_csv_file, monkeypatch):
        """Test fallback detection when magic is not available"""
        # Mock magic as None to test fallback detection
        monkeypatch.setattr('core.file_parser.parser_factory.magic', None)
        
        parser = ParserFactory.create_parser(temp_csv_file)
        assert isinstance(parser, CSVParser)

    def test_detect_parser_by_content(self, temp_json_file, monkeypatch):
        """Test detection by content"""
        # Rename the file to remove extension
        new_path = temp_json_file.with_suffix('')
        temp_json_file.rename(new_path)
        
        # Test if content detection works
        parser = ParserFactory.create_parser(new_path)
        assert isinstance(parser, JSONParser)

    def test_default_to_csv(self, tmp_path, monkeypatch):
        """Test defaulting to CSV when no parser is found"""
        # Create a file with unknown extension
        unknown_file = tmp_path / "test_data.unknown"
        with open(unknown_file, 'w') as f:
            f.write("This is a test file with unknown format")
        
        # Mock _detect_parser_by_content to return None
        monkeypatch.setattr(
            'core.file_parser.parser_factory.ParserFactory._detect_parser_by_content',
            lambda cls, path: None
        )
        
        parser = ParserFactory.create_parser(unknown_file)
        assert isinstance(parser, CSVParser)
