"""Factory for creating appropriate file parsers"""

import logging
from pathlib import Path
from typing import Optional

try:
    import magic
except ImportError:
    magic = None

from core.file_parser.base_parser import BaseParser
from core.file_parser.csv_parser import CSVParser
from core.file_parser.excel_parser import ExcelParser
from core.file_parser.pdf_parser import PDFParser
from core.file_parser.json_parser import JSONParser
from core.file_parser.xml_parser import XMLParser

class ParserFactory:
    """Factory for creating appropriate file parser"""

    # Mapping of file extensions to parser classes
    EXTENSION_MAP = {
        '.csv': CSVParser,
        '.tsv': CSVParser,
        '.txt': CSVParser,
        '.xlsx': ExcelParser,
        '.xls': ExcelParser,
        '.xlsm': ExcelParser,
        '.pdf': PDFParser,
        '.json': JSONParser,
        '.xml': XMLParser,
    }

    # Mapping of MIME types to parser classes
    MIME_MAP = {
        'text/csv': CSVParser,
        'text/plain': CSVParser,
        'application/vnd.ms-excel': ExcelParser,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ExcelParser,
        'application/pdf': PDFParser,
        'application/json': JSONParser,
        'text/xml': XMLParser,
        'application/xml': XMLParser,
    }

    @classmethod
    def create_parser(cls, file_path: str) -> BaseParser:
        """
        Create and return the appropriate parser for the file

        Args:
            file_path (str): Path to the input file

        Returns:
            BaseParser: Appropriate parser instance
        """
        logger = logging.getLogger(__name__)
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try to determine parser by file extension
        extension = file_path.suffix.lower()
        parser_class = cls.EXTENSION_MAP.get(extension)

        # If extension doesn't match, try to detect file type
        if not parser_class:
            logger.info(f"No parser found for extension {extension}, detecting file type")
            parser_class = cls._detect_parser_by_content(file_path)

        if not parser_class:
            logger.warning(f"No suitable parser found for {file_path}, defaulting to CSV")
            parser_class = CSVParser

        logger.info(f"Using {parser_class.__name__} for {file_path}")
        return parser_class(file_path)

    @classmethod
    def _detect_parser_by_content(cls, file_path: Path) -> Optional[type]:
        """
        Detect the appropriate parser based on file content

        Args:
            file_path (Path): Path to the input file

        Returns:
            Optional[type]: Parser class or None if detection failed
        """
        logger = logging.getLogger(__name__)

        # Check if magic module is available
        if magic is None:
            logger.warning("python-magic library not available, using fallback detection methods")
            return cls._fallback_detection(file_path)

        try:
            # Use python-magic to detect MIME type
            mime_type = magic.from_file(str(file_path), mime=True)
            logger.info(f"Detected MIME type: {mime_type}")

            parser_class = cls.MIME_MAP.get(mime_type)
            if parser_class:
                return parser_class

            # If MIME type detection didn't help, try more specific checks
            if mime_type == 'text/plain':
                return cls._detect_text_file_type(file_path)

            # Check for XML files
            if mime_type in ('text/plain', 'application/xml', 'text/xml'):
                with open(file_path, 'rb') as f:
                    sample = f.read(100).decode('utf-8', errors='replace')
                    if sample.strip().startswith('<?xml'):
                        return XMLParser

            return None
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            return cls._fallback_detection(file_path)

    @classmethod
    def _fallback_detection(cls, file_path: Path) -> Optional[type]:
        """
        Fallback method for file type detection when magic is not available

        Args:
            file_path (Path): Path to the input file

        Returns:
            Optional[type]: Parser class or None if detection failed
        """
        logger = logging.getLogger(__name__)
        logger.info("Using fallback file type detection")

        # Try to detect based on file content
        try:
            # Check if it's a JSON file
            with open(file_path, 'rb') as f:
                sample = f.read(100).decode('utf-8', errors='replace').strip()
                if sample.startswith('{') and sample.endswith('}'):
                    return JSONParser
                if sample.startswith('[') and sample.endswith(']'):
                    return JSONParser

            # Check if it's an XML file
            with open(file_path, 'rb') as f:
                sample = f.read(100).decode('utf-8', errors='replace').strip()
                if sample.startswith('<?xml') or sample.startswith('<'):
                    return XMLParser

            # Check if it's a CSV/TSV file
            return cls._detect_text_file_type(file_path)

        except Exception as e:
            logger.error(f"Error in fallback detection: {e}")
            return None

    @classmethod
    def _detect_text_file_type(cls, file_path: Path) -> Optional[type]:
        """
        Detect the type of a text file based on content analysis

        Args:
            file_path (Path): Path to the text file

        Returns:
            Optional[type]: Parser class or None if detection failed
        """
        try:
            # Check if it's a CSV/TSV file
            with open(file_path, 'rb') as f:
                sample = f.read(4096).decode('utf-8', errors='replace')

                # Count delimiters
                comma_count = sample.count(',')
                tab_count = sample.count('\t')
                pipe_count = sample.count('|')
                semicolon_count = sample.count(';')

                # If it has consistent delimiters, it's probably a CSV
                if max(comma_count, tab_count, pipe_count, semicolon_count) > 5:
                    return CSVParser

            return None
        except Exception as e:
            logging.getLogger(__name__).error(f"Error detecting text file type: {e}")
            return None
