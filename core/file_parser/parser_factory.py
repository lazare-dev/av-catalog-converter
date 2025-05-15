"""Factory for creating appropriate file parsers"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

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
from utils.logging.logger import Logger

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
        logger = Logger.get_logger(__name__)
        file_path = Path(file_path)

        # Log detailed file information
        file_stats = {}
        if file_path.exists():
            file_stats = {
                'size_bytes': file_path.stat().st_size,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified_time': file_path.stat().st_mtime,
                'absolute_path': str(file_path.absolute()),
                'extension': file_path.suffix.lower()
            }
            logger.debug(f"Processing file", file_info=file_stats)
        else:
            logger.error(f"File not found", path=str(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try to determine parser by file extension
        extension = file_path.suffix.lower()
        parser_class = cls.EXTENSION_MAP.get(extension)

        # Log the extension detection attempt
        logger.debug(f"Attempting to detect parser by extension",
                    extension=extension,
                    detected_parser=parser_class.__name__ if parser_class else None)

        # If extension doesn't match, try to detect file type
        if not parser_class:
            logger.info(f"No parser found for extension {extension}, detecting file type",
                       file_path=str(file_path),
                       extension=extension)
            parser_class = cls._detect_parser_by_content(file_path)

        if not parser_class:
            logger.warning(f"No suitable parser found for {file_path}, defaulting to CSV",
                          file_path=str(file_path),
                          attempted_methods=["extension", "content_detection"])
            parser_class = CSVParser

        logger.info(f"Using {parser_class.__name__} for {file_path}",
                   parser=parser_class.__name__,
                   file_path=str(file_path),
                   file_info=file_stats)
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
        logger = Logger.get_logger(__name__)

        # Log the content detection attempt
        logger.debug(f"Attempting to detect file type by content",
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size if file_path.exists() else 0)

        # Check if magic module is available
        if magic is None:
            logger.warning("python-magic library not available, using fallback detection methods",
                          file_path=str(file_path),
                          fallback="content-based detection")
            return cls._fallback_detection(file_path)

        try:
            # Use python-magic to detect MIME type
            mime_type = magic.from_file(str(file_path), mime=True)
            logger.info(f"Detected MIME type: {mime_type}",
                       file_path=str(file_path),
                       mime_type=mime_type,
                       detection_method="python-magic")

            parser_class = cls.MIME_MAP.get(mime_type)
            if parser_class:
                logger.debug(f"Found parser for MIME type",
                           mime_type=mime_type,
                           parser=parser_class.__name__)
                return parser_class

            # If MIME type detection didn't help, try more specific checks
            if mime_type == 'text/plain':
                logger.debug(f"MIME type is text/plain, performing additional text file analysis",
                           file_path=str(file_path))
                return cls._detect_text_file_type(file_path)

            # Check for XML files
            if mime_type in ('text/plain', 'application/xml', 'text/xml'):
                logger.debug(f"Checking if file is XML",
                           file_path=str(file_path),
                           mime_type=mime_type)
                with open(file_path, 'rb') as f:
                    sample = f.read(100).decode('utf-8', errors='replace')
                    if sample.strip().startswith('<?xml'):
                        logger.debug(f"File appears to be XML based on content",
                                   file_path=str(file_path))
                        return XMLParser

            logger.debug(f"Could not determine parser from MIME type",
                       file_path=str(file_path),
                       mime_type=mime_type)
            return None
        except Exception as e:
            logger.error(f"Error detecting file type: {e}",
                        file_path=str(file_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                        stack_info=True)
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
        logger = Logger.get_logger(__name__)
        logger.info("Using fallback file type detection",
                   file_path=str(file_path),
                   detection_method="content-based fallback")

        # Try to detect based on file content
        try:
            # Check if it's a JSON file
            with open(file_path, 'rb') as f:
                sample = f.read(100).decode('utf-8', errors='replace').strip()
                logger.debug(f"Analyzing file content sample for JSON detection",
                           file_path=str(file_path),
                           sample_start=sample[:20] if len(sample) > 20 else sample)

                if sample.startswith('{') and sample.endswith('}'):
                    logger.debug(f"File appears to be JSON (object)", file_path=str(file_path))
                    return JSONParser
                if sample.startswith('[') and sample.endswith(']'):
                    logger.debug(f"File appears to be JSON (array)", file_path=str(file_path))
                    return JSONParser

            # Check if it's an XML file
            with open(file_path, 'rb') as f:
                sample = f.read(100).decode('utf-8', errors='replace').strip()
                logger.debug(f"Analyzing file content sample for XML detection",
                           file_path=str(file_path),
                           sample_start=sample[:20] if len(sample) > 20 else sample)

                if sample.startswith('<?xml') or sample.startswith('<'):
                    logger.debug(f"File appears to be XML", file_path=str(file_path))
                    return XMLParser

            # Check if it's a CSV/TSV file
            logger.debug(f"Checking if file is a delimited text file", file_path=str(file_path))
            return cls._detect_text_file_type(file_path)

        except Exception as e:
            logger.error(f"Error in fallback detection: {e}",
                        file_path=str(file_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                        stack_info=True)
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
        logger = Logger.get_logger(__name__)
        logger.debug(f"Analyzing text file for delimiter pattern", file_path=str(file_path))

        try:
            # Check if it's a CSV/TSV file
            with open(file_path, 'rb') as f:
                sample = f.read(4096).decode('utf-8', errors='replace')

                # Count delimiters
                comma_count = sample.count(',')
                tab_count = sample.count('\t')
                pipe_count = sample.count('|')
                semicolon_count = sample.count(';')

                # Log delimiter counts
                delimiter_counts = {
                    'comma': comma_count,
                    'tab': tab_count,
                    'pipe': pipe_count,
                    'semicolon': semicolon_count
                }
                logger.debug(f"Delimiter counts in sample",
                           file_path=str(file_path),
                           delimiter_counts=delimiter_counts,
                           sample_size=len(sample),
                           lines=sample.count('\n'))

                # If it has consistent delimiters, it's probably a CSV
                max_delimiter = max(comma_count, tab_count, pipe_count, semicolon_count)
                if max_delimiter > 5:
                    # Determine which delimiter is most common
                    if max_delimiter == comma_count:
                        delimiter = 'comma'
                    elif max_delimiter == tab_count:
                        delimiter = 'tab'
                    elif max_delimiter == pipe_count:
                        delimiter = 'pipe'
                    else:
                        delimiter = 'semicolon'

                    logger.info(f"Detected delimited text file",
                               file_path=str(file_path),
                               delimiter=delimiter,
                               count=max_delimiter)
                    return CSVParser
                else:
                    logger.debug(f"Not enough delimiters found to classify as delimited text file",
                               file_path=str(file_path),
                               max_delimiter_count=max_delimiter)

            logger.debug(f"Could not determine text file type", file_path=str(file_path))
            return None
        except Exception as e:
            logger.error(f"Error detecting text file type: {e}",
                        file_path=str(file_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                        stack_info=True)
            return None
