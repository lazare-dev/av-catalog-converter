# core/file_parser/xml_parser.py
"""
XML-specific parsing implementation
"""
import logging
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, Optional

from core.file_parser.base_parser import BaseParser
from config.settings import PARSER_CONFIG

class XMLParser(BaseParser):
    """Parser for XML files"""

    def __init__(self, file_path):
        """
        Initialize the XML parser

        Args:
            file_path (str): Path to the XML file
        """
        super().__init__(file_path)
        self.logger = logging.getLogger(__name__)

        # Add XML config to PARSER_CONFIG if not present
        if 'xml' not in PARSER_CONFIG:
            PARSER_CONFIG['xml'] = {
                "flatten_attributes": True,
                "record_path": None,  # Auto-detect
                "attribute_prefix": "@"
            }

        self.config = PARSER_CONFIG.get('xml', {})
        self.root = None

    def parse(self) -> pd.DataFrame:
        """
        Parse the XML file into a pandas DataFrame

        Returns:
            pd.DataFrame: Parsed data
        """
        self.logger.info(f"Parsing XML file: {self.file_path}")

        try:
            # Parse the XML file
            tree = ET.parse(self.file_path)
            self.root = tree.getroot()

            # Find the most likely record path (repeating elements)
            record_path = self.config.get('record_path') or self._detect_record_path()

            if record_path:
                self.logger.info(f"Using record path: {record_path}")
                df = self._parse_records(record_path)
            else:
                self.logger.warning("No repeating elements found, parsing as single record")
                df = self._parse_as_single_record()

            # Apply common preprocessing
            df = self.preprocess_dataframe(df)

            self.logger.info(f"Successfully parsed XML with {len(df)} rows and {len(df.columns)} columns")
            return df

        except Exception as e:
            self.logger.error(f"Error parsing XML file: {str(e)}")
            return pd.DataFrame()

    def _detect_record_path(self) -> Optional[str]:
        """
        Detect the most likely record path in the XML

        Returns:
            Optional[str]: Detected record path or None
        """
        if self.root is None:
            return None

        # Count occurrences of each element path
        path_counts = {}

        def count_elements(element, path=""):
            # Build the current path
            current_path = f"{path}/{element.tag}" if path else element.tag

            # Count this element
            path_counts[current_path] = path_counts.get(current_path, 0) + 1

            # Recursively count children
            for child in element:
                count_elements(child, current_path)

        # Start counting from the root
        for child in self.root:
            count_elements(child)

        # Find paths with multiple occurrences (potential record paths)
        candidates = [(path, count) for path, count in path_counts.items() if count > 1]

        if not candidates:
            return None

        # Sort by count (descending) and path length (ascending)
        candidates.sort(key=lambda x: (-x[1], len(x[0].split('/'))))

        # Return the most likely record path
        return candidates[0][0]

    def _parse_records(self, record_path: str) -> pd.DataFrame:
        """
        Parse XML records at the specified path

        Args:
            record_path (str): Path to the records

        Returns:
            pd.DataFrame: Parsed records as DataFrame
        """
        # Find all elements at the record path
        path_parts = record_path.split('/')

        # Navigate to the parent element
        parent = self.root
        for part in path_parts[:-1]:
            parent = parent.find(part)
            if parent is None:
                self.logger.error(f"Could not find path: {record_path}")
                return pd.DataFrame()

        # Get all matching child elements
        records = parent.findall(path_parts[-1])

        # Convert records to dictionaries
        data = []
        for record in records:
            record_dict = self._element_to_dict(record)
            data.append(record_dict)

        # Convert to DataFrame
        return pd.DataFrame(data)

    def _parse_as_single_record(self) -> pd.DataFrame:
        """
        Parse the XML as a single record

        Returns:
            pd.DataFrame: Single-row DataFrame
        """
        if self.root is None:
            return pd.DataFrame()

        # Convert root to dictionary
        data = self._element_to_dict(self.root)

        # Return as single-row DataFrame
        return pd.DataFrame([data])

    def _element_to_dict(self, element: ET.Element) -> Dict:
        """
        Convert an XML element to a dictionary

        Args:
            element (ET.Element): XML element

        Returns:
            Dict: Element as dictionary
        """
        result = {}

        # Add attributes if present
        if element.attrib and self.config.get('flatten_attributes', True):
            prefix = self.config.get('attribute_prefix', '@')
            for key, value in element.attrib.items():
                result[f"{prefix}{key}"] = value

        # Process child elements
        for child in element:
            child_dict = self._element_to_dict(child)

            # If the child has no children and no attributes, use its text
            if not child_dict and child.text and child.text.strip():
                child_dict = child.text.strip()

            # Handle duplicate keys
            if child.tag in result:
                # If this key already exists, convert to list
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict

        # Add text content if present and no children
        if not result and element.text and element.text.strip():
            return element.text.strip()

        return result
