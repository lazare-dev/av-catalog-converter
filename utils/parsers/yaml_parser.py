# utils/parsers/yaml_parser.py
"""
YAML response parsing
"""
import logging
from typing import Dict, Any, Optional, List
import re

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

class YAMLParser:
    """Utility for parsing YAML from AI model responses"""

    def __init__(self):
        """Initialize the YAML parser"""
        self.logger = logging.getLogger(__name__)

        if not YAML_AVAILABLE:
            self.logger.warning("PyYAML library not available, YAML parsing disabled")

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse YAML from text

        Args:
            text (str): Text that may contain YAML

        Returns:
            Optional[Dict[str, Any]]: Parsed YAML or None if parsing failed
        """
        if not YAML_AVAILABLE:
            self.logger.error("PyYAML library not available, cannot parse YAML")
            return None

        if not text or not isinstance(text, str):
            self.logger.warning("Invalid input: text must be a non-empty string")
            return None

        # Try to extract YAML from code blocks first
        yaml_blocks = self._extract_yaml_blocks(text)

        # Try each extracted block
        for block in yaml_blocks:
            result = self._try_parse_yaml(block)
            if result is not None:
                return result

        # If no blocks were found or none could be parsed, try the whole text
        return self._try_parse_yaml(text)

    def _try_parse_yaml(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse a string as YAML

        Args:
            text (str): Text to parse

        Returns:
            Optional[Dict[str, Any]]: Parsed YAML or None if parsing failed
        """
        try:
            result = yaml.safe_load(text)
            # Ensure the result is a dictionary
            if isinstance(result, dict):
                return result
            elif result is None:
                # Empty YAML or just comments
                return {}
            else:
                self.logger.debug(f"YAML parsed but result is not a dictionary: {type(result)}")
                return None
        except yaml.YAMLError as e:
            self.logger.debug(f"YAML parsing error: {str(e)}")
            return None

    def _extract_yaml_blocks(self, text: str) -> List[str]:
        """
        Extract YAML blocks from text

        Args:
            text (str): Text to parse

        Returns:
            List[str]: Extracted YAML blocks
        """
        # Look for YAML in code blocks with explicit yaml/yml tag
        yaml_pattern = r'```(?:yaml|yml)\s*([\s\S]*?)\s*```'
        matches = re.findall(yaml_pattern, text)

        # If no explicit yaml blocks found, try generic code blocks
        if not matches:
            generic_pattern = r'```\s*([\s\S]*?)\s*```'
            matches = re.findall(generic_pattern, text)

        return matches