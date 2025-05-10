"""
Response parser utilities for LLM responses
"""
import logging
from typing import Dict, Any, Optional

from utils.parsers.json_parser import JSONParser
from utils.parsers.yaml_parser import YAMLParser

# Initialize parsers
json_parser = JSONParser()
yaml_parser = None  # Initialize if YAML parser is available

# Try to initialize YAML parser if available
try:
    yaml_parser = YAMLParser()
except ImportError:
    pass

def parse_llm_response(response: str) -> Dict:
    """
    Parse a response from the LLM to extract structured data.
    
    Args:
        response: Text response from LLM
        
    Returns:
        Parsed data as dictionary
    """
    logger = logging.getLogger(__name__)
    logger.debug("Parsing LLM response")
    
    # First try JSON parsing
    parsed_data = json_parser.parse(response)
    
    # If JSON parsing failed, try YAML if available
    if parsed_data is None and yaml_parser is not None:
        parsed_data = yaml_parser.parse(response)
    
    # Return empty dict if parsing failed
    if parsed_data is None:
        logger.warning("Failed to parse structured data from LLM response")
        return {}
        
    return parsed_data