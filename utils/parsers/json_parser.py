# utils/parsers/json_parser.py
"""
JSON response parsing
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List  # Added List import here

class JSONParser:
    """Utility for parsing JSON from AI model responses"""
    
    def __init__(self):
        """Initialize the JSON parser"""
        self.logger = logging.getLogger(__name__)
    
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from text
        
        Args:
            text (str): Text that may contain JSON
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON or None if parsing failed
        """
        # Try to parse the whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Try to extract JSON from code blocks
        json_blocks = self._extract_json_blocks(text)
        if json_blocks:
            try:
                return json.loads(json_blocks[0])
            except json.JSONDecodeError:
                pass
                
        # Try to extract JSON-like structures and fix common issues
        fixed_json = self._extract_and_fix_json(text)
        if fixed_json:
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
                
        self.logger.warning("Failed to parse JSON from text")
        return None
    
    def _extract_json_blocks(self, text: str) -> List[str]:
        """
        Extract JSON blocks from text
        
        Args:
            text (str): Text to parse
            
        Returns:
            List[str]: Extracted JSON blocks
        """
        # Look for JSON in code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, text)
        
        return matches
    
    def _extract_and_fix_json(self, text: str) -> Optional[str]:
        """
        Extract and fix JSON-like structures
        
        Args:
            text (str): Text to parse
            
        Returns:
            Optional[str]: Fixed JSON or None if no JSON-like structure found
        """
        # Find potential JSON objects (text between braces)
        obj_pattern = r'(\{[\s\S]*\})'
        matches = re.findall(obj_pattern, text)
        
        if not matches:
            return None
            
        # Get the largest match (likely the main result)
        largest_match = max(matches, key=len)
        
        # Fix common JSON issues
        
        # 1. Replace single quotes with double quotes
        fixed = largest_match.replace("'", '"')
        
        # 2. Ensure property names are quoted
        fixed = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', fixed)
        
        # 3. Remove trailing commas
        fixed = re.sub(r',(\s*[\]}])', r'\1', fixed)
        
        return fixed