"""Semantic field mapping using AI assistance"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json

from config.schema import OUTPUT_SCHEMA, FIELD_ORDER, REQUIRED_FIELDS
from core.llm.llm_factory import LLMFactory
from utils.caching.disk_cache import DiskCache
from config.settings import CACHE_DIR

class SemanticMapper:
    """Maps fields using semantic understanding and AI assistance"""
    
    def __init__(self):
        """Initialize the semantic mapper"""
        self.logger = logging.getLogger(__name__)
        self.llm_client = None  # Lazy initialization
        self.cache = DiskCache(f"{CACHE_DIR}/semantic_mapping", ttl=604800)  # 7 days
    
    def map_fields(self, columns: List[str], sample_data: List[Dict[str, Any]], 
                  already_mapped: List[str] = None) -> Dict[str, str]:
        """
        Map input columns to standard schema fields using semantic understanding
        
        Args:
            columns (List[str]): Input column names
            sample_data (List[Dict[str, Any]]): Sample data rows
            already_mapped (List[str], optional): Fields already mapped
            
        Returns:
            Dict[str, str]: Mapping from input columns to standard fields
        """
        already_mapped = already_mapped or []
        self.logger.info(f"Performing semantic mapping for {len(columns)} columns")
        
        # Create cache key from input data
        cache_key = {
            "columns": sorted(columns),
            "already_mapped": sorted(already_mapped),
            "sample_hash": hash(str(sample_data[:3]))  # Use first 3 rows for hash
        }
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.info("Using cached semantic mapping")
            return cached_result
        
        # Initialize LLM client if needed
        if not self.llm_client:
            self.llm_client = LLMFactory.create_client()
        
        # Prepare unmapped fields to focus on
        unmapped_fields = [f for f in FIELD_ORDER if f not in already_mapped]
        required_unmapped = [f for f in REQUIRED_FIELDS if f in unmapped_fields]
        
        # Prepare prompt with schema information and sample data
        prompt = self._create_mapping_prompt(columns, sample_data, unmapped_fields, required_unmapped)
        
        # Get mapping from LLM
        try:
            response = self.llm_client.generate_response(prompt)
            mapping = self._parse_mapping_response(response, columns)
            
            # Cache the result
            self.cache.set(cache_key, mapping)
            
            return mapping
        except Exception as e:
            self.logger.error(f"Error in semantic mapping: {str(e)}")
            return {}
    
    def _create_mapping_prompt(self, columns: List[str], sample_data: List[Dict[str, Any]],
                              unmapped_fields: List[str], required_unmapped: List[str]) -> str:
        """
        Create prompt for the LLM to map fields
        
        Args:
            columns (List[str]): Input column names
            sample_data (List[Dict[str, Any]]): Sample data rows
            unmapped_fields (List[str]): Fields that need mapping
            required_unmapped (List[str]): Required fields that need mapping
            
        Returns:
            str: Formatted prompt
        """
        # Format sample data for display
        sample_str = json.dumps(sample_data[:3], indent=2)
        
        # Format schema information
        schema_info = []
        for field in unmapped_fields:
            field_def = next((f for f in OUTPUT_SCHEMA if f.name == field), None)
            if field_def:
                required = "REQUIRED" if field in required_unmapped else "Optional"
                schema_info.append(f"- {field_def.name} ({required}): {field_def.description}")
                if field_def.examples:
                    schema_info.append(f"  Examples: {', '.join(field_def.examples[:3])}")
                if field_def.mapping_hints:
                    schema_info.append(f"  Common column names: {', '.join(field_def.mapping_hints[:5])}")
        
        schema_str = "\n".join(schema_info)
        
        # Create the prompt
        prompt = f"""
You are a data mapping expert. I need to map columns from a catalog file to a standardized schema.

INPUT COLUMNS:
{', '.join(columns)}

SAMPLE DATA:
{sample_str}

TARGET SCHEMA FIELDS TO MAP:
{schema_str}

Please analyze the input columns and sample data, then map each input column to the most appropriate target field.
Focus especially on the REQUIRED fields. If a column doesn't match any target field, don't map it.

Return your mapping as a JSON object with this format:
{{
  "input_column_name": "Target Field Name",
  ...
}}

Only include mappings you're confident about. Explain your reasoning briefly for each mapping.
"""
        return prompt
    
    def _parse_mapping_response(self, response: str, columns: List[str]) -> Dict[str, str]:
        """
        Parse the LLM response into a mapping dictionary
        
        Args:
            response (str): LLM response text
            columns (List[str]): Original input columns for validation
            
        Returns:
            Dict[str, str]: Mapping from input columns to standard fields
        """
        mapping = {}
        
        # Try to extract JSON from the response
        try:
            # Find JSON block in the response
            import re
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response)
            
            if json_match:
                json_str = json_match.group(1)
                mapping_data = json.loads(json_str)
                
                # Validate the mapping
                for source, target in mapping_data.items():
                    if source in columns and target in FIELD_ORDER:
                        mapping[source] = target
                    else:
                        self.logger.warning(f"Invalid mapping: {source} -> {target}")
            else:
                # Try to find JSON without code block markers
                json_match = re.search(r'{[\s\S]*?}', response)
                if json_match:
                    json_str = json_match.group(0)
                    mapping_data = json.loads(json_str)
                    
                    for source, target in mapping_data.items():
                        if source in columns and target in FIELD_ORDER:
                            mapping[source] = target
                        else:
                            self.logger.warning(f"Invalid mapping: {source} -> {target}")
        except Exception as e:
            self.logger.error(f"Error parsing mapping response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Fallback: try to extract mappings using regex
            try:
                pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
                matches = re.findall(pattern, response)
                
                for source, target in matches:
                    if source in columns and target in FIELD_ORDER:
                        mapping[source] = target
            except Exception as e2:
                self.logger.error(f"Fallback parsing also failed: {str(e2)}")
        
        self.logger.info(f"Extracted {len(mapping)} mappings from semantic analysis")
        return mapping
