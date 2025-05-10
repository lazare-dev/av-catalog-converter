"""Field mapping service for standardizing catalog data"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
import json

from config.schema import OUTPUT_SCHEMA, FIELD_ORDER, REQUIRED_FIELDS, SCHEMA_DICT
from utils.parsers.json_parser import JSONParser
from core.llm.llm_factory import LLMFactory
from services.mapping.direct_mapper import DirectMapper
from services.mapping.semantic_mapper import SemanticMapper
from services.mapping.pattern_mapper import PatternMapper
from prompts.field_mapping import get_field_mapping_prompt

logger = logging.getLogger(__name__)

class FieldMapper:
    """Service for mapping input fields to standardized schema"""
    
    def __init__(self):
        """Initialize the field mapper"""
        self.direct_mapper = DirectMapper()
        self.semantic_mapper = SemanticMapper()
        self.pattern_mapper = PatternMapper()
        self.json_parser = JSONParser()
        self.llm_client = LLMFactory.create_client()
        
        # Store user-provided mappings
        self.user_mappings = {}
        
        # Store last mapping results
        self.last_mapping_results = {}
        
        # Known manufacturer-specific patterns
        self.manufacturer_patterns = self._load_manufacturer_patterns()
    
    def _load_manufacturer_patterns(self) -> Dict[str, Dict[str, str]]:
        """
        Load manufacturer-specific mapping patterns
        
        Returns:
            Dict[str, Dict[str, str]]: Mapping patterns by manufacturer
        """
        try:
            # Try to load from a patterns file
            import os
            pattern_file = os.path.join(os.path.dirname(__file__), 'manufacturer_patterns.json')
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load manufacturer patterns: {str(e)}")
        
        # Return default patterns if file not found
        return {
            "sony": {
                "model_pattern": r"[A-Z]{3}-[A-Z0-9]{5,7}",
                "sku_prefix": "SNY-"
            },
            "samsung": {
                "model_pattern": r"[A-Z]{2}[0-9]{2}[A-Z][0-9]{4,5}",
                "sku_prefix": "SAM-"
            },
            "lg": {
                "model_pattern": r"[0-9]{2}[A-Z]{2}[0-9]{2,3}[A-Z]{1,2}",
                "sku_prefix": "LG-"
            },
            # Add more manufacturers as needed
        }
    
    def map(self, data: pd.DataFrame, structure_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Map input data to standardized schema
        
        Args:
            data (pd.DataFrame): Input data
            structure_info (Dict[str, Any]): Structure analysis results
            
        Returns:
            pd.DataFrame: Data with standardized fields
        """
        logger.info(f"Starting field mapping for {len(data)} rows")
        
        # Detect manufacturer if possible for manufacturer-specific patterns
        manufacturer = self._detect_manufacturer(data)
        logger.info(f"Detected manufacturer: {manufacturer}")
        
        # Multi-stage mapping process
        mapping_results = {}
        
        # Stage 1: Apply user-provided mappings if available
        if self.user_mappings:
            logger.info("Applying user-provided mappings")
            mapping_results = self.user_mappings.copy()
        
        # Stage 2: Direct header matching for unmapped fields
        if len(mapping_results) < len(FIELD_ORDER):
            logger.info("Applying direct header matching")
            direct_mappings = self.direct_mapper.map_fields(
                data.columns, 
                already_mapped=list(mapping_results.values())
            )
            
            # Add direct mappings that don't conflict with user mappings
            for source, target in direct_mappings.items():
                if target not in mapping_results.values():
                    mapping_results[source] = target
        
        # Stage 3: Pattern-based mapping for specific fields
        if len(mapping_results) < len(FIELD_ORDER):
            logger.info("Applying pattern-based mapping")
            pattern_mappings = self.pattern_mapper.map_fields(
                data, 
                manufacturer=manufacturer,
                already_mapped=list(mapping_results.values())
            )
            
            # Add pattern mappings that don't conflict
            for source, target in pattern_mappings.items():
                if target not in mapping_results.values():
                    mapping_results[source] = target
        
        # Stage 4: AI-assisted semantic mapping for remaining fields
        unmapped_required = [f for f in REQUIRED_FIELDS if f not in mapping_results.values()]
        if unmapped_required or len(mapping_results) < len(FIELD_ORDER) * 0.7:  # If less than 70% mapped
            logger.info("Applying AI-assisted semantic mapping")
            
            # Prepare sample data for the model
            sample_rows = min(10, len(data))
            data_sample = data.head(sample_rows).fillna("").to_dict(orient='records')
            
            # Generate
